"""
LLM Service - Unified interface for Large Language Model calls

Supports multiple LLM providers:
- OpenAI (GPT-4, GPT-4o-mini, etc.)
- Mistral API
- Ollama (local models)
- Mock LLM for testing
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Configuration
# ============================================================

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    MOCK = "mock"  # For testing without API


@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama or custom endpoints
    
    def __post_init__(self):
        # Try to get API key from environment
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.MISTRAL:
                self.api_key = os.getenv("MISTRAL_API_KEY")


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    finish_reason: str = ""
    raw_response: Optional[Dict] = None
    
    def __str__(self) -> str:
        return self.content


@dataclass
class Message:
    """Chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    
    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


# ============================================================
# Base LLM Provider
# ============================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def chat(self, 
             messages: List[Message],
             **kwargs) -> LLMResponse:
        """Generate response from chat messages."""
        pass


# ============================================================
# OpenAI Provider
# ============================================================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("llm.openai")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key)
            self._available = True
        except ImportError:
            self.logger.warning("OpenAI package not installed. Install with: pip install openai")
            self._available = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        return self._available and self.config.api_key is not None
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        messages = []
        if system_prompt:
            messages.append(Message("system", system_prompt))
        messages.append(Message("user", prompt))
        return self.chat(messages, **kwargs)
    
    def chat(self, 
             messages: List[Message],
             **kwargs) -> LLMResponse:
        """Generate response from chat messages."""
        if not self.is_available:
            raise RuntimeError("OpenAI provider is not available")
        
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            choice = response.choices[0]
            
            return LLMResponse(
                content=choice.message.content,
                model=model,
                provider="openai",
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=choice.finish_reason,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise


# ============================================================
# Ollama Provider (Local LLMs)
# ============================================================

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        self.logger = logging.getLogger("llm.ollama")
        
        try:
            import httpx
            self._client = httpx.Client(base_url=self.base_url, timeout=60.0)
            self._available = True
        except ImportError:
            try:
                import requests
                self._session = requests.Session()
                self._available = True
            except ImportError:
                self._available = False
    
    @property
    def is_available(self) -> bool:
        if not self._available:
            return False
        try:
            # Check if Ollama is running
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        model = kwargs.get("model", self.config.model)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            import httpx
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data.get("response", ""),
                model=model,
                provider="ollama",
                tokens_used=data.get("eval_count", 0),
                raw_response=data
            )
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            raise
    
    def chat(self, 
             messages: List[Message],
             **kwargs) -> LLMResponse:
        """Generate response from chat messages."""
        model = kwargs.get("model", self.config.model)
        
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False
        }
        
        try:
            import httpx
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=model,
                provider="ollama",
                tokens_used=data.get("eval_count", 0),
                raw_response=data
            )
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            raise


# ============================================================
# Mock Provider (For Testing)
# ============================================================

class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing without API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("llm.mock")
        self._responses: List[str] = []
        self._default_response = "This is a mock response. Configure OPENAI_API_KEY for real responses."
    
    @property
    def is_available(self) -> bool:
        return True
    
    def set_responses(self, responses: List[str]):
        """Set predefined responses for testing."""
        self._responses = responses
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """Generate mock response."""
        content = self._responses.pop(0) if self._responses else self._generate_contextual_response(prompt)
        
        return LLMResponse(
            content=content,
            model="mock-model",
            provider="mock",
            tokens_used=len(content.split())
        )
    
    def chat(self, 
             messages: List[Message],
             **kwargs) -> LLMResponse:
        """Generate mock response from messages."""
        last_message = messages[-1].content if messages else ""
        return self.generate(last_message, **kwargs)
    
    def _generate_contextual_response(self, prompt: str) -> str:
        """Generate a contextual mock response based on prompt."""
        prompt_lower = prompt.lower()
        
        # Detect if this is a RAG query with context
        if "contexte" in prompt_lower or "context" in prompt_lower:
            if "données" in prompt_lower or "data" in prompt_lower:
                return ("D'après le contexte fourni, les données sont protégées par "
                       "des mesures de sécurité incluant le chiffrement TLS 1.3 et "
                       "le stockage dans des bases de données cryptées.")
            elif "sécurité" in prompt_lower or "security" in prompt_lower:
                return ("Selon les informations du contexte, les mesures de sécurité "
                       "comprennent : chiffrement des données, authentification "
                       "multi-facteurs, et audits de sécurité réguliers.")
            else:
                return ("Basé sur le contexte fourni, je peux vous informer que les "
                       "documents décrivent des politiques de protection des données "
                       "et de confidentialité.")
        
        return self._default_response


# ============================================================
# LLM Service (Main Interface)
# ============================================================

class LLMService:
    """
    Unified LLM service supporting multiple providers.
    
    Usage:
        # With OpenAI
        service = LLMService()
        response = service.generate("What is machine learning?")
        
        # With Ollama
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama2")
        service = LLMService(config)
        
        # With chat history
        response = service.chat([
            Message("system", "You are a helpful assistant."),
            Message("user", "What is AI?")
        ])
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.logger = logging.getLogger("llm.service")
        
        # Initialize provider
        self._provider = self._create_provider()
    
    def _create_provider(self) -> BaseLLMProvider:
        """Create the appropriate LLM provider."""
        if self.config.provider == LLMProvider.OPENAI:
            provider = OpenAIProvider(self.config)
            if provider.is_available:
                return provider
            self.logger.warning("OpenAI not available, falling back to mock")
            return MockProvider(self.config)
            
        elif self.config.provider == LLMProvider.OLLAMA:
            provider = OllamaProvider(self.config)
            if provider.is_available:
                return provider
            self.logger.warning("Ollama not available, falling back to mock")
            return MockProvider(self.config)
            
        elif self.config.provider == LLMProvider.MOCK:
            return MockProvider(self.config)
            
        else:
            self.logger.warning(f"Unknown provider: {self.config.provider}, using mock")
            return MockProvider(self.config)
    
    @property
    def provider_name(self) -> str:
        """Get current provider name."""
        return self._provider.__class__.__name__
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._provider.is_available
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate response from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with generated content
        """
        return self._provider.generate(prompt, system_prompt, **kwargs)
    
    def chat(self, 
             messages: List[Message],
             **kwargs) -> LLMResponse:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        return self._provider.chat(messages, **kwargs)
    
    def simple_generate(self, prompt: str) -> str:
        """
        Simple generation returning just the content string.
        
        Args:
            prompt: User prompt
            
        Returns:
            Generated text content
        """
        response = self.generate(prompt)
        return response.content


# ============================================================
# Global Service Instance
# ============================================================

_service_instance: Optional[LLMService] = None


def get_llm_service(config: Optional[LLMConfig] = None) -> LLMService:
    """Get or create global LLM service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LLMService(config)
    return _service_instance


def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Simple function to call LLM.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        
    Returns:
        Generated text
    """
    service = get_llm_service()
    response = service.generate(prompt, system_prompt)
    return response.content


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing LLM Service...")
    print("=" * 60)
    
    # Test with default config (will use mock if no API key)
    service = LLMService()
    
    print(f"\n📦 Provider: {service.provider_name}")
    print(f"🔌 Available: {service.is_available}")
    print(f"🤖 Model: {service.config.model}")
    
    print("\n🧪 Test 1: Simple generation")
    response = service.generate("What is data protection?")
    print(f"   Response: {response.content[:100]}...")
    print(f"   Tokens: {response.tokens_used}")
    
    print("\n🧪 Test 2: With system prompt")
    response = service.generate(
        "What is encryption?",
        system_prompt="You are a security expert. Answer briefly."
    )
    print(f"   Response: {response.content[:100]}...")
    
    print("\n🧪 Test 3: Chat mode")
    messages = [
        Message("system", "Tu es un assistant expert en sécurité informatique."),
        Message("user", "Qu'est-ce que le RGPD ?")
    ]
    response = service.chat(messages)
    print(f"   Response: {response.content[:100]}...")
    
    print("\n🧪 Test 4: RAG-style prompt")
    context = """
    Les données sont protégées par chiffrement TLS 1.3.
    L'accès est restreint au personnel autorisé.
    Des audits de sécurité sont effectués régulièrement.
    """
    prompt = f"""
    CONTEXTE:
    {context}
    
    QUESTION:
    Quelles sont les mesures de sécurité en place ?
    """
    response = service.generate(prompt)
    print(f"   Response: {response.content[:150]}...")
    
    print("\n✅ LLM Service tests complete!")

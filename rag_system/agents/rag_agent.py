"""
RAG Agent - Intelligent agent that answers questions using retrieved context

This agent:
1. Receives a user question
2. Queries MCP for embeddings + retrieval + reranking
3. Builds an enriched prompt with context
4. Queries the LLM
5. Returns a reliable answer based on your documents

Features:
- Complete RAG pipeline via MCP
- Zero hallucination (answers only from documents)
- Full traceability of each step
- Multi-agent ready (can be used by WriterAgent, CriticAgent, etc.)
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcp_client import MCPClient, MCPClientConfig, get_mcp_client
from agents.llm_service import LLMService, LLMConfig, LLMProvider, Message, get_llm_service
from agents.prompts import PromptLibrary, get_prompt_library


# ============================================================
# Configuration
# ============================================================

class RetrievalStrategy(Enum):
    """Retrieval strategy options."""
    SIMPLE = "simple"           # Just retrieve, no rerank
    RERANK = "rerank"           # Retrieve then rerank
    MULTI_QUERY = "multi_query" # Multiple query variations


@dataclass
class RAGConfig:
    """Configuration for RAG Agent."""
    # MCP settings
    mcp_base_url: str = "http://localhost:8000"
    
    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.RERANK
    initial_top_k: int = 10       # Chunks to retrieve before rerank
    final_top_k: int = 5          # Chunks to use after rerank
    rerank_threshold: float = 0.0 # Minimum rerank score
    
    # LLM settings
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    
    # Response settings
    include_sources: bool = True
    include_trace: bool = True
    max_context_chunks: int = 5


@dataclass
class RAGStep:
    """A single step in the RAG pipeline."""
    name: str
    success: bool
    duration_ms: float
    input_summary: str = ""
    output_summary: str = ""
    error: Optional[str] = None


@dataclass
class RAGTrace:
    """Complete trace of a RAG execution."""
    question: str
    steps: List[RAGStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    timestamp: str = ""
    
    def add_step(self, step: RAGStep):
        self.steps.append(step)
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "steps": [asdict(s) for s in self.steps],
            "total_duration_ms": self.total_duration_ms,
            "timestamp": self.timestamp
        }


@dataclass
class RAGResponse:
    """Response from RAG Agent."""
    answer: str
    sources: List[str]
    chunks_used: int
    success: bool
    trace: Optional[RAGTrace] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = {
            "answer": self.answer,
            "sources": self.sources,
            "chunks_used": self.chunks_used,
            "success": self.success
        }
        if self.trace:
            result["trace"] = self.trace.to_dict()
        if self.error:
            result["error"] = self.error
        return result


# ============================================================
# RAG Agent
# ============================================================

class RAGAgent:
    """
    Intelligent RAG Agent that answers questions using document context.
    
    Usage:
        agent = RAGAgent()
        response = agent.answer("What are the security measures?")
        print(response.answer)
        print(response.sources)
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG Agent."""
        self.config = config or RAGConfig()
        self.logger = logging.getLogger("rag.agent")
        
        # Initialize components
        self._mcp_client: Optional[MCPClient] = None
        self._llm_service: Optional[LLMService] = None
        self._prompt_library = get_prompt_library()
        
        # Conversation memory (short-term)
        self._memory: List[Dict] = []
        self._max_memory = 10
    
    # ============================================================
    # Component Access (Lazy Loading)
    # ============================================================
    
    def _get_mcp_client(self) -> MCPClient:
        """Get or create MCP client."""
        if self._mcp_client is None:
            config = MCPClientConfig(base_url=self.config.mcp_base_url)
            self._mcp_client = MCPClient(config)
        return self._mcp_client
    
    def _get_llm_service(self) -> LLMService:
        """Get or create LLM service."""
        if self._llm_service is None:
            config = LLMConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            self._llm_service = LLMService(config)
        return self._llm_service
    
    # ============================================================
    # Main Answer Method
    # ============================================================
    
    def answer(self, 
               question: str,
               source_filter: Optional[str] = None,
               include_trace: Optional[bool] = None) -> RAGResponse:
        """
        Answer a question using RAG pipeline.
        
        Pipeline:
        1. Retrieve chunks via MCP
        2. Rerank chunks via MCP (if strategy = RERANK)
        3. Build context from top chunks
        4. Generate answer via LLM
        
        Args:
            question: User's question
            source_filter: Optional filter for specific source
            include_trace: Whether to include execution trace
            
        Returns:
            RAGResponse with answer, sources, and optional trace
        """
        include_trace = include_trace if include_trace is not None else self.config.include_trace
        
        start_time = time.time()
        trace = RAGTrace(question=question, timestamp=datetime.now().isoformat())
        
        mcp = self._get_mcp_client()
        llm = self._get_llm_service()
        
        try:
            # ========================================
            # Step 1: Retrieve chunks
            # ========================================
            step_start = time.time()
            
            retrieve_result = mcp.retrieve_chunks(
                query=question,
                top_k=self.config.initial_top_k,
                source_filter=source_filter
            )
            
            step_duration = (time.time() - step_start) * 1000
            
            if not retrieve_result.success:
                trace.add_step(RAGStep(
                    name="retrieve",
                    success=False,
                    duration_ms=step_duration,
                    error=retrieve_result.error
                ))
                return self._error_response("Retrieval failed", retrieve_result.error, trace)
            
            chunks = retrieve_result.get("chunks", [])
            
            trace.add_step(RAGStep(
                name="retrieve",
                success=True,
                duration_ms=step_duration,
                input_summary=f"query='{question[:50]}...', top_k={self.config.initial_top_k}",
                output_summary=f"found {len(chunks)} chunks"
            ))
            
            if not chunks:
                # No chunks found
                no_context_response = self._prompt_library.get_no_context_response(question)
                return RAGResponse(
                    answer=no_context_response,
                    sources=[],
                    chunks_used=0,
                    success=True,
                    trace=trace if include_trace else None
                )
            
            # ========================================
            # Step 2: Rerank (if enabled)
            # ========================================
            if self.config.retrieval_strategy == RetrievalStrategy.RERANK and len(chunks) > 1:
                step_start = time.time()
                
                rerank_result = mcp.rerank(
                    query=question,
                    chunks=chunks,
                    top_k=self.config.final_top_k
                )
                
                step_duration = (time.time() - step_start) * 1000
                
                if rerank_result.success:
                    reranked_chunks = rerank_result.get("chunks", [])
                    
                    # Apply threshold filter
                    if self.config.rerank_threshold > 0:
                        reranked_chunks = [
                            c for c in reranked_chunks 
                            if c.get("rerank_score", 0) >= self.config.rerank_threshold
                        ]
                    
                    chunks = reranked_chunks
                    
                    trace.add_step(RAGStep(
                        name="rerank",
                        success=True,
                        duration_ms=step_duration,
                        input_summary=f"{len(retrieve_result.get('chunks', []))} chunks",
                        output_summary=f"top {len(chunks)} after rerank"
                    ))
                else:
                    trace.add_step(RAGStep(
                        name="rerank",
                        success=False,
                        duration_ms=step_duration,
                        error=rerank_result.error
                    ))
                    # Continue with unreranked chunks
            
            # ========================================
            # Step 3: Build context
            # ========================================
            step_start = time.time()
            
            # Limit chunks for context
            context_chunks = chunks[:self.config.max_context_chunks]
            
            context, sources = self._prompt_library.format_context_from_chunks(context_chunks)
            
            step_duration = (time.time() - step_start) * 1000
            
            trace.add_step(RAGStep(
                name="build_context",
                success=True,
                duration_ms=step_duration,
                input_summary=f"{len(context_chunks)} chunks",
                output_summary=f"context={len(context)} chars, sources={sources}"
            ))
            
            # ========================================
            # Step 4: Generate answer via LLM
            # ========================================
            step_start = time.time()
            
            # Build prompt
            system_prompt = self._prompt_library.get_system_prompt()
            user_prompt = self._prompt_library.get_rag_prompt(
                context=context,
                question=question,
                sources=sources
            )
            
            # Call LLM
            llm_response = llm.generate(user_prompt, system_prompt)
            
            step_duration = (time.time() - step_start) * 1000
            
            trace.add_step(RAGStep(
                name="llm_generate",
                success=True,
                duration_ms=step_duration,
                input_summary=f"prompt={len(user_prompt)} chars",
                output_summary=f"response={len(llm_response.content)} chars, tokens={llm_response.tokens_used}"
            ))
            
            # ========================================
            # Build final response
            # ========================================
            total_duration = (time.time() - start_time) * 1000
            trace.total_duration_ms = total_duration
            
            # Save to memory
            self._add_to_memory(question, llm_response.content, sources)
            
            return RAGResponse(
                answer=llm_response.content,
                sources=sources if self.config.include_sources else [],
                chunks_used=len(context_chunks),
                success=True,
                trace=trace if include_trace else None
            )
            
        except Exception as e:
            self.logger.exception(f"RAG pipeline error: {e}")
            return self._error_response("Pipeline error", str(e), trace)
    
    # ============================================================
    # Direct Tool Access Methods
    # ============================================================
    
    def retrieve_only(self, 
                      question: str, 
                      top_k: int = 5,
                      source_filter: Optional[str] = None) -> List[Dict]:
        """
        Just retrieve chunks without LLM generation.
        Useful for debugging or custom pipelines.
        """
        mcp = self._get_mcp_client()
        result = mcp.retrieve_chunks(question, top_k, source_filter=source_filter)
        return result.get("chunks", []) if result.success else []
    
    def retrieve_and_rerank(self,
                            question: str,
                            initial_k: int = 10,
                            final_k: int = 5) -> List[Dict]:
        """
        Retrieve and rerank without LLM generation.
        """
        mcp = self._get_mcp_client()
        
        # Retrieve
        retrieve_result = mcp.retrieve_chunks(question, initial_k)
        if not retrieve_result.success:
            return []
        
        chunks = retrieve_result.get("chunks", [])
        if not chunks:
            return []
        
        # Rerank
        rerank_result = mcp.rerank(question, chunks, final_k)
        return rerank_result.get("chunks", []) if rerank_result.success else chunks[:final_k]
    
    # ============================================================
    # Memory Management
    # ============================================================
    
    def _add_to_memory(self, question: str, answer: str, sources: List[str]):
        """Add Q&A to memory."""
        self._memory.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim memory if too large
        if len(self._memory) > self._max_memory:
            self._memory = self._memory[-self._max_memory:]
    
    def get_memory(self) -> List[Dict]:
        """Get conversation memory."""
        return self._memory.copy()
    
    def clear_memory(self):
        """Clear conversation memory."""
        self._memory.clear()
    
    # ============================================================
    # Helper Methods
    # ============================================================
    
    def _error_response(self, 
                        message: str, 
                        error: str,
                        trace: RAGTrace) -> RAGResponse:
        """Build error response."""
        return RAGResponse(
            answer=f"Désolé, une erreur s'est produite: {message}",
            sources=[],
            chunks_used=0,
            success=False,
            trace=trace if self.config.include_trace else None,
            error=error
        )
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        mcp = self._get_mcp_client()
        llm = self._get_llm_service()
        
        return {
            "mcp_server": mcp.health_check(),
            "llm_service": llm.is_available
        }
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        mcp = self._get_mcp_client()
        
        return {
            "mcp_stats": mcp.get_stats(),
            "memory_size": len(self._memory),
            "config": {
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "llm_model": self.config.llm_model
            }
        }


# ============================================================
# Convenience Functions
# ============================================================

_agent_instance: Optional[RAGAgent] = None


def get_rag_agent(config: Optional[RAGConfig] = None) -> RAGAgent:
    """Get or create global RAG agent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RAGAgent(config)
    return _agent_instance


def rag_answer(question: str, **kwargs) -> str:
    """
    Simple function to get RAG answer.
    
    Args:
        question: User question
        **kwargs: Additional options
        
    Returns:
        Answer string
    """
    agent = get_rag_agent()
    response = agent.answer(question, **kwargs)
    return response.answer


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🤖 RAG Agent Test")
    print("=" * 60)
    
    # Check if MCP server is running
    agent = RAGAgent()
    health = agent.health_check()
    
    print(f"\n🔍 Health Check:")
    print(f"   MCP Server: {'✅' if health['mcp_server'] else '❌'}")
    print(f"   LLM Service: {'✅' if health['llm_service'] else '❌ (using mock)'}")
    
    if not health['mcp_server']:
        print("\n⚠️  MCP Server not running!")
        print("   Start with: cd mcp_server && uvicorn main:app --reload")
        exit(1)
    
    # Test questions
    questions = [
        "Quelles sont les mesures de sécurité pour protéger les données ?",
        "Comment fonctionne le chiffrement des données ?",
        "Quels sont mes droits concernant mes données personnelles ?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 Test {i}: {question}")
        print("-" * 60)
        
        response = agent.answer(question)
        
        print(f"\n📖 Réponse:")
        print(f"   {response.answer[:300]}...")
        
        print(f"\n📚 Sources: {response.sources}")
        print(f"📊 Chunks utilisés: {response.chunks_used}")
        print(f"✅ Succès: {response.success}")
        
        if response.trace:
            print(f"\n🔍 Trace:")
            for step in response.trace.steps:
                status = "✅" if step.success else "❌"
                print(f"   {status} {step.name}: {step.duration_ms:.1f}ms - {step.output_summary}")
            print(f"   ⏱️  Total: {response.trace.total_duration_ms:.1f}ms")
    
    # Show stats
    print(f"\n{'='*60}")
    print("📈 Agent Stats:")
    stats = agent.get_stats()
    print(f"   MCP calls: {stats['mcp_stats']['total_calls']}")
    print(f"   Memory size: {stats['memory_size']}")
    
    print("\n✅ RAG Agent tests complete!")

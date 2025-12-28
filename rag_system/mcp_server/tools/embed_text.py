"""
MCP Tool: embed_text

Provides text embedding functionality via Model Context Protocol.
Agents can request embeddings for any text through this tool.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class EmbedTextRequest:
    """Request schema for embed_text tool."""
    text: str
    model: Optional[str] = None  # Override default model
    return_format: str = "list"  # 'list', 'base64', or 'json'


@dataclass
class EmbedTextResponse:
    """Response schema for embed_text tool."""
    success: bool
    vector: Optional[List[float]] = None
    dimension: Optional[int] = None
    model: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = {"success": self.success}
        if self.success:
            result["vector"] = self.vector
            result["dimension"] = self.dimension
            result["model"] = self.model
        else:
            result["error"] = self.error
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EmbedTextTool:
    """
    MCP Tool for generating text embeddings.
    
    Usage via MCP:
        {
            "tool": "embed_text",
            "params": {
                "text": "AI agents are autonomous systems...",
                "model": "all-MiniLM-L6-v2"  // optional
            }
        }
    
    Response:
        {
            "success": true,
            "vector": [0.023, -0.156, ...],
            "dimension": 384,
            "model": "all-MiniLM-L6-v2"
        }
    """
    
    # Tool metadata for MCP registry
    TOOL_NAME = "embed_text"
    TOOL_DESCRIPTION = "Generate vector embeddings for text. Returns a numerical vector representation suitable for semantic search."
    
    TOOL_SCHEMA = {
        "name": "embed_text",
        "description": "Generate vector embeddings for text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to embed"
                },
                "model": {
                    "type": "string",
                    "description": "Optional: embedding model to use",
                    "default": "all-MiniLM-L6-v2"
                }
            },
            "required": ["text"]
        }
    }
    
    def __init__(self, 
                 model_type: str = "sentence-transformers",
                 model_name: str = None,
                 cache_enabled: bool = True):
        """
        Initialize the embed_text tool.
        
        Args:
            model_type: 'sentence-transformers' or 'openai'
            model_name: Specific model name
            cache_enabled: Whether to cache embeddings
        """
        self.model_type = model_type
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self._embedding_service = None
    
    def _get_embedding_service(self):
        """Lazy-load the embedding service."""
        if self._embedding_service is None:
            from embeddings.embedding_models import get_embedding_service
            self._embedding_service = get_embedding_service(
                model_type=self.model_type,
                model_name=self.model_name
            )
        return self._embedding_service
    
    def execute(self, params: Dict[str, Any]) -> EmbedTextResponse:
        """
        Execute the embed_text tool.
        
        Args:
            params: Tool parameters with 'text' field
        
        Returns:
            EmbedTextResponse with vector or error
        """
        # Validate input
        text = params.get("text")
        if not text:
            return EmbedTextResponse(
                success=False,
                error="Missing required parameter: 'text'"
            )
        
        if not isinstance(text, str):
            return EmbedTextResponse(
                success=False,
                error="Parameter 'text' must be a string"
            )
        
        if len(text.strip()) == 0:
            return EmbedTextResponse(
                success=False,
                error="Parameter 'text' cannot be empty"
            )
        
        try:
            # Get embedding service
            service = self._get_embedding_service()
            
            # Generate embedding
            vector = service.embed(text)
            
            return EmbedTextResponse(
                success=True,
                vector=vector.tolist(),
                dimension=len(vector),
                model=service.model_name
            )
            
        except Exception as e:
            return EmbedTextResponse(
                success=False,
                error=f"Embedding failed: {str(e)}"
            )
    
    def execute_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Execute batch embedding for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Dictionary with 'vectors' and metadata
        """
        try:
            service = self._get_embedding_service()
            vectors = service.embed_batch(texts, show_progress=False)
            
            return {
                "success": True,
                "vectors": vectors.tolist(),
                "count": len(texts),
                "dimension": service.dimension,
                "model": service.model_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allow calling the tool as a function."""
        return self.execute(params).to_dict()
    
    @classmethod
    def get_schema(cls) -> Dict:
        """Return the tool schema for MCP registration."""
        return cls.TOOL_SCHEMA


# Singleton instance for easy access
_tool_instance: Optional[EmbedTextTool] = None


def get_embed_text_tool() -> EmbedTextTool:
    """Get or create the singleton EmbedTextTool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = EmbedTextTool()
    return _tool_instance


def embed_text(text: str, model: str = None) -> Union[List[float], Dict]:
    """
    Convenience function to embed text.
    
    Args:
        text: Text to embed
        model: Optional model name
    
    Returns:
        Vector as list of floats, or error dict
    """
    tool = get_embed_text_tool()
    result = tool.execute({"text": text, "model": model})
    
    if result.success:
        return result.vector
    else:
        return {"error": result.error}


# MCP Handler
def handle_mcp_request(request: Dict) -> Dict:
    """
    Handle an MCP request for embed_text.
    
    Args:
        request: MCP request with 'tool' and 'params'
    
    Returns:
        MCP response dictionary
    """
    tool_name = request.get("tool")
    
    if tool_name != "embed_text":
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    params = request.get("params", {})
    tool = get_embed_text_tool()
    
    return tool(params)


if __name__ == "__main__":
    # Test the tool
    print("Testing embed_text MCP tool...")
    
    tool = EmbedTextTool()
    
    # Test single embedding
    result = tool.execute({"text": "AI agents are autonomous systems that can perform tasks."})
    
    if result.success:
        print(f"✅ Embedding successful!")
        print(f"   Model: {result.model}")
        print(f"   Dimension: {result.dimension}")
        print(f"   Vector (first 5): {result.vector[:5]}")
    else:
        print(f"❌ Error: {result.error}")
    
    # Test MCP request format
    print("\n--- Testing MCP format ---")
    mcp_request = {
        "tool": "embed_text",
        "params": {"text": "Test embedding via MCP"}
    }
    
    response = handle_mcp_request(mcp_request)
    print(f"MCP Response success: {response['success']}")
    if response['success']:
        print(f"   Dimension: {response['dimension']}")
    
    # Test convenience function
    print("\n--- Testing convenience function ---")
    vector = embed_text("Quick test")
    if isinstance(vector, list):
        print(f"✅ Got vector with {len(vector)} dimensions")
    else:
        print(f"❌ Error: {vector}")

"""
MCP Tool: retrieve_chunks

Retrieves the most relevant chunks for a query using semantic search.
Uses the FAISS vector store created in Phase 2.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    score: float
    source: str = ""
    section: str = ""
    page: Optional[int] = None
    rank: int = 0
    
    def to_dict(self) -> Dict:
        result = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "rank": self.rank
        }
        if self.section:
            result["section"] = self.section
        if self.page is not None:
            result["page"] = self.page
        return result


@dataclass
class RetrieveChunksRequest:
    """Request schema for retrieve_chunks tool."""
    query: str
    top_k: int = 5
    threshold: Optional[float] = None  # Min similarity threshold
    source_filter: Optional[str] = None  # Filter by source file
    section_filter: Optional[str] = None  # Filter by section


@dataclass
class RetrieveChunksResponse:
    """Response schema for retrieve_chunks tool."""
    success: bool
    query: str = ""
    chunks: List[Dict] = field(default_factory=list)
    total_found: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        if self.success:
            return {
                "success": True,
                "query": self.query,
                "chunks": self.chunks,
                "total_found": self.total_found
            }
        else:
            return {
                "success": False,
                "error": self.error
            }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RetrieveChunksTool:
    """
    MCP Tool for retrieving relevant chunks via semantic search.
    
    Usage via MCP:
        {
            "tool": "retrieve_chunks",
            "params": {
                "query": "AI agents are autonomous",
                "top_k": 5,
                "threshold": 0.5,
                "source_filter": "privacy_policy.md"
            }
        }
    
    Response:
        {
            "success": true,
            "query": "AI agents are autonomous",
            "chunks": [
                {
                    "chunk_id": "doc1_001",
                    "text": "...",
                    "score": 0.87,
                    "source": "privacy_policy.md",
                    "section": "User Data",
                    "rank": 1
                }
            ],
            "total_found": 5
        }
    """
    
    # Tool metadata for MCP registry
    TOOL_NAME = "retrieve_chunks"
    TOOL_DESCRIPTION = "Retrieve the most relevant text chunks for a query using semantic search."
    
    TOOL_SCHEMA = {
        "name": "retrieve_chunks",
        "description": "Retrieve relevant chunks from the knowledge base using semantic search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity score (0-1 for cosine, or distance for L2)",
                    "default": None
                },
                "source_filter": {
                    "type": "string",
                    "description": "Filter results by source file",
                    "default": None
                }
            },
            "required": ["query"]
        }
    }
    
    def __init__(self, 
                 vector_store_path: str = None,
                 embedding_model_type: str = "sentence-transformers",
                 embedding_model_name: str = None):
        """
        Initialize the retrieve_chunks tool.
        
        Args:
            vector_store_path: Path to vector_store directory
            embedding_model_type: 'sentence-transformers' or 'openai'
            embedding_model_name: Specific model name
        """
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name
        
        # Determine vector store path
        if vector_store_path:
            self.vector_store_path = Path(vector_store_path)
        else:
            # Default: relative to rag_system/
            base_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.vector_store_path = base_path / "vector_store"
        
        # Lazy-loaded components
        self._vector_store = None
        self._embedding_service = None
    
    def _get_embedding_service(self):
        """Lazy-load the embedding service."""
        if self._embedding_service is None:
            from embeddings.embedding_models import get_embedding_service
            self._embedding_service = get_embedding_service(
                model_type=self.embedding_model_type,
                model_name=self.embedding_model_name
            )
        return self._embedding_service
    
    def _get_vector_store(self):
        """Lazy-load the vector store."""
        if self._vector_store is None:
            from vector_store.faiss_store import FAISSVectorStore
            
            if not self.vector_store_path.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {self.vector_store_path}. "
                    "Run the indexing pipeline first."
                )
            
            self._vector_store = FAISSVectorStore.load(str(self.vector_store_path))
        return self._vector_store
    
    def execute(self, params: Dict[str, Any]) -> RetrieveChunksResponse:
        """
        Execute the retrieve_chunks tool.
        
        Args:
            params: Tool parameters with 'query' and optional 'top_k', 'threshold', etc.
        
        Returns:
            RetrieveChunksResponse with retrieved chunks
        """
        # Validate input
        query = params.get("query")
        if not query:
            return RetrieveChunksResponse(
                success=False,
                error="Missing required parameter: 'query'"
            )
        
        if not isinstance(query, str):
            return RetrieveChunksResponse(
                success=False,
                error="Parameter 'query' must be a string"
            )
        
        if len(query.strip()) == 0:
            return RetrieveChunksResponse(
                success=False,
                error="Parameter 'query' cannot be empty"
            )
        
        # Get optional parameters
        top_k = params.get("top_k", 5)
        threshold = params.get("threshold")
        source_filter = params.get("source_filter")
        section_filter = params.get("section_filter")
        
        try:
            # Get components
            embedding_service = self._get_embedding_service()
            vector_store = self._get_vector_store()
            
            # Build filter function if needed
            filter_fn = None
            if source_filter or section_filter:
                def filter_fn(meta):
                    if source_filter and meta.get("source", "") != source_filter:
                        return False
                    if section_filter and section_filter.lower() not in meta.get("section", "").lower():
                        return False
                    return True
            
            # Search
            results = vector_store.search_by_text(
                query_text=query,
                embedding_service=embedding_service,
                k=top_k * 2 if filter_fn else top_k  # Get more if filtering
            )
            
            # Apply filter if provided
            if filter_fn:
                results = [r for r in results if filter_fn({
                    "source": r.source,
                    "section": r.section
                })][:top_k]
            
            # Apply threshold if provided
            if threshold is not None:
                results = [r for r in results if r.score <= threshold]
            
            # Convert to response format
            chunks = []
            for rank, result in enumerate(results, 1):
                chunk = RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=result.score,
                    source=result.source,
                    section=result.section,
                    page=result.page,
                    rank=rank
                )
                chunks.append(chunk.to_dict())
            
            return RetrieveChunksResponse(
                success=True,
                query=query,
                chunks=chunks,
                total_found=len(chunks)
            )
            
        except FileNotFoundError as e:
            return RetrieveChunksResponse(
                success=False,
                error=str(e)
            )
        except Exception as e:
            return RetrieveChunksResponse(
                success=False,
                error=f"Retrieval failed: {str(e)}"
            )
    
    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allow calling the tool as a function."""
        return self.execute(params).to_dict()
    
    @classmethod
    def get_schema(cls) -> Dict:
        """Return the tool schema for MCP registration."""
        return cls.TOOL_SCHEMA
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        try:
            store = self._get_vector_store()
            return {
                "status": "loaded",
                "total_chunks": store.size,
                "dimension": store.dimension,
                "model": store.model_name,
                "index_type": store.index_type
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Singleton instance for easy access
_tool_instance: Optional[RetrieveChunksTool] = None


def get_retrieve_chunks_tool() -> RetrieveChunksTool:
    """Get or create the singleton RetrieveChunksTool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = RetrieveChunksTool()
    return _tool_instance


def retrieve_chunks(query: str, 
                    top_k: int = 5,
                    threshold: float = None,
                    source_filter: str = None) -> Union[List[Dict], Dict]:
    """
    Convenience function to retrieve chunks.
    
    Args:
        query: Search query
        top_k: Number of results
        threshold: Minimum similarity threshold
        source_filter: Filter by source file
    
    Returns:
        List of chunk dictionaries, or error dict
    """
    tool = get_retrieve_chunks_tool()
    result = tool.execute({
        "query": query,
        "top_k": top_k,
        "threshold": threshold,
        "source_filter": source_filter
    })
    
    if result.success:
        return result.chunks
    else:
        return {"error": result.error}


# MCP Handler
def handle_mcp_request(request: Dict) -> Dict:
    """
    Handle an MCP request for retrieve_chunks.
    
    Args:
        request: MCP request with 'tool' and 'params'
    
    Returns:
        MCP response dictionary
    """
    tool_name = request.get("tool")
    
    if tool_name != "retrieve_chunks":
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    params = request.get("params", {})
    tool = get_retrieve_chunks_tool()
    
    return tool(params)


if __name__ == "__main__":
    # Test the tool
    print("Testing retrieve_chunks MCP tool...")
    print("=" * 60)
    
    tool = RetrieveChunksTool()
    
    # Test 1: Basic retrieval
    print("\n📝 Test 1: Basic retrieval")
    result = tool.execute({
        "query": "data protection and privacy",
        "top_k": 3
    })
    
    if result.success:
        print(f"✅ Retrieved {result.total_found} chunks")
        for chunk in result.chunks:
            print(f"\n   [{chunk['rank']}] {chunk['chunk_id']} (score: {chunk['score']})")
            print(f"       Source: {chunk['source']}")
            print(f"       Text: {chunk['text'][:80]}...")
    else:
        print(f"❌ Error: {result.error}")
    
    # Test 2: With source filter
    print("\n" + "=" * 60)
    print("\n📝 Test 2: With source filter")
    result = tool.execute({
        "query": "security encryption",
        "top_k": 3,
        "source_filter": "notes.txt"
    })
    
    if result.success:
        print(f"✅ Retrieved {result.total_found} chunks from notes.txt")
        for chunk in result.chunks:
            print(f"   [{chunk['rank']}] {chunk['source']}: {chunk['text'][:60]}...")
    else:
        print(f"❌ Error: {result.error}")
    
    # Test 3: MCP request format
    print("\n" + "=" * 60)
    print("\n📝 Test 3: MCP request format")
    mcp_request = {
        "tool": "retrieve_chunks",
        "params": {
            "query": "financial report revenue",
            "top_k": 2
        }
    }
    
    response = handle_mcp_request(mcp_request)
    print(f"MCP Response success: {response['success']}")
    if response['success']:
        print(f"   Found {response['total_found']} chunks")
    
    # Test 4: Convenience function
    print("\n" + "=" * 60)
    print("\n📝 Test 4: Convenience function")
    chunks = retrieve_chunks("GDPR compliance", top_k=2)
    if isinstance(chunks, list):
        print(f"✅ Got {len(chunks)} chunks via convenience function")
    else:
        print(f"❌ Error: {chunks}")
    
    # Show stats
    print("\n" + "=" * 60)
    print("\n📊 Vector Store Stats:")
    stats = tool.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

"""
MCP Tool: rerank

Reranks retrieved chunks using a cross-encoder model for improved relevance.
Cross-encoders evaluate query-document pairs together, providing more accurate
relevance scores than bi-encoder similarity.
"""

import sys
import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class RerankRequest:
    """Request schema for rerank tool."""
    query: str
    chunks: List[Dict]  # List of chunks with 'text' field
    top_k: Optional[int] = None  # Limit results after reranking


@dataclass 
class RerankedChunk:
    """A reranked chunk with updated scores."""
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    source: str = ""
    section: str = ""
    rank: int = 0
    
    def to_dict(self) -> Dict:
        result = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "original_score": round(self.original_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "rank": self.rank
        }
        if self.source:
            result["source"] = self.source
        if self.section:
            result["section"] = self.section
        return result


@dataclass
class RerankResponse:
    """Response schema for rerank tool."""
    success: bool
    query: str = ""
    chunks: List[Dict] = field(default_factory=list)
    model: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        if self.success:
            return {
                "success": True,
                "query": self.query,
                "chunks": self.chunks,
                "model": self.model
            }
        else:
            return {
                "success": False,
                "error": self.error
            }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class CrossEncoderReranker:
    """
    Reranker using Cross-Encoder models from SentenceTransformers.
    
    Cross-encoders process query and document together, providing
    more accurate relevance scores than bi-encoder similarity.
    
    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality
    - cross-encoder/ms-marco-TinyBERT-L-2-v2: Very fast, smaller
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Higher quality
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: str = None):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"📦 Loading cross-encoder: {self._model_name}...")
                self._model = CrossEncoder(self._model_name)
                print(f"✅ Cross-encoder loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(self, 
               query: str, 
               chunks: List[Dict],
               top_k: int = None) -> List[Dict]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: The search query
            chunks: List of chunks with 'text' field
            top_k: Optional limit on returned results
        
        Returns:
            List of chunks with rerank_score, sorted by relevance
        """
        if not chunks:
            return []
        
        model = self._load_model()
        
        # Create query-document pairs
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        
        # Get relevance scores
        scores = model.predict(pairs)
        
        # Add rerank scores to chunks
        reranked = []
        for chunk, score in zip(chunks, scores):
            reranked_chunk = chunk.copy()
            reranked_chunk["rerank_score"] = float(score)
            reranked_chunk["original_score"] = chunk.get("score", 0.0)
            reranked.append(reranked_chunk)
        
        # Sort by rerank score (higher is better for cross-encoder)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Update ranks
        for i, chunk in enumerate(reranked, 1):
            chunk["rank"] = i
        
        # Apply top_k limit if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
    
    @property
    def model_name(self) -> str:
        return self._model_name


class RerankTool:
    """
    MCP Tool for reranking retrieved chunks.
    
    Usage via MCP:
        {
            "tool": "rerank",
            "params": {
                "query": "AI agents are autonomous",
                "chunks": [
                    {"text": "...", "score": 0.87, "chunk_id": "doc1_001"},
                    {"text": "...", "score": 0.82, "chunk_id": "doc1_002"}
                ],
                "top_k": 5
            }
        }
    
    Response:
        {
            "success": true,
            "query": "AI agents are autonomous",
            "chunks": [
                {
                    "chunk_id": "doc1_002",
                    "text": "...",
                    "original_score": 0.82,
                    "rerank_score": 0.91,
                    "rank": 1
                }
            ],
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
    """
    
    # Tool metadata for MCP registry
    TOOL_NAME = "rerank"
    TOOL_DESCRIPTION = "Rerank retrieved chunks using a cross-encoder for improved relevance scoring."
    
    TOOL_SCHEMA = {
        "name": "rerank",
        "description": "Rerank chunks using cross-encoder for better relevance",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "chunks": {
                    "type": "array",
                    "description": "List of chunks to rerank (must have 'text' field)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "score": {"type": "number"},
                            "chunk_id": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return after reranking",
                    "default": None
                }
            },
            "required": ["query", "chunks"]
        }
    }
    
    def __init__(self, model_name: str = None):
        """
        Initialize the rerank tool.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.reranker = CrossEncoderReranker(model_name)
    
    def execute(self, params: Dict[str, Any]) -> RerankResponse:
        """
        Execute the rerank tool.
        
        Args:
            params: Tool parameters with 'query' and 'chunks'
        
        Returns:
            RerankResponse with reranked chunks
        """
        # Validate query
        query = params.get("query")
        if not query:
            return RerankResponse(
                success=False,
                error="Missing required parameter: 'query'"
            )
        
        if not isinstance(query, str) or len(query.strip()) == 0:
            return RerankResponse(
                success=False,
                error="Parameter 'query' must be a non-empty string"
            )
        
        # Validate chunks
        chunks = params.get("chunks")
        if not chunks:
            return RerankResponse(
                success=False,
                error="Missing required parameter: 'chunks'"
            )
        
        if not isinstance(chunks, list):
            return RerankResponse(
                success=False,
                error="Parameter 'chunks' must be a list"
            )
        
        if len(chunks) == 0:
            return RerankResponse(
                success=True,
                query=query,
                chunks=[],
                model=self.reranker.model_name
            )
        
        # Validate each chunk has 'text'
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                return RerankResponse(
                    success=False,
                    error=f"Chunk at index {i} must be a dictionary"
                )
            if "text" not in chunk:
                return RerankResponse(
                    success=False,
                    error=f"Chunk at index {i} missing required 'text' field"
                )
        
        # Get optional parameters
        top_k = params.get("top_k")
        
        try:
            # Rerank
            reranked = self.reranker.rerank(query, chunks, top_k)
            
            return RerankResponse(
                success=True,
                query=query,
                chunks=reranked,
                model=self.reranker.model_name
            )
            
        except Exception as e:
            return RerankResponse(
                success=False,
                error=f"Reranking failed: {str(e)}"
            )
    
    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allow calling the tool as a function."""
        return self.execute(params).to_dict()
    
    @classmethod
    def get_schema(cls) -> Dict:
        """Return the tool schema for MCP registration."""
        return cls.TOOL_SCHEMA


# Singleton instance
_tool_instance: Optional[RerankTool] = None


def get_rerank_tool() -> RerankTool:
    """Get or create the singleton RerankTool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = RerankTool()
    return _tool_instance


def rerank(query: str, 
           chunks: List[Dict], 
           top_k: int = None) -> Union[List[Dict], Dict]:
    """
    Convenience function to rerank chunks.
    
    Args:
        query: Search query
        chunks: List of chunks with 'text' field
        top_k: Optional limit on results
    
    Returns:
        List of reranked chunks, or error dict
    """
    tool = get_rerank_tool()
    result = tool.execute({
        "query": query,
        "chunks": chunks,
        "top_k": top_k
    })
    
    if result.success:
        return result.chunks
    else:
        return {"error": result.error}


# MCP Handler
def handle_mcp_request(request: Dict) -> Dict:
    """
    Handle an MCP request for rerank.
    
    Args:
        request: MCP request with 'tool' and 'params'
    
    Returns:
        MCP response dictionary
    """
    tool_name = request.get("tool")
    
    if tool_name != "rerank":
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    params = request.get("params", {})
    tool = get_rerank_tool()
    
    return tool(params)


if __name__ == "__main__":
    # Test the rerank tool
    print("Testing rerank MCP tool...")
    print("=" * 60)
    
    # Simulated chunks from retrieve_chunks
    test_chunks = [
        {
            "chunk_id": "privacy_policy_002",
            "text": "We implement robust security measures to protect your data: All data is encrypted in transit using TLS 1.3. Personal information is stored in encrypted databases.",
            "score": 0.64,
            "source": "privacy_policy.md",
            "section": "Data Protection Measures"
        },
        {
            "chunk_id": "privacy_policy_000",
            "text": "This privacy policy explains how we collect, use, and protect your personal information when you use our services.",
            "score": 0.71,
            "source": "privacy_policy.md"
        },
        {
            "chunk_id": "notes_006",
            "text": "All services are containerized using Docker and orchestrated with Kubernetes. Security is implemented at multiple layers including network, application, and data levels.",
            "score": 0.85,
            "source": "notes.txt",
            "section": "Deployment Architecture"
        },
        {
            "chunk_id": "finance_report_000",
            "text": "This report presents the financial performance for the fourth quarter of 2024. Overall, the company has demonstrated strong growth.",
            "score": 0.92,
            "source": "finance_report.txt"
        }
    ]
    
    tool = RerankTool()
    
    # Test 1: Basic reranking
    print("\n📝 Test 1: Rerank for 'data protection security'")
    result = tool.execute({
        "query": "data protection security measures",
        "chunks": test_chunks
    })
    
    if result.success:
        print(f"✅ Reranked {len(result.chunks)} chunks")
        print(f"   Model: {result.model}")
        print("\n   Results (sorted by rerank_score):")
        for chunk in result.chunks:
            print(f"\n   [{chunk['rank']}] {chunk['chunk_id']}")
            print(f"       Original score: {chunk['original_score']}")
            print(f"       Rerank score: {chunk['rerank_score']:.4f}")
            print(f"       Text: {chunk['text'][:60]}...")
    else:
        print(f"❌ Error: {result.error}")
    
    # Test 2: With top_k limit
    print("\n" + "=" * 60)
    print("\n📝 Test 2: Rerank with top_k=2")
    result = tool.execute({
        "query": "encryption and security",
        "chunks": test_chunks,
        "top_k": 2
    })
    
    if result.success:
        print(f"✅ Got top {len(result.chunks)} chunks after reranking")
        for chunk in result.chunks:
            print(f"   [{chunk['rank']}] {chunk['chunk_id']} (rerank: {chunk['rerank_score']:.4f})")
    
    # Test 3: MCP request format
    print("\n" + "=" * 60)
    print("\n📝 Test 3: MCP request format")
    mcp_request = {
        "tool": "rerank",
        "params": {
            "query": "financial performance revenue",
            "chunks": test_chunks[:2],
            "top_k": 2
        }
    }
    
    response = handle_mcp_request(mcp_request)
    print(f"MCP Response success: {response['success']}")
    if response['success']:
        print(f"   Model: {response['model']}")
        print(f"   Chunks reranked: {len(response['chunks'])}")
    
    # Test 4: Convenience function
    print("\n" + "=" * 60)
    print("\n📝 Test 4: Convenience function")
    reranked = rerank("privacy policy personal data", test_chunks, top_k=3)
    if isinstance(reranked, list):
        print(f"✅ Got {len(reranked)} reranked chunks via convenience function")
        for chunk in reranked:
            print(f"   [{chunk['rank']}] rerank_score: {chunk['rerank_score']:.4f}")
    else:
        print(f"❌ Error: {reranked}")

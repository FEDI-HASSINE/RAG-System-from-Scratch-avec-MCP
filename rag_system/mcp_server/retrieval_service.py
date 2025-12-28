"""
Retrieval Service - Unified interface for semantic search

This service provides a high-level API for retrieving relevant chunks
from the vector store, with support for:
- Semantic search
- Filtering by source/section
- Threshold-based filtering
- Result formatting for agents
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_models import get_embedding_service, EmbeddingService
from vector_store.faiss_store import FAISSVectorStore, SearchResult


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval service."""
    vector_store_path: str = "vector_store"
    embedding_model_type: str = "sentence-transformers"
    embedding_model_name: Optional[str] = None
    default_top_k: int = 5
    default_threshold: Optional[float] = None  # No threshold by default


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    chunks: List[Dict]
    total_found: int
    search_time_ms: float = 0.0
    
    def to_context_string(self, max_chunks: Optional[int] = None) -> str:
        """
        Convert results to a context string for LLM.
        
        Args:
            max_chunks: Maximum chunks to include
        
        Returns:
            Formatted context string
        """
        chunks = self.chunks[:max_chunks] if max_chunks else self.chunks
        
        context_parts = []
        for chunk in chunks:
            part = f"[Source: {chunk.get('source', 'unknown')}"
            if chunk.get('section'):
                part += f" | Section: {chunk['section']}"
            part += f"]\n{chunk['text']}"
            context_parts.append(part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "chunks": self.chunks,
            "total_found": self.total_found,
            "search_time_ms": self.search_time_ms
        }


class RetrievalService:
    """
    High-level retrieval service for RAG systems.
    
    Usage:
        service = RetrievalService()
        results = service.search("What is data protection?")
        context = results.to_context_string()
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize the retrieval service."""
        self.config = config or RetrievalConfig()
        
        # Resolve vector store path - go up to rag_system level
        base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(self.config.vector_store_path):
            self.vector_store_path = base_path / self.config.vector_store_path
        else:
            self.vector_store_path = Path(self.config.vector_store_path)
        
        # Lazy-loaded components
        self._vector_store: Optional[FAISSVectorStore] = None
        self._embedding_service: Optional[EmbeddingService] = None
    
    def _get_embedding_service(self) -> EmbeddingService:
        """Get or create embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service(
                model_type=self.config.embedding_model_type,
                model_name=self.config.embedding_model_name
            )
        return self._embedding_service
    
    def _get_vector_store(self) -> FAISSVectorStore:
        """Get or load vector store."""
        if self._vector_store is None:
            if not self.vector_store_path.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {self.vector_store_path}"
                )
            self._vector_store = FAISSVectorStore.load(str(self.vector_store_path))
        return self._vector_store
    
    def search(self,
               query: str,
               top_k: Optional[int] = None,
               threshold: Optional[float] = None,
               source_filter: Optional[str] = None,
               section_filter: Optional[str] = None,
               custom_filter: Optional[Callable[[Dict], bool]] = None) -> RetrievalResult:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results (default from config)
            threshold: Maximum distance threshold
            source_filter: Filter by source file name
            section_filter: Filter by section (partial match)
            custom_filter: Custom filter function
        
        Returns:
            RetrievalResult with matching chunks
        """
        import time
        start_time = time.time()
        
        # Use defaults from config
        top_k = top_k or self.config.default_top_k
        threshold = threshold if threshold is not None else self.config.default_threshold
        
        # Get components
        embedding_service = self._get_embedding_service()
        vector_store = self._get_vector_store()
        
        # Build combined filter
        def combined_filter(meta: Dict) -> bool:
            # Source filter
            if source_filter:
                if meta.get("source", "") != source_filter:
                    return False
            
            # Section filter (partial match)
            if section_filter:
                section = meta.get("section", "")
                if section_filter.lower() not in section.lower():
                    return False
            
            # Custom filter
            if custom_filter:
                if not custom_filter(meta):
                    return False
            
            return True
        
        # Determine if we need filtering
        filter_fn = combined_filter if (source_filter or section_filter or custom_filter) else None
        
        # Execute search
        results = vector_store.search_by_text(
            query_text=query,
            embedding_service=embedding_service,
            k=top_k * 3 if filter_fn else top_k  # Get more if filtering
        )
        
        # Apply filtering
        if filter_fn:
            filtered = []
            for r in results:
                meta = {
                    "source": r.source,
                    "section": r.section,
                    "chunk_id": r.chunk_id
                }
                if combined_filter(meta):
                    filtered.append(r)
            results = filtered[:top_k]
        
        # Apply threshold
        if threshold is not None:
            results = [r for r in results if r.score <= threshold]
        
        # Format results
        chunks = []
        for rank, result in enumerate(results, 1):
            chunk = {
                "chunk_id": result.chunk_id,
                "text": result.text,
                "score": round(result.score, 4),
                "source": result.source,
                "rank": rank
            }
            if result.section:
                chunk["section"] = result.section
            if result.page is not None:
                chunk["page"] = result.page
            chunks.append(chunk)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_found=len(chunks),
            search_time_ms=round(search_time_ms, 2)
        )
    
    def search_multi(self, 
                     queries: List[str],
                     top_k: Optional[int] = None,
                     deduplicate: bool = True) -> RetrievalResult:
        """
        Search with multiple queries and combine results.
        
        Args:
            queries: List of search queries
            top_k: Total number of unique results
            deduplicate: Remove duplicate chunks
        
        Returns:
            Combined RetrievalResult
        """
        top_k = top_k or self.config.default_top_k
        
        all_chunks = []
        seen_ids = set()
        
        for query in queries:
            result = self.search(query, top_k=top_k)
            
            for chunk in result.chunks:
                if deduplicate:
                    if chunk["chunk_id"] in seen_ids:
                        continue
                    seen_ids.add(chunk["chunk_id"])
                
                all_chunks.append(chunk)
        
        # Sort by score and limit
        all_chunks.sort(key=lambda x: x["score"])
        all_chunks = all_chunks[:top_k]
        
        # Re-rank
        for rank, chunk in enumerate(all_chunks, 1):
            chunk["rank"] = rank
        
        return RetrievalResult(
            query=" | ".join(queries),
            chunks=all_chunks,
            total_found=len(all_chunks)
        )
    
    def get_by_source(self, source: str, limit: int = 100) -> List[Dict]:
        """
        Get all chunks from a specific source.
        
        Args:
            source: Source file name
            limit: Maximum chunks to return
        
        Returns:
            List of chunks from the source
        """
        vector_store = self._get_vector_store()
        
        chunks = []
        for meta in vector_store.metadata:
            if meta.get("source") == source:
                chunks.append({
                    "chunk_id": meta.get("chunk_id"),
                    "text": meta.get("text"),
                    "section": meta.get("section", ""),
                    "page": meta.get("page")
                })
                if len(chunks) >= limit:
                    break
        
        return chunks
    
    def get_sources(self) -> List[str]:
        """Get list of all sources in the vector store."""
        vector_store = self._get_vector_store()
        
        sources = set()
        for meta in vector_store.metadata:
            source = meta.get("source")
            if source:
                sources.add(source)
        
        return sorted(list(sources))
    
    def get_stats(self) -> Dict:
        """Get statistics about the retrieval service."""
        try:
            vector_store = self._get_vector_store()
            sources = self.get_sources()
            
            return {
                "status": "ready",
                "total_chunks": vector_store.size,
                "dimension": vector_store.dimension,
                "model": vector_store.model_name,
                "sources": sources,
                "source_count": len(sources)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global service instance
_service_instance: Optional[RetrievalService] = None


def get_retrieval_service(config: Optional[RetrievalConfig] = None) -> RetrievalService:
    """Get or create the global retrieval service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RetrievalService(config)
    return _service_instance


if __name__ == "__main__":
    # Test the retrieval service
    print("Testing Retrieval Service...")
    print("=" * 60)
    
    service = RetrievalService()
    
    # Test basic search
    print("\n📝 Test 1: Basic search")
    result = service.search("data protection encryption", top_k=3)
    print(f"Found {result.total_found} chunks in {result.search_time_ms}ms")
    for chunk in result.chunks:
        print(f"  [{chunk['rank']}] {chunk['source']}: {chunk['text'][:50]}...")
    
    # Test context generation
    print("\n📝 Test 2: Context string generation")
    context = result.to_context_string(max_chunks=2)
    print(f"Context ({len(context)} chars):")
    print(context[:300] + "...")
    
    # Test source filtering
    print("\n📝 Test 3: Source filtering")
    result = service.search("security", source_filter="notes.txt", top_k=2)
    print(f"Found {result.total_found} chunks from notes.txt")
    
    # Test multi-query
    print("\n📝 Test 4: Multi-query search")
    result = service.search_multi([
        "data protection",
        "security measures",
        "encryption"
    ], top_k=3)
    print(f"Found {result.total_found} unique chunks")
    
    # Test stats
    print("\n📝 Test 5: Service stats")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

"""
Reranking Service - High-level service combining retrieval and reranking

This service provides an integrated pipeline:
1. Retrieve chunks using semantic search
2. Rerank using cross-encoder
3. Return top results with improved relevance
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.retrieval_service import RetrievalService, RetrievalConfig, RetrievalResult
from mcp_server.tools.rerank import CrossEncoderReranker


@dataclass
class RerankingConfig:
    """Configuration for the reranking service."""
    # Retrieval settings
    vector_store_path: str = "vector_store"
    embedding_model_type: str = "sentence-transformers"
    embedding_model_name: Optional[str] = None
    
    # Reranking settings
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Pipeline settings
    initial_top_k: int = 20  # Get more candidates for reranking
    final_top_k: int = 5     # Return fewer after reranking
    rerank_threshold: Optional[float] = None  # Optional minimum rerank score


class RerankingService:
    """
    Combined retrieval + reranking service.
    
    This service retrieves a larger set of candidates using bi-encoder
    similarity, then reranks them with a cross-encoder for better accuracy.
    
    Usage:
        service = RerankingService()
        results = service.search_and_rerank("What is data protection?")
    """
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        """Initialize the reranking service."""
        self.config = config or RerankingConfig()
        
        # Initialize retrieval service
        retrieval_config = RetrievalConfig(
            vector_store_path=self.config.vector_store_path,
            embedding_model_type=self.config.embedding_model_type,
            embedding_model_name=self.config.embedding_model_name,
            default_top_k=self.config.initial_top_k
        )
        self.retrieval_service = RetrievalService(retrieval_config)
        
        # Lazy-load reranker
        self._reranker: Optional[CrossEncoderReranker] = None
    
    def _get_reranker(self) -> CrossEncoderReranker:
        """Get or create reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.rerank_model)
        return self._reranker
    
    def search_and_rerank(self,
                          query: str,
                          initial_k: Optional[int] = None,
                          final_k: Optional[int] = None,
                          source_filter: Optional[str] = None,
                          section_filter: Optional[str] = None,
                          rerank_threshold: Optional[float] = None) -> RetrievalResult:
        """
        Search and rerank in one step.
        
        Args:
            query: Search query
            initial_k: Number of candidates to retrieve (default: 20)
            final_k: Number of results after reranking (default: 5)
            source_filter: Filter by source file
            section_filter: Filter by section
            rerank_threshold: Minimum rerank score to include
        
        Returns:
            RetrievalResult with reranked chunks
        """
        start_time = time.time()
        
        initial_k = initial_k or self.config.initial_top_k
        final_k = final_k or self.config.final_top_k
        rerank_threshold = rerank_threshold if rerank_threshold is not None else self.config.rerank_threshold
        
        # Step 1: Retrieve candidates
        retrieval_result = self.retrieval_service.search(
            query=query,
            top_k=initial_k,
            source_filter=source_filter,
            section_filter=section_filter
        )
        
        if not retrieval_result.chunks:
            return retrieval_result
        
        # Step 2: Rerank
        reranker = self._get_reranker()
        reranked = reranker.rerank(query, retrieval_result.chunks, top_k=final_k)
        
        # Step 3: Apply threshold if specified
        if rerank_threshold is not None:
            reranked = [c for c in reranked if c.get("rerank_score", 0) >= rerank_threshold]
        
        # Update ranks
        for i, chunk in enumerate(reranked, 1):
            chunk["rank"] = i
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            chunks=reranked,
            total_found=len(reranked),
            search_time_ms=round(search_time_ms, 2)
        )
    
    def compare_with_without_reranking(self,
                                        query: str,
                                        top_k: int = 5) -> Dict:
        """
        Compare results with and without reranking.
        
        Useful for evaluating the benefit of reranking.
        
        Args:
            query: Search query
            top_k: Number of results to compare
        
        Returns:
            Dictionary with 'without_reranking' and 'with_reranking' results
        """
        # Without reranking
        without = self.retrieval_service.search(query, top_k=top_k)
        
        # With reranking
        with_rerank = self.search_and_rerank(query, initial_k=top_k * 3, final_k=top_k)
        
        # Calculate position changes
        without_ids = [c["chunk_id"] for c in without.chunks]
        with_ids = [c["chunk_id"] for c in with_rerank.chunks]
        
        position_changes = []
        for i, chunk_id in enumerate(with_ids):
            if chunk_id in without_ids:
                old_pos = without_ids.index(chunk_id) + 1
                new_pos = i + 1
                change = old_pos - new_pos
                position_changes.append({
                    "chunk_id": chunk_id,
                    "old_position": old_pos,
                    "new_position": new_pos,
                    "change": change
                })
        
        return {
            "query": query,
            "without_reranking": {
                "chunks": without.chunks,
                "time_ms": without.search_time_ms
            },
            "with_reranking": {
                "chunks": with_rerank.chunks,
                "time_ms": with_rerank.search_time_ms
            },
            "position_changes": position_changes,
            "new_in_top_k": [c for c in with_ids if c not in without_ids],
            "dropped_from_top_k": [c for c in without_ids if c not in with_ids]
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the service."""
        retrieval_stats = self.retrieval_service.get_stats()
        
        return {
            **retrieval_stats,
            "rerank_model": self.config.rerank_model,
            "initial_top_k": self.config.initial_top_k,
            "final_top_k": self.config.final_top_k
        }


# Global service instance
_service_instance: Optional[RerankingService] = None


def get_reranking_service(config: Optional[RerankingConfig] = None) -> RerankingService:
    """Get or create the global reranking service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RerankingService(config)
    return _service_instance


if __name__ == "__main__":
    print("Testing Reranking Service...")
    print("=" * 60)
    
    service = RerankingService()
    
    # Test 1: Search and rerank
    print("\n📝 Test 1: Search and rerank")
    result = service.search_and_rerank(
        "data protection security encryption",
        initial_k=10,
        final_k=3
    )
    
    print(f"Found {result.total_found} chunks in {result.search_time_ms}ms")
    for chunk in result.chunks:
        print(f"\n  [{chunk['rank']}] {chunk['chunk_id']}")
        print(f"      Original: {chunk.get('original_score', 'N/A')}")
        print(f"      Rerank: {chunk.get('rerank_score', 'N/A'):.4f}")
        print(f"      Text: {chunk['text'][:60]}...")
    
    # Test 2: Compare with/without reranking
    print("\n" + "=" * 60)
    print("\n📝 Test 2: Compare with/without reranking")
    comparison = service.compare_with_without_reranking(
        "GDPR compliance personal data",
        top_k=3
    )
    
    print("\nWithout reranking:")
    for chunk in comparison["without_reranking"]["chunks"]:
        print(f"  [{chunk['rank']}] {chunk['chunk_id']} (score: {chunk['score']})")
    
    print("\nWith reranking:")
    for chunk in comparison["with_reranking"]["chunks"]:
        print(f"  [{chunk['rank']}] {chunk['chunk_id']} (rerank: {chunk.get('rerank_score', 0):.4f})")
    
    print(f"\nPosition changes: {comparison['position_changes']}")
    print(f"New in top-k: {comparison['new_in_top_k']}")
    print(f"Dropped: {comparison['dropped_from_top_k']}")
    
    # Test 3: Stats
    print("\n" + "=" * 60)
    print("\n📝 Test 3: Service stats")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

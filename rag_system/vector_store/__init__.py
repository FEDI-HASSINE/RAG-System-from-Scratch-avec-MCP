"""
Vector Store Package

Provides vector storage and retrieval using FAISS or Pinecone.
"""

from .faiss_store import FAISSVectorStore, VectorStoreManager, SearchResult

# Lazy import for Pinecone (optional dependency)
def get_pinecone_store():
    from .pinecone_store import PineconeVectorStore
    return PineconeVectorStore

__all__ = [
    "FAISSVectorStore",
    "VectorStoreManager",
    "SearchResult",
    "get_pinecone_store",
]

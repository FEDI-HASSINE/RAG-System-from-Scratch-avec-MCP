"""
Vector Store Package

Provides vector storage and retrieval using FAISS.
"""

from .faiss_store import (
    FAISSVectorStore,
    VectorStoreManager,
    SearchResult
)

__all__ = [
    "FAISSVectorStore",
    "VectorStoreManager",
    "SearchResult"
]

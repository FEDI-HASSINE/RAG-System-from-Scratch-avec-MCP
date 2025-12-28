"""
Embeddings Package

Provides text embedding models and services.
"""

from .embedding_models import (
    EmbeddingService,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    EmbeddingResult,
    get_embedding_service
)

__all__ = [
    "EmbeddingService",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "EmbeddingResult",
    "get_embedding_service"
]

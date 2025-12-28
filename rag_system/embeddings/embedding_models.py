"""
Embedding Models - Unified interface for text embeddings

Supports:
- SentenceTransformers (local, free)
- OpenAI Embeddings API (cloud, paid)
"""

import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    text: str
    vector: np.ndarray
    model: str
    dimension: int
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "vector": self.vector.tolist(),
            "model": self.model,
            "dimension": self.dimension
        }


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Embedding using SentenceTransformers (local, free).
    
    Popular models:
    - all-MiniLM-L6-v2: Fast, 384 dimensions
    - all-mpnet-base-v2: Better quality, 768 dimensions
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = None):
        """
        Initialize the SentenceTransformer model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"📦 Loading model: {self._model_name}...")
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                print(f"✅ Model loaded (dimension: {self._dimension})")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        self._load_model()
        return self._model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts with batching."""
        self._load_model()
        return self._model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAIEmbedding(BaseEmbeddingModel):
    """
    Embedding using OpenAI API (cloud, paid).
    
    Models:
    - text-embedding-3-small: Fast, 1536 dimensions
    - text-embedding-3-large: Better quality, 3072 dimensions
    - text-embedding-ada-002: Legacy, 1536 dimensions
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    def __init__(self, 
                 model_name: str = None,
                 api_key: str = None):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        
        if not self._api_key:
            print("⚠️  OPENAI_API_KEY not set. OpenAI embeddings will not work.")
    
    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai is required. Install with: pip install openai"
                )
        return self._client
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str], 
                    batch_size: int = 100,
                    show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts with batching."""
        client = self._get_client()
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self._model_name,
                input=batch
            )
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        
        return np.array(all_embeddings)
    
    @property
    def dimension(self) -> int:
        return self.DIMENSIONS.get(self._model_name, 1536)
    
    @property
    def model_name(self) -> str:
        return self._model_name


class EmbeddingService:
    """
    High-level embedding service with caching and unified interface.
    """
    
    def __init__(self, 
                 model_type: str = "sentence-transformers",
                 model_name: str = None,
                 cache_embeddings: bool = True):
        """
        Initialize the embedding service.
        
        Args:
            model_type: 'sentence-transformers' or 'openai'
            model_name: Specific model name (optional)
            cache_embeddings: Whether to cache embeddings in memory
        """
        self.model_type = model_type
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}
        
        # Initialize the appropriate model
        if model_type == "sentence-transformers":
            self.model = SentenceTransformerEmbedding(model_name)
        elif model_type == "openai":
            self.model = OpenAIEmbedding(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached result if available
        
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if use_cache and self.cache_embeddings and text in self._cache:
            return self._cache[text]
        
        # Generate embedding
        vector = self.model.embed(text)
        
        # Store in cache
        if self.cache_embeddings:
            self._cache[text] = vector
        
        return vector
    
    def embed_batch(self, texts: List[str], 
                    show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
        
        Returns:
            2D numpy array of embeddings
        """
        return self.model.embed_batch(texts, show_progress=show_progress)
    
    def embed_chunks(self, chunks: List[Dict], 
                     text_key: str = "text") -> List[Dict]:
        """
        Embed chunks and return with vectors included.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key containing the text to embed
        
        Returns:
            Chunks with 'vector' field added
        """
        texts = [chunk[text_key] for chunk in chunks]
        vectors = self.embed_batch(texts)
        
        result = []
        for chunk, vector in zip(chunks, vectors):
            chunk_with_vector = chunk.copy()
            chunk_with_vector["vector"] = vector.tolist()
            result.append(chunk_with_vector)
        
        return result
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.dimension
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.model.model_name
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


def get_embedding_service(model_type: str = "sentence-transformers",
                          model_name: Optional[str] = None) -> EmbeddingService:
    """
    Factory function to create an embedding service.
    
    Args:
        model_type: 'sentence-transformers' or 'openai'
        model_name: Specific model name (optional)
    
    Returns:
        Configured EmbeddingService
    """
    return EmbeddingService(model_type=model_type, model_name=model_name)


if __name__ == "__main__":
    # Test the embedding service
    print("Testing SentenceTransformers embedding...")
    
    service = get_embedding_service("sentence-transformers")
    
    # Test single embedding
    text = "This is a test sentence for embedding."
    vector = service.embed(text)
    print(f"Text: {text}")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector (first 5): {vector[:5]}")
    
    # Test batch embedding
    texts = [
        "First sentence about data protection.",
        "Second sentence about machine learning.",
        "Third sentence about privacy policies."
    ]
    vectors = service.embed_batch(texts, show_progress=False)
    print(f"\nBatch embedding shape: {vectors.shape}")
    
    print(f"\n✅ Embedding service working (dimension: {service.dimension})")

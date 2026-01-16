"""
Vector Store - FAISS-based vector storage and retrieval

Features:
- Efficient similarity search
- Metadata storage
- Persistence to disk
- Multiple index types support
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    text: str
    score: float  # Distance (lower is better for L2, higher for cosine)
    source: str = ""
    section: str = ""
    page: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": float(self.score),
            "source": self.source
        }
        if self.section:
            result["section"] = self.section
        if self.page is not None:
            result["page"] = self.page
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    
    Supports:
    - L2 (Euclidean) distance
    - Inner product (cosine similarity with normalized vectors)
    - IVF index for large datasets
    """
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = "flat",
                 metric: str = "l2"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Vector dimension
            index_type: 'flat' (exact) or 'ivf' (approximate, faster for large data)
            metric: 'l2' (Euclidean) or 'cosine' (inner product with normalization)
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Create the index
        self.index = self._create_index()
        
        # Metadata storage: list aligned with index positions
        self.metadata: List[Dict] = []
        
        # Mapping chunk_id -> index position
        self.id_to_position: Dict[str, int] = {}
        
        # Store info
        self.created_at = datetime.now().isoformat()
        self.model_name: Optional[str] = None
    
    def _create_index(self):
        """Create FAISS index based on configuration."""
        if self.metric == "cosine":
            # For cosine similarity, use inner product with normalized vectors
            if self.index_type == "flat":
                return self.faiss.IndexFlatIP(self.dimension)
            else:
                # IVF with inner product
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                return self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            # L2 distance
            if self.index_type == "flat":
                return self.faiss.IndexFlatL2(self.dimension)
            else:
                # IVF for large datasets
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                return self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add(self, 
            vectors: np.ndarray, 
            metadata_list: List[Dict]) -> List[str]:
        """
        Add vectors with metadata to the store.
        
        Args:
            vectors: 2D numpy array of vectors (n_vectors, dimension)
            metadata_list: List of metadata dicts (must include 'chunk_id' and 'text')
        
        Returns:
            List of chunk_ids added
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata must have same length")
        
        # Ensure vectors are float32
        vectors = np.array(vectors, dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            vectors = self._normalize(vectors)
        
        # Ensure C-contiguous for FAISS
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        
        # Get current position
        start_position = len(self.metadata)
        
        # Add to FAISS index
        if self.index_type == "ivf" and self.index.ntotal == 0:
            # IVF index needs training before adding vectors
            train_vectors = np.ascontiguousarray(vectors[:min(len(vectors), 256)], dtype=np.float32)
            self.index.train(train_vectors)  # type: ignore
        self.index.add(vectors.astype(np.float32))  # type: ignore
        
        # Store metadata and build mappings
        chunk_ids = []
        for i, meta in enumerate(metadata_list):
            chunk_id = meta.get("chunk_id", f"chunk_{start_position + i}")
            self.metadata.append(meta)
            self.id_to_position[chunk_id] = start_position + i
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def add_single(self, vector: np.ndarray, metadata: Dict) -> str:
        """Add a single vector with metadata."""
        vectors = np.array([vector], dtype=np.float32)
        return self.add(vectors, [metadata])[0]
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5,
               filter_fn: Optional[Callable] = None) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_fn: Optional function to filter results (receives metadata, returns bool)
        
        Returns:
            List of SearchResult objects
        """
        if self.index.ntotal == 0:
            return []
        
        # Prepare query
        query = np.array([query_vector], dtype=np.float32)
        if self.metric == "cosine":
            query = self._normalize(query)
        
        # Search (get more results if filtering)
        search_k = k * 3 if filter_fn else k
        search_k = min(search_k, self.index.ntotal)
        
        distances, indices = self.index.search(query, search_k)  # type: ignore
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            meta = self.metadata[idx]
            
            # Apply filter
            if filter_fn and not filter_fn(meta):
                continue
            
            result = SearchResult(
                chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                text=meta.get("text", ""),
                score=float(dist),
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                page=meta.get("page"),
                metadata={k: v for k, v in meta.items() 
                         if k not in ["chunk_id", "text", "source", "section", "page", "vector"]}
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_text(self, 
                       query_text: str,
                       embedding_service,
                       k: int = 5) -> List[SearchResult]:
        """
        Search using text query (embeds the query first).
        
        Args:
            query_text: Text to search for
            embedding_service: EmbeddingService instance
            k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        query_vector = embedding_service.embed(query_text)
        return self.search(query_vector, k)
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get metadata by chunk_id."""
        position = self.id_to_position.get(chunk_id)
        if position is not None:
            return self.metadata[position]
        return None
    
    def save(self, directory: str):
        """
        Save the vector store to disk.
        
        Creates:
        - index.faiss: The FAISS index
        - metadata.json: Chunk metadata and mappings
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        self.faiss.write_index(self.index, str(index_path))
        print(f"💾 Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = path / "metadata.json"
        meta_export = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "total_vectors": self.index.ntotal,
            "chunks": self.metadata,
            "id_to_position": self.id_to_position
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta_export, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved metadata to {metadata_path}")
        
        print(f"✅ Vector store saved: {self.index.ntotal} vectors")
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing index.faiss and metadata.json
        
        Returns:
            Loaded FAISSVectorStore instance
        """
        import faiss
        
        path = Path(directory)
        
        # Load metadata first to get configuration
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_export = json.load(f)
        
        # Create instance with saved configuration
        store = cls(
            dimension=meta_export["dimension"],
            index_type=meta_export["index_type"],
            metric=meta_export["metric"]
        )
        
        # Load FAISS index
        index_path = path / "index.faiss"
        store.index = faiss.read_index(str(index_path))
        
        # Restore metadata
        store.metadata = meta_export["chunks"]
        store.id_to_position = meta_export["id_to_position"]
        store.created_at = meta_export["created_at"]
        store.model_name = meta_export.get("model_name")
        
        print(f"✅ Loaded vector store: {store.index.ntotal} vectors")
        return store
    
    @property
    def size(self) -> int:
        """Return number of vectors in the store."""
        return self.index.ntotal
    
    def __len__(self) -> int:
        return self.size


class VectorStoreManager:
    """
    High-level manager for vector store operations.
    
    Combines embedding service with vector store for easy usage.
    """
    
    def __init__(self, 
                 store_path: str = "vector_store",
                 embedding_service = None):
        """
        Initialize the manager.
        
        Args:
            store_path: Directory for vector store files
            embedding_service: EmbeddingService instance (created if not provided)
        """
        self.store_path = Path(store_path)
        
        # Initialize or load embedding service
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            from rag_system.embeddings.embedding_models import get_embedding_service
            self.embedding_service = get_embedding_service()
        
        # Try to load existing store or create new
        self.vector_store: Optional[FAISSVectorStore] = None
        self._try_load_store()
    
    def _try_load_store(self):
        """Try to load existing vector store."""
        index_path = self.store_path / "index.faiss"
        if index_path.exists():
            try:
                self.vector_store = FAISSVectorStore.load(str(self.store_path))
            except Exception as e:
                print(f"⚠️  Could not load existing store: {e}")
    
    def index_chunks(self, chunks: List[Dict], 
                     text_key: str = "text",
                     save: bool = True) -> int:
        """
        Index chunks into the vector store.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key containing text to embed
            save: Whether to save after indexing
        
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        print(f"🔄 Indexing {len(chunks)} chunks...")
        
        # Generate embeddings
        texts = [chunk[text_key] for chunk in chunks]
        vectors = self.embedding_service.embed_batch(texts)
        
        # Create vector store if needed
        if self.vector_store is None:
            self.vector_store = FAISSVectorStore(
                dimension=self.embedding_service.dimension,
                index_type="flat",
                metric="l2"
            )
            self.vector_store.model_name = self.embedding_service.model_name
        
        # Add to store
        self.vector_store.add(vectors, chunks)
        
        # Save if requested
        if save:
            self.save()
        
        print(f"✅ Indexed {len(chunks)} chunks (total: {self.vector_store.size})")
        return len(chunks)
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query text
            k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        if self.vector_store is None or self.vector_store.size == 0:
            return []
        
        return self.vector_store.search_by_text(
            query, 
            self.embedding_service, 
            k=k
        )
    
    def save(self):
        """Save the vector store to disk."""
        if self.vector_store:
            self.store_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save(str(self.store_path))
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.vector_store is None:
            return {"status": "empty", "total_vectors": 0}
        
        return {
            "status": "loaded",
            "total_vectors": self.vector_store.size,
            "dimension": self.vector_store.dimension,
            "model": self.vector_store.model_name,
            "index_type": self.vector_store.index_type,
            "metric": self.vector_store.metric
        }


if __name__ == "__main__":
    # Test the vector store
    import numpy as np
    
    print("Testing FAISSVectorStore...")
    
    # Create store
    store = FAISSVectorStore(dimension=384, metric="l2")
    
    # Add some test vectors
    np.random.seed(42)
    vectors = np.random.randn(5, 384).astype(np.float32)
    metadata = [
        {"chunk_id": "test_001", "text": "First test chunk", "source": "test.txt"},
        {"chunk_id": "test_002", "text": "Second test chunk", "source": "test.txt"},
        {"chunk_id": "test_003", "text": "Third test chunk", "source": "test.txt"},
        {"chunk_id": "test_004", "text": "Fourth test chunk", "source": "other.txt"},
        {"chunk_id": "test_005", "text": "Fifth test chunk", "source": "other.txt"},
    ]
    
    store.add(vectors, metadata)
    print(f"Added {store.size} vectors")
    
    # Search
    query = vectors[0]  # Use first vector as query
    results = store.search(query, k=3)
    
    print("\nSearch results:")
    for r in results:
        print(f"  {r.chunk_id}: {r.text} (score: {r.score:.4f})")
    
    print("\n✅ Vector store working correctly!")

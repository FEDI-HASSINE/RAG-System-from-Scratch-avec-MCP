"""
Vector Store - Pinecone-based storage and retrieval

This module provides a drop-in alternative to FAISSVectorStore using
Pinecone's managed vector database. It keeps a similar surface area:
- add / add_single
- search / search_by_text
- basic stats (size, dimension, model_name)

Configuration can be provided via parameters or environment variables:
- PINECONE_API_KEY (required)
- PINECONE_INDEX (defaults to 'rag-index')
- PINECONE_NAMESPACE (defaults to 'default')
- PINECONE_CLOUD (defaults to 'aws')
- PINECONE_REGION (defaults to 'us-east-1')
- VECTOR_STORE_METRIC (defaults to 'cosine')
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from uuid import uuid4

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError("Pinecone client is required. Install with: pip install pinecone-client") from e


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    text: str
    score: float  # Higher is better (similarity)
    source: str = ""
    section: str = ""
    page: Optional[int] = None

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
        return result


class PineconeVectorStore:
    """Pinecone-backed vector store with a FAISS-like interface."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "default",
        dimension: int = 384,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        self.client = Pinecone(api_key=self.api_key)
        self._ensure_index()
        self.index = self.client.Index(self.index_name)

        # Stats
        self.model_name: Optional[str] = None
        self.size: int = 0
        self.index_type: str = "pinecone"

    @classmethod
    def from_env(
        cls,
        dimension: int,
        metric: str = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Create a store from environment variables."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone backend")

        return cls(
            api_key=api_key,
            index_name=index_name or os.getenv("PINECONE_INDEX", "rag-index"),
            namespace=namespace or os.getenv("PINECONE_NAMESPACE", "demo"),  # Changed default to 'demo'
            dimension=dimension,
            metric=metric or os.getenv("VECTOR_STORE_METRIC", "cosine"),
            cloud=cloud or os.getenv("PINECONE_CLOUD", "aws"),
            region=region or os.getenv("PINECONE_REGION", "us-east-1"),
        )

    def _ensure_index(self):
        """Create the index if it does not exist."""
        existing = {idx.name for idx in self.client.list_indexes()}
        if self.index_name not in existing:
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        # Wait until ready
        while True:
            desc = self.client.describe_index(self.index_name)
            if desc.status.get("ready"):
                break
            time.sleep(1)

    def add(self, vectors: np.ndarray, metadata_list: List[Dict], 
             batch_size: int = 100, max_retries: int = 3) -> List[str]:
        """Upsert vectors with metadata using batching and retry logic.
        
        Args:
            vectors: numpy array of vectors
            metadata_list: list of metadata dicts
            batch_size: number of vectors per batch (default: 100)
            max_retries: max retry attempts per batch (default: 3)
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata must have same length")

        # Prepare all upserts
        all_upserts = []
        ids = []
        for vec, meta in zip(vectors, metadata_list):
            chunk_id = meta.get("chunk_id") or str(uuid4())
            ids.append(chunk_id)
            all_upserts.append({
                "id": chunk_id,
                "values": np.asarray(vec, dtype=float).tolist(),
                "metadata": meta
            })
        
        # Batch upsert with retry logic
        total_batches = (len(all_upserts) + batch_size - 1) // batch_size
        for batch_idx in range(0, len(all_upserts), batch_size):
            batch = all_upserts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            for attempt in range(max_retries):
                try:
                    self.index.upsert(vectors=batch, namespace=self.namespace)
                    break  # Success
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️  Batch {batch_num}/{total_batches} failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Failed to upsert batch {batch_num} after {max_retries} attempts: {e}")
        
        self.size += len(all_upserts)
        return ids

    def add_single(self, vector: np.ndarray, metadata: Dict) -> str:
        """Add a single vector."""
        return self.add(np.array([vector]), [metadata])[0]

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search similar vectors."""
        query = np.asarray(query_vector, dtype=float).tolist()

        response = self.index.query(
            vector=query,
            top_k=k * 3 if filter_fn else k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter_dict,
        )

        results: List[SearchResult] = []
        for match in response.matches:
            meta = match.metadata or {}
            if filter_fn and not filter_fn(meta):
                continue
            results.append(
                SearchResult(
                    chunk_id=match.id,
                    text=meta.get("text", ""),
                    score=float(match.score),
                    source=meta.get("source", ""),
                    section=meta.get("section", ""),
                    page=meta.get("page"),
                )
            )
            if len(results) >= k:
                break
        return results

    def search_by_text(self, query_text: str, embedding_service, k: int = 5) -> List[SearchResult]:
        """Search using a text query by embedding it first."""
        query_vector = embedding_service.embed(query_text)
        return self.search(query_vector, k=k)

    def save(self, directory: str):  # pragma: no cover - not needed for Pinecone
        """No-op for Pinecone (managed persistence)."""
        return

    @classmethod
    def load(cls, *args, **kwargs):  # pragma: no cover - compatibility placeholder
        raise NotImplementedError("Use from_env() to instantiate PineconeVectorStore")


__all__ = ["PineconeVectorStore", "SearchResult"]

"""
Indexing Pipeline - Create embeddings and build vector store from chunks

This pipeline:
1. Loads chunks.json from Phase 1
2. Generates embeddings for all chunks
3. Builds FAISS index
4. Saves to vector_store/ directory
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_models import get_embedding_service, EmbeddingService
from vector_store.faiss_store import FAISSVectorStore, VectorStoreManager


@dataclass
class IndexingConfig:
    """Configuration for the indexing pipeline."""
    # Input
    chunks_file: str = "data/chunks.json"
    
    # Output
    vector_store_dir: str = "vector_store"
    vector_backend: str = "faiss"  # 'faiss' or 'pinecone'
    
    # Model settings
    model_type: str = "sentence-transformers"  # or 'openai'
    model_name: Optional[str] = None  # None = use default
    
    # Index settings
    index_type: str = "flat"  # 'flat' or 'ivf'
    metric: str = "l2"  # 'l2' or 'cosine'

    # Pinecone settings (optional; fallback to env)
    pinecone_index: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    
    # Processing
    batch_size: int = 32
    show_progress: bool = True


@dataclass
class IndexingResult:
    """Results from the indexing pipeline."""
    success: bool
    chunks_indexed: int
    dimension: int
    model: str
    index_path: str
    metadata_path: str
    errors: List[str]
    processing_time: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class IndexingPipeline:
    """
    Pipeline for creating embeddings and building vector index.
    
    Usage:
        pipeline = IndexingPipeline()
        result = pipeline.run()
    """
    
    def __init__(self, config: Optional[IndexingConfig] = None):
        """Initialize the pipeline with configuration."""
        self.config = config or IndexingConfig()
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__)))

    def _backend(self) -> str:
        """Return normalized backend name."""
        env_backend = os.getenv("VECTOR_STORE_BACKEND")
        if env_backend:
            return env_backend.lower()
        return (self.config.vector_backend or "faiss").lower()
    
    def run(self, 
            chunks_file: Optional[str] = None,
            output_dir: Optional[str] = None) -> IndexingResult:
        """
        Run the complete indexing pipeline.
        
        Args:
            chunks_file: Path to chunks.json (overrides config)
            output_dir: Path to vector_store directory (overrides config)
        
        Returns:
            IndexingResult with statistics
        """
        chunks_file_path = chunks_file or self.config.chunks_file
        output_dir_path = output_dir or self.config.vector_store_dir
        
        # Resolve paths relative to rag_system/
        chunks_file_resolved: Path
        output_dir_resolved: Path
        if not os.path.isabs(chunks_file_path):
            chunks_file_resolved = self.base_path / chunks_file_path
        else:
            chunks_file_resolved = Path(chunks_file_path)
        if not os.path.isabs(output_dir_path):
            output_dir_resolved = self.base_path / output_dir_path
        else:
            output_dir_resolved = Path(output_dir_path)
        
        print("=" * 60)
        print("🚀 Starting Indexing Pipeline (Phase 2)")
        print("=" * 60)
        print(f"📂 Input: {chunks_file_resolved}")
        print(f"📂 Output: {output_dir_resolved}")
        print(f"🤖 Model: {self.config.model_type} / {self.config.model_name or 'default'}")
        print("=" * 60)
        
        errors = []
        start_time = datetime.now()
        
        # Step 1: Load chunks
        print("\n📖 Step 1: Loading chunks...")
        try:
            chunks = self._load_chunks(str(chunks_file_resolved))
            print(f"   ✅ Loaded {len(chunks)} chunks")
        except Exception as e:
            error = f"Failed to load chunks: {e}"
            print(f"   ❌ {error}")
            return IndexingResult(
                success=False,
                chunks_indexed=0,
                dimension=0,
                model="",
                index_path="",
                metadata_path="",
                errors=[error],
                processing_time="0:00:00"
            )
        
        # Step 2: Initialize embedding service
        print("\n🤖 Step 2: Loading embedding model...")
        try:
            embedding_service = get_embedding_service(
                model_type=self.config.model_type,
                model_name=self.config.model_name
            )
            print(f"   ✅ Model loaded: {embedding_service.model_name}")
            print(f"   📐 Dimension: {embedding_service.dimension}")
        except Exception as e:
            error = f"Failed to load embedding model: {e}"
            print(f"   ❌ {error}")
            return IndexingResult(
                success=False,
                chunks_indexed=0,
                dimension=0,
                model="",
                index_path="",
                metadata_path="",
                errors=[error],
                processing_time=str(datetime.now() - start_time)
            )
        
        # Step 3: Generate embeddings
        print("\n🔄 Step 3: Generating embeddings...")
        try:
            texts = [chunk["text"] for chunk in chunks]
            vectors = embedding_service.embed_batch(
                texts, 
                show_progress=self.config.show_progress
            )
            print(f"   ✅ Generated {len(vectors)} embeddings")
        except Exception as e:
            error = f"Failed to generate embeddings: {e}"
            print(f"   ❌ {error}")
            errors.append(error)
            return IndexingResult(
                success=False,
                chunks_indexed=0,
                dimension=embedding_service.dimension,
                model=embedding_service.model_name,
                index_path="",
                metadata_path="",
                errors=errors,
                processing_time=str(datetime.now() - start_time)
            )
        
        # Step 4: Create vector store (FAISS or Pinecone)
        backend = self._backend()
        metric = self.config.metric
        if backend == "pinecone" and metric == "l2":
            metric = "euclidean"  # Pinecone naming
        print(f"\n📊 Step 4: Building vector index ({backend})...")
        try:
            if backend == "pinecone":
                from vector_store.pinecone_store import PineconeVectorStore
                vector_store = PineconeVectorStore.from_env(
                    dimension=embedding_service.dimension,
                    metric=metric,
                    index_name=self.config.pinecone_index,
                    namespace=self.config.pinecone_namespace,
                    cloud=self.config.pinecone_cloud,
                    region=self.config.pinecone_region,
                )
                vector_store.model_name = embedding_service.model_name
                vector_store.add(vectors, chunks)
                index_path = f"pinecone://{vector_store.index_name}/{vector_store.namespace}"
                metadata_path = index_path
                print(f"   ✅ Upserted {len(vectors)} vectors to Pinecone index '{vector_store.index_name}' (ns: {vector_store.namespace})")
            else:
                vector_store = FAISSVectorStore(
                    dimension=embedding_service.dimension,
                    index_type=self.config.index_type,
                    metric=self.config.metric
                )
                vector_store.model_name = embedding_service.model_name
                vector_store.add(vectors, chunks)
                print(f"   ✅ Added {vector_store.size} vectors to index")
                index_path = output_dir_resolved / "index.faiss"
                metadata_path = output_dir_resolved / "metadata.json"
                # Save to disk (FAISS only)
                output_dir_resolved.mkdir(parents=True, exist_ok=True)
                vector_store.save(str(output_dir_resolved))
                print(f"   ✅ Saved index to {index_path}")
                print(f"   ✅ Saved metadata to {metadata_path}")
        except Exception as e:
            error = f"Failed to build index: {e}"
            print(f"   ❌ {error}")
            errors.append(error)
            return IndexingResult(
                success=False,
                chunks_indexed=0,
                dimension=embedding_service.dimension,
                model=embedding_service.model_name,
                index_path="",
                metadata_path="",
                errors=errors,
                processing_time=str(datetime.now() - start_time)
            )
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = str(end_time - start_time)
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Indexing Pipeline Complete!")
        print("=" * 60)
        print(f"📊 Chunks indexed: {vector_store.size}")
        print(f"📐 Vector dimension: {embedding_service.dimension}")
        print(f"🤖 Model: {embedding_service.model_name}")
        print(f"⏱️  Processing time: {processing_time}")
        
        if errors:
            print(f"⚠️  Errors: {len(errors)}")
        
        return IndexingResult(
            success=len(errors) == 0,
            chunks_indexed=vector_store.size,
            dimension=embedding_service.dimension,
            model=embedding_service.model_name,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            errors=errors,
            processing_time=processing_time
        )
    
    def _load_chunks(self, chunks_file: str) -> List[Dict]:
        """Load chunks from JSON file."""
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_search(self, query: str, k: int = 3):
        """
        Test search functionality after indexing.
        
        Args:
            query: Search query text
            k: Number of results
        """
        print(f"\n🔍 Testing search: '{query}'")
        
        # Load the store
        store_path = self.base_path / self.config.vector_store_dir
        vector_store = FAISSVectorStore.load(str(store_path))
        
        # Get embedding service
        embedding_service = get_embedding_service(
            model_type=self.config.model_type,
            model_name=self.config.model_name
        )
        
        # Search
        results = vector_store.search_by_text(query, embedding_service, k=k)
        
        print(f"\nTop {k} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.chunk_id}] (score: {result.score:.4f})")
            print(f"   Source: {result.source}")
            if result.section:
                print(f"   Section: {result.section}")
            print(f"   Text: {result.text[:150]}...")
        
        return results


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings and build vector index"
    )
    parser.add_argument(
        "-i", "--input",
        default="data/chunks.json",
        help="Input chunks.json file (default: data/chunks.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default="vector_store",
        help="Output directory (default: vector_store)"
    )
    parser.add_argument(
        "--model-type",
        choices=["sentence-transformers", "openai"],
        default="sentence-transformers",
        help="Embedding model type"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Specific model name (e.g., all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--test-query",
        default=None,
        help="Run a test search after indexing"
    )
    
    args = parser.parse_args()
    
    config = IndexingConfig(
        chunks_file=args.input,
        vector_store_dir=args.output,
        model_type=args.model_type,
        model_name=args.model_name
    )
    
    pipeline = IndexingPipeline(config)
    result = pipeline.run()
    
    # Optional test search
    if args.test_query and result.success:
        pipeline.test_search(args.test_query)
    
    # Exit with appropriate code
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

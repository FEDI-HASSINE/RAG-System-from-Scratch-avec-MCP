#!/usr/bin/env python3
"""
Run the embedding/indexing pipeline (Phase 2).

Usage:
    python run_indexing.py
    python run_indexing.py --test-query "data protection"
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings.indexing_pipeline import IndexingPipeline, IndexingConfig


def main():
    """Run the indexing pipeline with default settings."""
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Configure the pipeline
    config = IndexingConfig(
        chunks_file=os.path.join(base_path, "data", "chunks.json"),
        vector_store_dir=os.path.join(base_path, "vector_store"),
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        index_type="flat",
        metric="l2",
        batch_size=32,
        show_progress=True
    )
    
    # Create and run pipeline
    pipeline = IndexingPipeline(config)
    result = pipeline.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 INDEXING SUMMARY")
    print("=" * 60)
    print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Dimension: {result.dimension}")
    print(f"Model: {result.model}")
    print(f"Processing time: {result.processing_time}")
    
    if result.errors:
        print(f"\n⚠️  Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    
    # Run a test search
    if result.success:
        print("\n" + "=" * 60)
        print("🔍 TEST SEARCH")
        print("=" * 60)
        pipeline.test_search("data protection privacy GDPR", k=3)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

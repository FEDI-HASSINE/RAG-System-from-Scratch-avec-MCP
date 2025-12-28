#!/usr/bin/env python3
"""
Run the data ingestion pipeline.

Usage:
    python run_ingestion.py
    python run_ingestion.py --input ./data/raw_docs --output ./data/chunks.json
    python run_ingestion.py --max-tokens 400 --overlap 100
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ingestion_pipeline import DataIngestionPipeline, IngestionConfig


def main():
    """Run the ingestion pipeline with default settings."""
    
    # Configure the pipeline
    config = IngestionConfig(
        input_dir=os.path.join(os.path.dirname(__file__), "data", "raw_docs"),
        output_file=os.path.join(os.path.dirname(__file__), "data", "chunks.json"),
        max_tokens=500,
        overlap_tokens=80,
        min_line_length=20,
        remove_urls=True,
        detect_headers=True,
        detect_lists=True,
        validate_output=True
    )
    
    # Create and run pipeline
    pipeline = DataIngestionPipeline(config)
    result = pipeline.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 INGESTION SUMMARY")
    print("=" * 60)
    print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Documents: {result.documents_processed}")
    print(f"Chunks: {result.total_chunks}")
    print(f"Output: {result.output_file}")
    
    if result.errors:
        print(f"\n⚠️  Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

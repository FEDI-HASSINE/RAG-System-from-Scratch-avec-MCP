"""
Data Ingestion Pipeline - Main orchestrator for document processing

This module combines all data processing steps:
1. Load documents (PDF, MD, TXT)
2. Clean text
3. Detect structure
4. Chunk intelligently
5. Output to chunks.json
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Local imports
from .loaders import UnifiedDocumentLoader, LoadedDocument
from .cleaner import TextCleaner, CleaningStats
from .structure_detector import StructureDetector, StructuredElement
from .chunker import IntelligentChunker, Chunk, validate_chunks


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    # Input/Output
    input_dir: str = "raw_docs"
    output_file: str = "chunks.json"
    
    # Cleaning options
    min_line_length: int = 20
    remove_urls: bool = True
    normalize_unicode: bool = True
    
    # Chunking options
    max_tokens: int = 500
    overlap_tokens: int = 80
    token_method: str = "simple"  # 'simple' or 'tiktoken'
    
    # Structure detection
    detect_headers: bool = True
    detect_lists: bool = True
    
    # Validation
    validate_output: bool = True
    
    # Optional: activer Chonkie pour extraire des éléments riches (tables/images)
    use_chonkie: bool = False


@dataclass
class IngestionResult:
    """Results from the ingestion pipeline."""
    success: bool
    documents_processed: int
    total_chunks: int
    output_file: str
    errors: List[str]
    stats: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DataIngestionPipeline:
    """
    Main pipeline for ingesting documents and producing chunks.
    
    Usage:
        pipeline = DataIngestionPipeline()
        result = pipeline.run("path/to/raw_docs", "output/chunks.json")
    """
    
    def __init__(self, config: IngestionConfig = None):
        """Initialize the pipeline with configuration."""
        self.config = config or IngestionConfig()
        
        # Initialize components
        self.loader = UnifiedDocumentLoader()
        self.cleaner = TextCleaner(
            min_line_length=self.config.min_line_length,
            remove_urls=self.config.remove_urls,
            normalize_unicode=self.config.normalize_unicode
        )
        self.structure_detector = StructureDetector(
            detect_headers=self.config.detect_headers,
            detect_lists=self.config.detect_lists
        )
        self.chunker = IntelligentChunker(
            max_tokens=self.config.max_tokens,
            overlap_tokens=self.config.overlap_tokens,
            token_method=self.config.token_method
        )
        # Chonkie adapter (optionnel)
        self._chonkie_adapter = None
        if self.config.use_chonkie:
            try:
                from .chonkie_adapter import ChonkieAdapter
                self._chonkie_adapter = ChonkieAdapter(
                    max_tokens=self.config.max_tokens,
                    overlap_tokens=self.config.overlap_tokens,
                    token_method=self.config.token_method,
                )
                print("   🧩 Chonkie adapter activé")
            except Exception as e:
                print(f"   ⚠️ Impossible d'activer Chonkie: {e}")
                print("   ▶️ Fallback sur le détecteur de structure interne.")
                self._chonkie_adapter = None
    
    def run(self, 
            input_dir: str = None, 
            output_file: str = None) -> IngestionResult:
        """
        Run the complete ingestion pipeline.
        
        Args:
            input_dir: Directory containing raw documents
            output_file: Path to output chunks.json file
        
        Returns:
            IngestionResult with processing statistics
        """
        input_dir = input_dir or self.config.input_dir
        output_file = output_file or self.config.output_file
        
        print("=" * 60)
        print("🚀 Starting Data Ingestion Pipeline")
        print("=" * 60)
        print(f"📂 Input: {input_dir}")
        print(f"📄 Output: {output_file}")
        print(f"⚙️  Max tokens: {self.config.max_tokens}, Overlap: {self.config.overlap_tokens}")
        print("=" * 60)
        
        errors = []
        all_chunks = []
        stats = {
            "documents": {},
            "cleaning": {},
            "total_original_chars": 0,
            "total_cleaned_chars": 0,
            "processing_time": None
        }
        
        start_time = datetime.now()
        
        # Step 1: Load documents
        print("\n📖 Step 1: Loading documents...")
        documents = self.loader.load_directory(input_dir)
        
        if not documents:
            return IngestionResult(
                success=False,
                documents_processed=0,
                total_chunks=0,
                output_file=output_file,
                errors=["No documents found in input directory"],
                stats=stats
            )
        
        # Step 2-5: Process each document
        for doc in documents:
            print(f"\n📄 Processing: {doc.filename}")
            
            try:
                chunks = self._process_document(doc, stats)
                all_chunks.extend(chunks)
                print(f"   ✅ Generated {len(chunks)} chunks")
            except Exception as e:
                error_msg = f"Error processing {doc.filename}: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
        
        # Step 6: Validate chunks
        if self.config.validate_output:
            print("\n🔍 Step 6: Validating chunks...")
            validation = validate_chunks(all_chunks, self.config.max_tokens)
            
            if not validation["valid"]:
                for issue in validation["issues"]:
                    print(f"   ⚠️  {issue}")
                    errors.append(issue)
            else:
                print(f"   ✅ All {len(all_chunks)} chunks valid")
            
            stats["validation"] = validation["stats"]
        
        # Step 7: Write output
        print(f"\n💾 Step 7: Writing {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_data = [chunk.to_dict() for chunk in all_chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Wrote {len(all_chunks)} chunks to {output_file}")
        
        # Calculate processing time
        end_time = datetime.now()
        stats["processing_time"] = str(end_time - start_time)
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print("=" * 60)
        print(f"📊 Documents processed: {len(documents)}")
        print(f"📊 Total chunks: {len(all_chunks)}")
        print(f"📊 Processing time: {stats['processing_time']}")
        
        if stats.get("validation"):
            print(f"📊 Avg tokens/chunk: {stats['validation']['avg_tokens']:.1f}")
        
        if errors:
            print(f"⚠️  Errors: {len(errors)}")
        
        return IngestionResult(
            success=len(errors) == 0,
            documents_processed=len(documents),
            total_chunks=len(all_chunks),
            output_file=str(output_path),
            errors=errors,
            stats=stats
        )
    
    def _process_document(self, doc: LoadedDocument, stats: Dict) -> List[Chunk]:
        """Process a single document through the pipeline."""

        # Step 2: Clean text
        print(f"   🧹 Cleaning text...")
        cleaned_text, cleaning_stats = self.cleaner.clean(doc.content)
        
        stats["total_original_chars"] += cleaning_stats.original_length
        stats["total_cleaned_chars"] += cleaning_stats.cleaned_length
        stats["documents"][doc.filename] = {
            "original_chars": cleaning_stats.original_length,
            "cleaned_chars": cleaning_stats.cleaned_length,
            "reduction": f"{(1 - cleaning_stats.cleaned_length/max(cleaning_stats.original_length, 1))*100:.1f}%"
        }
        
        # Step 3: Detect structure
        print(f"   🔍 Detecting structure...")
        elements = self.structure_detector.detect_structure(cleaned_text)
        sections = self.structure_detector.get_sections(elements)
        
        print(f"   📑 Found {len(sections)} sections")
        
        # Step 4: Chunk (try Chonkie first if activé)
        print(f"   ✂️  Chunking...")

        # Use page information if available (for PDFs)
        cleaned_pages = None
        if doc.pages:
            cleaned_pages = []
            for page in doc.pages:
                page_text, _ = self.cleaner.clean(page["text"])
                cleaned_pages.append({
                    "page": page["page"],
                    "text": page_text
                })

        if self._chonkie_adapter is not None:
            try:
                chonkie_chunks = self._chonkie_adapter.process_document(
                    text=cleaned_text,
                    doc=doc,
                    sections=sections,
                    pages=cleaned_pages,
                )
                if chonkie_chunks:
                    return chonkie_chunks
                print("   ⚠️ Chonkie n'a retourné aucun chunk; fallback sur chunker interne.")
            except Exception as e:
                print(f"   ⚠️ Chonkie a échoué: {e}. Fallback sur chunker interne.")

        # Fallback chunker interne
        if cleaned_pages:
            return self.chunker.chunk_document(
                text=cleaned_text,
                source=doc.filename,
                pages=cleaned_pages,
                sections=sections
            )
        return self.chunker.chunk_document(
            text=cleaned_text,
            source=doc.filename,
            sections=sections
        )
    
    def process_single_file(self, file_path: str) -> List[Chunk]:
        """
        Process a single file and return chunks (without saving).
        
        Useful for testing or real-time processing.
        """
        doc = self.loader.load_file(file_path)
        if not doc:
            return []
        
        stats = {"documents": {}}
        return self._process_document(doc, stats)


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest documents and generate chunks for RAG"
    )
    parser.add_argument(
        "-i", "--input",
        default="raw_docs",
        help="Input directory with documents (default: raw_docs)"
    )
    parser.add_argument(
        "-o", "--output",
        default="chunks.json",
        help="Output file path (default: chunks.json)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens per chunk (default: 500)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        help="Overlap tokens between chunks (default: 80)"
    )
    parser.add_argument(
        "--tiktoken",
        action="store_true",
        help="Use tiktoken for accurate token counting"
    )
    parser.add_argument(
        "--use-chonkie",
        action="store_true",
        help="Use Chonkie to extract richer elements (tables/images) when available"
    )
    
    args = parser.parse_args()
    
    config = IngestionConfig(
        input_dir=args.input,
        output_file=args.output,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        token_method="tiktoken" if args.tiktoken else "simple",
        use_chonkie=args.use_chonkie
    )
    
    pipeline = DataIngestionPipeline(config)
    result = pipeline.run()
    
    # Exit with appropriate code
    exit(0 if result.success else 1)


if __name__ == "__main__":
    main()

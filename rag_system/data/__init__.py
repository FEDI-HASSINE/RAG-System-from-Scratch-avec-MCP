"""
Data Ingestion Package

Modules for loading, cleaning, and chunking documents.
"""

from .loaders import UnifiedDocumentLoader, LoadedDocument
from .cleaner import TextCleaner, CleaningStats
from .structure_detector import StructureDetector, StructuredElement, ElementType
from .chunker import IntelligentChunker, Chunk, validate_chunks
from .ingestion_pipeline import DataIngestionPipeline, IngestionConfig, IngestionResult

__all__ = [
    # Loaders
    "UnifiedDocumentLoader",
    "LoadedDocument",
    
    # Cleaner
    "TextCleaner",
    "CleaningStats",
    
    # Structure
    "StructureDetector",
    "StructuredElement",
    "ElementType",
    
    # Chunker
    "IntelligentChunker",
    "Chunk",
    "validate_chunks",
    
    # Pipeline
    "DataIngestionPipeline",
    "IngestionConfig",
    "IngestionResult",
]

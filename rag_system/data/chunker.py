"""
Intelligent Chunker - Smart text chunking with overlap and sentence preservation
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: str
    source: str
    text: str
    tokens: int
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "tokens": self.tokens
        }
        if self.page is not None:
            result["page"] = self.page
        if self.section:
            result["section"] = self.section
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class TokenCounter:
    """Count tokens in text."""
    
    def __init__(self, method: str = "simple"):
        """
        Initialize token counter.
        
        Args:
            method: 'simple' for word-based, 'tiktoken' for OpenAI tokenizer
        """
        self.method = method
        self._tiktoken_encoder = None
        
        if method == "tiktoken":
            try:
                import tiktoken
                self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                print("⚠️ tiktoken not available, falling back to simple counting")
                self.method = "simple"
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self.method == "tiktoken" and self._tiktoken_encoder:
            return len(self._tiktoken_encoder.encode(text))
        else:
            # Simple word-based counting (roughly 1.3 tokens per word)
            words = len(text.split())
            return int(words * 1.3)


class SentenceSplitter:
    """Split text into sentences without breaking them."""
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Period/exclamation/question followed by capital
        r'(?<=[.!?])\s*\n+|'        # Period followed by newline
        r'\n{2,}'                    # Multiple newlines (paragraph break)
    )
    
    # Patterns that should NOT end a sentence
    ABBREVIATIONS = {
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
        'vs.', 'etc.', 'e.g.', 'i.e.', 'Fig.', 'fig.',
        'Vol.', 'vol.', 'No.', 'no.', 'Inc.', 'Ltd.', 'Corp.'
    }
    
    def split(self, text: str) -> List[str]:
        """Split text into sentences."""
        # First, protect abbreviations
        protected_text = text
        placeholder_map = {}
        
        for i, abbr in enumerate(self.ABBREVIATIONS):
            placeholder = f"__ABBR_{i}__"
            placeholder_map[placeholder] = abbr
            protected_text = protected_text.replace(abbr, placeholder)
        
        # Split on sentence boundaries
        sentences = self.SENTENCE_ENDINGS.split(protected_text)
        
        # Restore abbreviations and clean up
        result = []
        for sentence in sentences:
            for placeholder, abbr in placeholder_map.items():
                sentence = sentence.replace(placeholder, abbr)
            
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        return result


class IntelligentChunker:
    """
    Intelligent text chunker that respects document structure.
    
    Features:
    - Token-based chunking with configurable size
    - Overlap between chunks for context preservation
    - Never cuts sentences in the middle
    - Preserves section information
    - Handles page boundaries for PDFs
    """
    
    def __init__(self,
                 max_tokens: int = 500,
                 overlap_tokens: int = 80,
                 token_method: str = "simple"):
        """
        Initialize the chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk (default: 500)
            overlap_tokens: Token overlap between chunks (default: 80)
            token_method: 'simple' or 'tiktoken'
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = TokenCounter(method=token_method)
        self.sentence_splitter = SentenceSplitter()
    
    def chunk_document(self, 
                      text: str,
                      source: str,
                      pages: List[Dict] = None,
                      sections: List[Dict] = None) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            text: The document text
            source: Source filename
            pages: Optional list of page info [{"page": 1, "text": "..."}]
            sections: Optional list of section info from structure detector
        
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # If we have page information, chunk by page first
        if pages:
            chunks = self._chunk_with_pages(pages, source, sections)
        elif sections:
            chunks = self._chunk_with_sections(sections, source)
        else:
            chunks = self._chunk_text(text, source)
        
        return chunks
    
    def _chunk_text(self, text: str, source: str, 
                   page: int = None, section: str = None) -> List[Chunk]:
        """Chunk plain text into pieces."""
        chunks = []
        sentences = self.sentence_splitter.split(text)
        
        if not sentences:
            return chunks
        
        current_sentences = []
        current_tokens = 0
        chunk_idx = 0
        
        source_base = source.rsplit('.', 1)[0] if '.' in source else source
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.token_counter.count(sentence)
            
            # If a single sentence exceeds max tokens, we need to split it
            if sentence_tokens > self.max_tokens:
                # Flush current chunk first
                if current_sentences:
                    chunk_text = ' '.join(current_sentences)
                    chunks.append(Chunk(
                        chunk_id=f"{source_base}_{chunk_idx:03d}",
                        source=source,
                        text=chunk_text,
                        tokens=current_tokens,
                        page=page,
                        section=section
                    ))
                    chunk_idx += 1
                    current_sentences = []
                    current_tokens = 0
                
                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence, source_base, 
                                                       chunk_idx, source, page, section)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                i += 1
                continue
            
            # Check if adding this sentence exceeds max tokens
            if current_tokens + sentence_tokens > self.max_tokens:
                # Create chunk from current sentences
                if current_sentences:
                    chunk_text = ' '.join(current_sentences)
                    chunks.append(Chunk(
                        chunk_id=f"{source_base}_{chunk_idx:03d}",
                        source=source,
                        text=chunk_text,
                        tokens=current_tokens,
                        page=page,
                        section=section
                    ))
                    chunk_idx += 1
                    
                    # Calculate overlap - keep last few sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_sentences, self.overlap_tokens
                    )
                    current_sentences = overlap_sentences
                    current_tokens = sum(
                        self.token_counter.count(s) for s in current_sentences
                    )
                else:
                    current_sentences = []
                    current_tokens = 0
            
            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{source_base}_{chunk_idx:03d}",
                source=source,
                text=chunk_text,
                tokens=self.token_counter.count(chunk_text),
                page=page,
                section=section
            ))
        
        return chunks
    
    def _chunk_with_pages(self, pages: List[Dict], source: str,
                         sections: List[Dict] = None) -> List[Chunk]:
        """Chunk document considering page boundaries."""
        all_chunks = []
        
        for page_info in pages:
            page_num = page_info["page"]
            page_text = page_info["text"]
            
            # Determine section for this page content
            current_section = self._find_section_for_text(page_text, sections)
            
            page_chunks = self._chunk_text(
                page_text, source, page=page_num, section=current_section
            )
            all_chunks.extend(page_chunks)
        
        # Re-number chunks sequentially
        source_base = source.rsplit('.', 1)[0] if '.' in source else source
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"{source_base}_{i:03d}"
        
        return all_chunks
    
    def _chunk_with_sections(self, sections: List[Dict], source: str) -> List[Chunk]:
        """Chunk document by sections."""
        all_chunks = []
        
        for section in sections:
            section_name = section.get("header", "")
            elements = section.get("elements", [])
            
            # Combine elements into text
            section_text = ' '.join(
                elem.text if hasattr(elem, 'text') else str(elem) 
                for elem in elements
            )
            
            if section_text.strip():
                section_chunks = self._chunk_text(
                    section_text, source, section=section_name
                )
                all_chunks.extend(section_chunks)
        
        # Re-number chunks
        source_base = source.rsplit('.', 1)[0] if '.' in source else source
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"{source_base}_{i:03d}"
        
        return all_chunks
    
    def _get_overlap_sentences(self, sentences: List[str], 
                               target_tokens: int) -> List[str]:
        """Get sentences for overlap from the end of the list."""
        overlap = []
        current_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                overlap.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return overlap
    
    def _split_long_sentence(self, sentence: str, source_base: str,
                            start_idx: int, source: str,
                            page: int = None, section: str = None) -> List[Chunk]:
        """Split a sentence that's too long into smaller pieces."""
        chunks = []
        
        # Try to split on commas, semicolons, or colons
        parts = re.split(r'(?<=[,;:])\s+', sentence)
        
        current_text = []
        current_tokens = 0
        chunk_idx = start_idx
        
        for part in parts:
            part_tokens = self.token_counter.count(part)
            
            if current_tokens + part_tokens > self.max_tokens:
                if current_text:
                    text = ' '.join(current_text)
                    chunks.append(Chunk(
                        chunk_id=f"{source_base}_{chunk_idx:03d}",
                        source=source,
                        text=text,
                        tokens=self.token_counter.count(text),
                        page=page,
                        section=section
                    ))
                    chunk_idx += 1
                    current_text = []
                    current_tokens = 0
            
            current_text.append(part)
            current_tokens += part_tokens
        
        # Last piece
        if current_text:
            text = ' '.join(current_text)
            chunks.append(Chunk(
                chunk_id=f"{source_base}_{chunk_idx:03d}",
                source=source,
                text=text,
                tokens=self.token_counter.count(text),
                page=page,
                section=section
            ))
        
        return chunks
    
    def _find_section_for_text(self, text: str, 
                               sections: List[Dict] = None) -> Optional[str]:
        """Find which section a piece of text belongs to."""
        if not sections:
            return None
        
        # Simple heuristic: check which section's content overlaps most
        text_lower = text.lower()[:200]  # Check first 200 chars
        
        for section in sections:
            header = section.get("header", "")
            if header and header.lower() in text_lower:
                return header
        
        return None


def validate_chunks(chunks: List[Chunk], max_tokens: int = 500) -> Dict:
    """
    Validate chunks meet quality requirements.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "total_chunks": len(chunks),
        "valid": True,
        "issues": [],
        "stats": {
            "avg_tokens": 0,
            "min_tokens": float('inf'),
            "max_tokens": 0,
            "chunks_with_section": 0,
            "chunks_with_page": 0
        }
    }
    
    total_tokens = 0
    
    for chunk in chunks:
        total_tokens += chunk.tokens
        
        # Check token limit
        if chunk.tokens > max_tokens:
            results["valid"] = False
            results["issues"].append(
                f"Chunk {chunk.chunk_id} exceeds max tokens: {chunk.tokens} > {max_tokens}"
            )
        
        # Update stats
        results["stats"]["min_tokens"] = min(results["stats"]["min_tokens"], chunk.tokens)
        results["stats"]["max_tokens"] = max(results["stats"]["max_tokens"], chunk.tokens)
        
        if chunk.section:
            results["stats"]["chunks_with_section"] += 1
        if chunk.page:
            results["stats"]["chunks_with_page"] += 1
    
    if chunks:
        results["stats"]["avg_tokens"] = total_tokens / len(chunks)
    else:
        results["stats"]["min_tokens"] = 0
    
    return results


if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    Introduction to Data Protection

    The General Data Protection Regulation (GDPR) is a regulation in EU law on data 
    protection and privacy in the European Union and the European Economic Area. 
    It also addresses the transfer of personal data outside the EU and EEA areas.
    
    The GDPR aims primarily to give control to individuals over their personal data 
    and to simplify the regulatory environment for international business by unifying 
    the regulation within the EU. This regulation has become a model for many other 
    data protection laws around the world.
    
    Key Principles
    
    Personal data must be processed lawfully, fairly and in a transparent manner. 
    Data should be collected for specified, explicit and legitimate purposes. 
    The data collected should be adequate, relevant and limited to what is necessary.
    
    Organizations must ensure that personal data is accurate and kept up to date. 
    Data should not be kept longer than necessary for the purposes for which it was collected.
    Personal data must be processed in a manner that ensures appropriate security.
    """
    
    chunker = IntelligentChunker(max_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk_document(sample_text, "gdpr_overview.txt")
    
    print("=== Generated Chunks ===")
    for chunk in chunks:
        print(f"\n{chunk.chunk_id} ({chunk.tokens} tokens):")
        print(f"  {chunk.text[:100]}...")
    
    print("\n=== Validation ===")
    validation = validate_chunks(chunks, max_tokens=100)
    print(f"Valid: {validation['valid']}")
    print(f"Stats: {validation['stats']}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")

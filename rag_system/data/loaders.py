"""
Document Loaders - Unified loader for PDF, Markdown, and TXT files
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LoadedDocument:
    """Represents a loaded document with metadata."""
    filename: str
    file_type: str
    content: str
    pages: List[Dict] = field(default_factory=list)  # For PDFs: [{"page": 1, "text": "..."}]
    metadata: Dict = field(default_factory=dict)


class PDFLoader:
    """Load and extract text from PDF files."""
    
    def __init__(self):
        try:
            import pypdf
            self.pypdf = pypdf
        except ImportError:
            raise ImportError("pypdf is required. Install with: pip install pypdf")
    
    def load(self, file_path: str) -> LoadedDocument:
        """Extract text from PDF file with page information."""
        pages = []
        full_text = []
        
        with open(file_path, 'rb') as f:
            reader = self.pypdf.PdfReader(f)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "page": page_num,
                    "text": text
                })
                full_text.append(text)
        
        return LoadedDocument(
            filename=os.path.basename(file_path),
            file_type="pdf",
            content="\n\n".join(full_text),
            pages=pages,
            metadata={"total_pages": len(pages)}
        )


class MarkdownLoader:
    """Load and parse Markdown files."""
    
    def __init__(self):
        try:
            from markdown_it import MarkdownIt
            self.md = MarkdownIt()
        except ImportError:
            self.md = None
    
    def load(self, file_path: str) -> LoadedDocument:
        """Load markdown file and extract clean text."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract headers for section detection
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        
        # Convert markdown to plain text (remove formatting)
        clean_text = self._markdown_to_text(content)
        
        return LoadedDocument(
            filename=os.path.basename(file_path),
            file_type="markdown",
            content=clean_text,
            metadata={
                "headers": [{"level": len(h[0]), "text": h[1]} for h in headers]
            }
        )
    
    def _markdown_to_text(self, md_content: str) -> str:
        """Convert markdown to plain text."""
        text = md_content
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        
        # Convert links to just text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert headers to text (keep the text, remove #)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()


class TextLoader:
    """Load plain text files."""
    
    def load(self, file_path: str) -> LoadedDocument:
        """Load text file directly."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return LoadedDocument(
            filename=os.path.basename(file_path),
            file_type="txt",
            content=content,
            metadata={}
        )


class UnifiedDocumentLoader:
    """Unified loader that handles multiple document types."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.txt', '.markdown'}
    
    def __init__(self):
        self.pdf_loader = None
        self.md_loader = MarkdownLoader()
        self.txt_loader = TextLoader()
    
    def _get_pdf_loader(self):
        """Lazy load PDF loader."""
        if self.pdf_loader is None:
            self.pdf_loader = PDFLoader()
        return self.pdf_loader
    
    def load_file(self, file_path: str) -> Optional[LoadedDocument]:
        """Load a single file based on its extension."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"⚠️ Unsupported file type: {ext}")
            return None
        
        try:
            if ext == '.pdf':
                return self._get_pdf_loader().load(file_path)
            elif ext in ['.md', '.markdown']:
                return self.md_loader.load(file_path)
            elif ext == '.txt':
                return self.txt_loader.load(file_path)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def load_directory(self, dir_path: str) -> List[LoadedDocument]:
        """Load all supported documents from a directory."""
        documents = []
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            print(f"❌ Directory not found: {dir_path}")
            return documents
        
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                print(f"📄 Loading: {file_path.name}")
                doc = self.load_file(str(file_path))
                if doc:
                    documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents


if __name__ == "__main__":
    # Test the loaders
    loader = UnifiedDocumentLoader()
    docs = loader.load_directory("raw_docs")
    for doc in docs:
        print(f"- {doc.filename}: {len(doc.content)} chars")

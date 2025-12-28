"""
Text Cleaner - Advanced text cleaning and normalization
"""

import re
from typing import List, Set
from dataclasses import dataclass


@dataclass
class CleaningStats:
    """Statistics about the cleaning process."""
    original_length: int
    cleaned_length: int
    removed_headers: int
    removed_page_numbers: int
    removed_short_lines: int
    normalized_spaces: int


class TextCleaner:
    """Advanced text cleaner for document preprocessing."""
    
    # Common header/footer patterns to remove
    HEADER_PATTERNS = [
        r'^(Page|PAGE)\s*\d+\s*(of|OF|/)\s*\d+\s*$',  # Page 1 of 10
        r'^\d+\s*(of|OF|/)\s*\d+\s*$',  # 1 of 10
        r'^-\s*\d+\s*-\s*$',  # - 1 -
        r'^\[\s*\d+\s*\]\s*$',  # [1]
        r'^(CONFIDENTIAL|DRAFT|INTERNAL)\s*$',  # Document markers
        r'^(Copyright|©).*\d{4}.*$',  # Copyright notices
        r'^(All rights reserved).*$',
        r'^\d{1,2}/\d{1,2}/\d{2,4}\s*$',  # Dates alone
        r'^(www\.|http|https).*$',  # URLs alone on a line
    ]
    
    # Page number patterns
    PAGE_NUMBER_PATTERNS = [
        r'^\s*\d{1,4}\s*$',  # Just a number
        r'^\s*-\s*\d{1,4}\s*-\s*$',  # - 5 -
        r'^\s*\[\d{1,4}\]\s*$',  # [5]
        r'^\s*\(\d{1,4}\)\s*$',  # (5)
    ]
    
    def __init__(self, 
                 min_line_length: int = 20,
                 remove_urls: bool = True,
                 remove_emails: bool = False,
                 normalize_unicode: bool = True):
        self.min_line_length = min_line_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        
        # Compile patterns for efficiency
        self.header_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                            for p in self.HEADER_PATTERNS]
        self.page_regex = [re.compile(p, re.MULTILINE) 
                          for p in self.PAGE_NUMBER_PATTERNS]
    
    def clean(self, text: str) -> tuple[str, CleaningStats]:
        """
        Clean text with full statistics.
        
        Returns:
            Tuple of (cleaned_text, stats)
        """
        original_length = len(text)
        removed_headers = 0
        removed_page_numbers = 0
        removed_short_lines = 0
        normalized_spaces = 0
        
        # Step 1: Normalize unicode characters
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Remove headers/footers
        text, removed_headers = self._remove_headers(text)
        
        # Step 3: Remove page numbers
        text, removed_page_numbers = self._remove_page_numbers(text)
        
        # Step 4: Remove URLs if configured
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Step 5: Remove emails if configured
        if self.remove_emails:
            text = self._remove_emails(text)
        
        # Step 6: Remove short/useless lines
        text, removed_short_lines = self._remove_short_lines(text)
        
        # Step 7: Normalize whitespace
        text, normalized_spaces = self._normalize_whitespace(text)
        
        # Step 8: Remove empty table remnants
        text = self._remove_empty_tables(text)
        
        # Step 9: Final cleanup
        text = self._final_cleanup(text)
        
        stats = CleaningStats(
            original_length=original_length,
            cleaned_length=len(text),
            removed_headers=removed_headers,
            removed_page_numbers=removed_page_numbers,
            removed_short_lines=removed_short_lines,
            normalized_spaces=normalized_spaces
        )
        
        return text, stats
    
    def clean_simple(self, text: str) -> str:
        """Simple cleaning without statistics."""
        cleaned, _ = self.clean(text)
        return cleaned
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        
        # Normalize to NFKC form
        text = unicodedata.normalize('NFKC', text)
        
        # Replace common problematic characters
        replacements = {
            '\u2019': "'",  # Right single quote
            '\u2018': "'",  # Left single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',   # Zero-width space
            '\ufeff': '',   # BOM
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_headers(self, text: str) -> tuple[str, int]:
        """Remove common header/footer patterns."""
        count = 0
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_header = False
            for pattern in self.header_regex:
                if pattern.match(line.strip()):
                    is_header = True
                    count += 1
                    break
            
            if not is_header:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines), count
    
    def _remove_page_numbers(self, text: str) -> tuple[str, int]:
        """Remove standalone page numbers."""
        count = 0
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_page_num = False
            stripped = line.strip()
            
            for pattern in self.page_regex:
                if pattern.match(stripped):
                    is_page_num = True
                    count += 1
                    break
            
            if not is_page_num:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines), count
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def _remove_short_lines(self, text: str) -> tuple[str, int]:
        """Remove lines shorter than min_line_length."""
        count = 0
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Keep empty lines for paragraph detection
            # Remove short non-empty lines (likely noise)
            if stripped == '' or len(stripped) >= self.min_line_length:
                cleaned_lines.append(line)
            else:
                # Check if it's a meaningful short line (e.g., a header)
                if self._is_meaningful_short_line(stripped):
                    cleaned_lines.append(line)
                else:
                    count += 1
        
        return '\n'.join(cleaned_lines), count
    
    def _is_meaningful_short_line(self, line: str) -> bool:
        """Check if a short line is meaningful (like a header)."""
        # Headers often end with colon
        if line.endswith(':'):
            return True
        
        # Numbered items
        if re.match(r'^\d+[\.\)]\s*\w+', line):
            return True
        
        # Bullet points with content
        if re.match(r'^[-•*]\s+\w+', line):
            return True
        
        # All caps (likely a section header)
        if line.isupper() and len(line) > 3:
            return True
        
        return False
    
    def _normalize_whitespace(self, text: str) -> tuple[str, int]:
        """Normalize multiple spaces and blank lines."""
        original = text
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Count normalizations (rough estimate)
        count = abs(len(original) - len(text))
        
        return text, count
    
    def _remove_empty_tables(self, text: str) -> str:
        """Remove remnants of empty or broken tables."""
        # Remove lines that are just separators
        text = re.sub(r'^[\|\-\+]+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove lines with only table cell separators
        text = re.sub(r'^\s*\|\s*\|\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure single newline at end
        if text and not text.endswith('\n'):
            text += '\n'
        
        return text


if __name__ == "__main__":
    # Test the cleaner
    sample_text = """
    Page 1 of 10
    
    CONFIDENTIAL
    
    This is a sample document about data protection.
    
    - 5 -
    
    Personal data must be handled carefully.
    
    xy
    
    https://example.com/privacy
    
    © 2024 Company Name. All rights reserved.
    
    The GDPR requires companies to protect user data.
    """
    
    cleaner = TextCleaner()
    cleaned, stats = cleaner.clean(sample_text)
    
    print("=== Cleaned Text ===")
    print(cleaned)
    print("\n=== Stats ===")
    print(f"Original: {stats.original_length} chars")
    print(f"Cleaned: {stats.cleaned_length} chars")
    print(f"Removed headers: {stats.removed_headers}")
    print(f"Removed page numbers: {stats.removed_page_numbers}")
    print(f"Removed short lines: {stats.removed_short_lines}")

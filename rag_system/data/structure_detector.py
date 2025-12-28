"""
Structure Detector - Identify document structure (headers, paragraphs, lists)
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ElementType(Enum):
    """Types of document elements."""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    TABLE = "table"
    UNKNOWN = "unknown"


@dataclass
class StructuredElement:
    """A structured element from the document."""
    type: ElementType
    text: str
    level: int = 0  # For headers: 1-6, for lists: nesting level
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "text": self.text,
            "level": self.level,
            "metadata": self.metadata
        }


class StructureDetector:
    """Detect document structure from cleaned text."""
    
    # Header patterns (ordered by priority)
    HEADER_PATTERNS = [
        # Markdown style: # Header
        (r'^(#{1,6})\s+(.+)$', lambda m: (len(m.group(1)), m.group(2))),
        
        # Numbered headers: 1. Header or 1.1 Header
        (r'^(\d+(?:\.\d+)*)\s+([A-Z][^.!?]*?)$', lambda m: (1, f"{m.group(1)} {m.group(2)}")),
        
        # ALL CAPS headers
        (r'^([A-Z][A-Z\s]{3,50})$', lambda m: (2, m.group(1).strip())),
        
        # Title Case line followed by blank (detected contextually)
        (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$', lambda m: (2, m.group(1))),
    ]
    
    # List patterns
    LIST_PATTERNS = [
        r'^\s*([-•*])\s+(.+)$',  # Bullet lists
        r'^\s*(\d+)[\.\)]\s+(.+)$',  # Numbered lists
        r'^\s*([a-z])[\.\)]\s+(.+)$',  # Lettered lists
    ]
    
    def __init__(self, 
                 detect_headers: bool = True,
                 detect_lists: bool = True,
                 min_paragraph_length: int = 50):
        self.detect_headers = detect_headers
        self.detect_lists = detect_lists
        self.min_paragraph_length = min_paragraph_length
    
    def detect_structure(self, text: str) -> List[StructuredElement]:
        """
        Detect structure in text and return list of structured elements.
        """
        elements = []
        lines = text.split('\n')
        
        current_paragraph = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                # Flush current paragraph if any
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    elements.append(StructuredElement(
                        type=ElementType.PARAGRAPH,
                        text=para_text
                    ))
                    current_paragraph = []
                i += 1
                continue
            
            # Check for headers
            if self.detect_headers:
                header = self._detect_header(stripped, lines, i)
                if header:
                    # Flush paragraph first
                    if current_paragraph:
                        para_text = ' '.join(current_paragraph)
                        elements.append(StructuredElement(
                            type=ElementType.PARAGRAPH,
                            text=para_text
                        ))
                        current_paragraph = []
                    
                    elements.append(header)
                    i += 1
                    continue
            
            # Check for list items
            if self.detect_lists:
                list_item = self._detect_list_item(stripped)
                if list_item:
                    # Flush paragraph first
                    if current_paragraph:
                        para_text = ' '.join(current_paragraph)
                        elements.append(StructuredElement(
                            type=ElementType.PARAGRAPH,
                            text=para_text
                        ))
                        current_paragraph = []
                    
                    elements.append(list_item)
                    i += 1
                    continue
            
            # Regular text - add to current paragraph
            current_paragraph.append(stripped)
            i += 1
        
        # Flush remaining paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            elements.append(StructuredElement(
                type=ElementType.PARAGRAPH,
                text=para_text
            ))
        
        return elements
    
    def _detect_header(self, line: str, all_lines: List[str], 
                       current_idx: int) -> Optional[StructuredElement]:
        """Detect if a line is a header."""
        
        # Check markdown-style headers
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            return StructuredElement(
                type=ElementType.HEADER,
                text=md_match.group(2).strip(),
                level=len(md_match.group(1))
            )
        
        # Check ALL CAPS headers (min 4 chars, max 60)
        if (line.isupper() and 
            4 <= len(line) <= 60 and 
            not line.startswith(('-', '*', '•'))):
            return StructuredElement(
                type=ElementType.HEADER,
                text=line,
                level=2
            )
        
        # Check numbered section headers: "1. Introduction" or "1.1 Overview"
        num_match = re.match(r'^(\d+(?:\.\d+)*)[.\s]+([A-Z][^\n]{2,50})$', line)
        if num_match:
            section_num = num_match.group(1)
            depth = len(section_num.split('.'))
            return StructuredElement(
                type=ElementType.HEADER,
                text=num_match.group(2).strip(),
                level=min(depth, 6),
                metadata={"section_number": section_num}
            )
        
        # Check for short Title Case lines followed by longer text
        if self._is_title_case(line) and len(line) <= 60:
            # Look ahead - if next non-empty line is much longer, this is likely a header
            next_idx = current_idx + 1
            while next_idx < len(all_lines) and not all_lines[next_idx].strip():
                next_idx += 1
            
            if next_idx < len(all_lines):
                next_line = all_lines[next_idx].strip()
                if len(next_line) > len(line) * 2 and len(next_line) > 50:
                    return StructuredElement(
                        type=ElementType.HEADER,
                        text=line,
                        level=3
                    )
        
        return None
    
    def _detect_list_item(self, line: str) -> Optional[StructuredElement]:
        """Detect if a line is a list item."""
        
        # Bullet list
        bullet_match = re.match(r'^([-•*])\s+(.+)$', line)
        if bullet_match:
            return StructuredElement(
                type=ElementType.LIST_ITEM,
                text=bullet_match.group(2),
                level=1,
                metadata={"bullet": bullet_match.group(1)}
            )
        
        # Numbered list
        num_match = re.match(r'^(\d+)[\.\)]\s+(.+)$', line)
        if num_match:
            return StructuredElement(
                type=ElementType.LIST_ITEM,
                text=num_match.group(2),
                level=1,
                metadata={"number": int(num_match.group(1))}
            )
        
        # Lettered list
        letter_match = re.match(r'^([a-zA-Z])[\.\)]\s+(.+)$', line)
        if letter_match and len(letter_match.group(1)) == 1:
            return StructuredElement(
                type=ElementType.LIST_ITEM,
                text=letter_match.group(2),
                level=1,
                metadata={"letter": letter_match.group(1)}
            )
        
        return None
    
    def _is_title_case(self, text: str) -> bool:
        """Check if text is in Title Case."""
        words = text.split()
        if len(words) < 2:
            return False
        
        # Common words that don't need to be capitalized
        skip_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'from', 'as'}
        
        capitalized = 0
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word:
                continue
            
            # First word or non-skip word should be capitalized
            if i == 0 or word.lower() not in skip_words:
                if clean_word[0].isupper():
                    capitalized += 1
        
        # At least 60% of significant words should be capitalized
        return capitalized / len(words) >= 0.6 if words else False
    
    def get_sections(self, elements: List[StructuredElement]) -> List[Dict]:
        """
        Group elements by sections based on headers.
        Returns list of sections with their content.
        """
        sections = []
        current_section = {
            "header": None,
            "level": 0,
            "elements": []
        }
        
        for element in elements:
            if element.type == ElementType.HEADER:
                # Save previous section if it has content
                if current_section["elements"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "header": element.text,
                    "level": element.level,
                    "elements": []
                }
            else:
                current_section["elements"].append(element)
        
        # Don't forget the last section
        if current_section["elements"] or current_section["header"]:
            sections.append(current_section)
        
        return sections


if __name__ == "__main__":
    # Test the structure detector
    sample_text = """
# Introduction

This document describes our data protection policies.
We take privacy very seriously.

## User Data

Personal information includes:
- Name and email address
- Phone number
- Billing information

1. Users must consent to data collection
2. Data must be encrypted
3. Access is logged

GDPR COMPLIANCE

The General Data Protection Regulation requires that we:
- Obtain explicit consent
- Allow data portability
- Enable right to be forgotten
"""
    
    detector = StructureDetector()
    elements = detector.detect_structure(sample_text)
    
    print("=== Detected Elements ===")
    for elem in elements:
        print(f"[{elem.type.value}] (L{elem.level}): {elem.text[:50]}...")
    
    print("\n=== Sections ===")
    sections = detector.get_sections(elements)
    for section in sections:
        print(f"Section: {section['header']} ({len(section['elements'])} elements)")

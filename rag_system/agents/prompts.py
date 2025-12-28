"""
Prompt Templates - Jinja2-based templates for RAG prompts

Provides:
- System prompts for different agent roles
- RAG answer prompts with context
- Structured output templates
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# ============================================================
# Template Strings
# ============================================================

# System prompt for RAG assistant
RAG_SYSTEM_PROMPT = """Tu es un assistant expert qui répond aux questions en utilisant UNIQUEMENT les informations fournies dans le contexte.

Règles strictes :
1. Réponds UNIQUEMENT avec les informations du contexte fourni
2. Si l'information n'est pas dans le contexte, dis-le clairement
3. Ne fabrique jamais d'informations (pas d'hallucination)
4. Cite les sources quand c'est pertinent
5. Sois concis et précis
6. Réponds dans la même langue que la question"""


# Main RAG prompt template
RAG_PROMPT_TEMPLATE = """Utilise les informations suivantes pour répondre à la question.

CONTEXTE:
{context}

SOURCES UTILISÉES:
{sources}

QUESTION:
{question}

INSTRUCTIONS:
- Réponds uniquement avec les informations du contexte
- Si tu ne trouves pas l'information, dis "Je n'ai pas trouvé cette information dans les documents disponibles."
- Cite les sources pertinentes

RÉPONSE:"""


# Template for when no context is found
NO_CONTEXT_TEMPLATE = """Je n'ai pas trouvé d'informations pertinentes dans les documents disponibles pour répondre à votre question.

Question posée : {question}

Suggestions :
- Reformulez votre question avec des termes différents
- Vérifiez que les documents contiennent cette information
- Posez une question plus spécifique"""


# Template for structured output
STRUCTURED_OUTPUT_TEMPLATE = """Basé sur le contexte fourni, réponds au format suivant:

RÉSUMÉ: (réponse concise en 1-2 phrases)
DÉTAILS: (explication plus détaillée si nécessaire)
SOURCES: (liste des sources utilisées)
CONFIANCE: (élevée/moyenne/faible selon la qualité du contexte)

CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE STRUCTURÉE:"""


# ============================================================
# Template Classes
# ============================================================

@dataclass
class PromptTemplate:
    """A prompt template with variable substitution."""
    template: str
    name: str = "default"
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)
    
    def get_variables(self) -> List[str]:
        """Extract variable names from template."""
        import re
        return re.findall(r'\{(\w+)\}', self.template)


class PromptLibrary:
    """
    Library of prompt templates for RAG.
    
    Usage:
        library = PromptLibrary()
        prompt = library.get_rag_prompt(
            context="...",
            sources=["doc1.pdf", "doc2.md"],
            question="What is data protection?"
        )
    """
    
    def __init__(self):
        self.templates = {
            "rag_system": PromptTemplate(
                template=RAG_SYSTEM_PROMPT,
                name="rag_system",
                description="System prompt for RAG assistant"
            ),
            "rag_answer": PromptTemplate(
                template=RAG_PROMPT_TEMPLATE,
                name="rag_answer",
                description="Main RAG answer prompt with context"
            ),
            "no_context": PromptTemplate(
                template=NO_CONTEXT_TEMPLATE,
                name="no_context",
                description="Response when no relevant context found"
            ),
            "structured": PromptTemplate(
                template=STRUCTURED_OUTPUT_TEMPLATE,
                name="structured",
                description="Structured output format"
            )
        }
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get template by name."""
        if name not in self.templates:
            raise KeyError(f"Template not found: {name}")
        return self.templates[name]
    
    def get_system_prompt(self) -> str:
        """Get the RAG system prompt."""
        return RAG_SYSTEM_PROMPT
    
    def get_rag_prompt(self,
                       context: str,
                       question: str,
                       sources: Optional[List[str]] = None) -> str:
        """
        Build complete RAG prompt.
        
        Args:
            context: Retrieved context text
            question: User question
            sources: List of source documents
            
        Returns:
            Formatted prompt string
        """
        sources_str = "\n".join(f"- {s}" for s in (sources or ["Non spécifié"]))
        
        return RAG_PROMPT_TEMPLATE.format(
            context=context,
            sources=sources_str,
            question=question
        )
    
    def get_no_context_response(self, question: str) -> str:
        """Get response for when no context is found."""
        return NO_CONTEXT_TEMPLATE.format(question=question)
    
    def get_structured_prompt(self, context: str, question: str) -> str:
        """Get structured output prompt."""
        return STRUCTURED_OUTPUT_TEMPLATE.format(
            context=context,
            question=question
        )
    
    def format_context_from_chunks(self, 
                                    chunks: List[Dict],
                                    max_chunks: int = 5) -> tuple:
        """
        Format context string from retrieved chunks.
        
        Args:
            chunks: List of chunk dictionaries
            max_chunks: Maximum chunks to include
            
        Returns:
            Tuple of (context_string, sources_list)
        """
        chunks = chunks[:max_chunks]
        
        context_parts = []
        sources = set()
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            section = chunk.get("section", "")
            
            # Add source info
            sources.add(source)
            
            # Format chunk
            header = f"[{i}] Source: {source}"
            if section:
                header += f" | Section: {section}"
            
            context_parts.append(f"{header}\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, list(sources)
    
    def register_template(self, 
                          name: str, 
                          template: str,
                          description: str = "") -> None:
        """Register a custom template."""
        self.templates[name] = PromptTemplate(
            template=template,
            name=name,
            description=description
        )


# ============================================================
# Global Instance
# ============================================================

_library_instance: Optional[PromptLibrary] = None


def get_prompt_library() -> PromptLibrary:
    """Get global prompt library instance."""
    global _library_instance
    if _library_instance is None:
        _library_instance = PromptLibrary()
    return _library_instance


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing Prompt Templates...")
    print("=" * 60)
    
    library = PromptLibrary()
    
    print("\n📝 Test 1: RAG System Prompt")
    system_prompt = library.get_system_prompt()
    print(f"   Length: {len(system_prompt)} chars")
    print(f"   Preview: {system_prompt[:100]}...")
    
    print("\n📝 Test 2: RAG Prompt with context")
    context = """
    Les données sont protégées par chiffrement TLS 1.3.
    L'authentification multi-facteurs est requise.
    """
    prompt = library.get_rag_prompt(
        context=context,
        question="Comment sont protégées les données ?",
        sources=["privacy_policy.md", "security.txt"]
    )
    print(f"   Length: {len(prompt)} chars")
    print(f"   Preview:\n{prompt[:300]}...")
    
    print("\n📝 Test 3: Format chunks")
    chunks = [
        {"text": "Premier chunk de texte.", "source": "doc1.pdf", "section": "Introduction"},
        {"text": "Deuxième chunk de texte.", "source": "doc2.md", "section": "Détails"}
    ]
    context, sources = library.format_context_from_chunks(chunks)
    print(f"   Context length: {len(context)} chars")
    print(f"   Sources: {sources}")
    
    print("\n📝 Test 4: No context response")
    response = library.get_no_context_response("Quelle est la météo ?")
    print(f"   Preview: {response[:150]}...")
    
    print("\n✅ Prompt templates tests complete!")

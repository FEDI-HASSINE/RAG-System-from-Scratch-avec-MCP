"""
Chonkie Adapter

Ce module fournit une interface d'intégration optionnelle avec Chonkie
(Open Source Data Ingestion for AI) pour extraire des éléments riches
(tables, images + légendes, figures) et produire des chunks plus pertinents.

Remarque: Chonkie n'est pas une dépendance obligatoire. Ce module
échoue proprement si la librairie n'est pas installée.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .chunker import Chunk, IntelligentChunker, TokenCounter


class ChonkieNotAvailable(RuntimeError):
    pass


@dataclass
class _ChonkieElement:
    """Représente un élément structuré renvoyé par Chonkie (forme générique)."""
    type: str  # "text" | "table" | "image" | "code" | ...
    text: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    bbox: Optional[List[float]] = None  # [x1,y1,x2,y2] si dispo
    caption: Optional[str] = None
    table_markdown: Optional[str] = None
    table_html: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChonkieAdapter:
    """
    Adapter minimal pour brancher Chonkie dans la pipeline.

    - Si Chonkie est installé, `process_file()` doit appeler la pipeline Chonkie
      et convertir ses sorties en `Chunk` en préservant les métadonnées utiles.
    - Si non installé, l'adapter lève une erreur claire et la pipeline retombe
      sur le chemin texte classique.
    """

    def __init__(self, 
                 max_tokens: int = 500, 
                 overlap_tokens: int = 80,
                 token_method: str = "simple") -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = TokenCounter(method=token_method)
        self._chunker = None
        self._import_error: Optional[Exception] = None

        # Tentative d'import léger pour détecter la présence de Chonkie
        try:
            from chonkie import RecursiveChunker

            # On choisit RecursiveChunker (structure hiérarchique) avec un chunk_size
            # aligné sur notre config. La notion de token dépend du tokenizer interne;
            # on laisse les defaults de Chonkie pour la robustesse.
            self._chunker = RecursiveChunker(chunk_size=max_tokens)
        except Exception as e:
            self._chunker = None
            self._import_error = e

    def process_document(
        self,
        text: str,
        doc,
        sections: Optional[List[Dict]] = None,
        pages: Optional[List[Dict]] = None,
    ) -> List[Chunk]:
        """Chunk un document (texte déjà nettoyé) avec Chonkie si dispo."""

        if self._chunker is None:
            raise ChonkieNotAvailable(
                "La librairie 'chonkie' n'est pas installée ou a échoué à l'import.\n"
                f"Détail: {self._import_error}"
            )

        try:
            chonkie_chunks = self._chunker(text)
        except Exception as e:
            raise ChonkieNotAvailable(f"Échec du chunking Chonkie: {e}") from e

        # Convertir en Chunk maison
        base = doc.filename.rsplit('.', 1)[0] if '.' in doc.filename else doc.filename
        results: List[Chunk] = []

        for idx, ch in enumerate(chonkie_chunks):
            chunk_text = getattr(ch, "text", None) or str(ch)
            token_count = getattr(ch, "token_count", None)
            if token_count is None:
                token_count = self.token_counter.count(chunk_text)

            metadata: Dict[str, Any] = {}
            ch_meta = getattr(ch, "metadata", None)
            if ch_meta:
                try:
                    metadata.update(dict(ch_meta))
                except Exception:
                    pass
            metadata["chunker"] = "chonkie_recursive"
            if sections:
                # Heuristique minimale: ranger le nom de fichier + éventuelle première section
                metadata.setdefault("section_hint", sections[0].get("header") if sections and sections[0].get("header") else None)
            if pages:
                # Si nous avons des pages nettoyées, tenter une attribution grossière en fonction du nombre de chunks
                # (optionnel, mieux vaudrait un alignement basé sur offsets si dispo)
                page_num = pages[min(idx // max(1, len(chonkie_chunks)//max(1,len(pages))), len(pages)-1)]["page"] if pages else None
                if page_num is not None:
                    metadata["page"] = page_num

            results.append(
                Chunk(
                    chunk_id=f"{base}_{idx:03d}",
                    source=doc.filename,
                    text=chunk_text,
                    tokens=token_count,
                    page=metadata.get("page"),
                    section=metadata.get("section_hint"),
                    metadata=metadata,
                )
            )

        return results

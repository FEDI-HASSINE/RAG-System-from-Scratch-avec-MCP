# 🧰 MCP Tools — Implémentations (Phases 3-4)

## 🎯 Objectif

Fournir les outils atomiques du pipeline RAG, exposables via MCP.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Embedding à la demande | `embed_text.py` — Vectorisation single/batch |
| Recherche sémantique | `retrieve_chunks.py` — Query → top-k chunks |
| Pertinence faible | `rerank.py` — Cross-encoder pour réordonner |
| Lazy loading coûteux | Singletons avec chargement différé |

## 📁 Fichiers

```
tools/
├── embed_text.py       # Outil d'embedding
├── retrieve_chunks.py  # Outil de recherche
└── rerank.py           # Outil de reranking
```

## 🔧 embed_text

Vectorise un ou plusieurs textes.

```python
from mcp_server.tools.embed_text import get_embed_text_tool

tool = get_embed_text_tool()
result = tool.execute({"text": "Hello world"})
# {"embedding": [0.12, -0.34, ...], "dimension": 384}
```

## 🔍 retrieve_chunks

Recherche les chunks les plus pertinents.

```python
from mcp_server.tools.retrieve_chunks import get_retrieve_chunks_tool

tool = get_retrieve_chunks_tool()
result = tool.execute({
    "query": "system architecture",
    "top_k": 5,
    "source_filter": "notes.txt"  # optionnel
})
# {"chunks": [...], "total_found": 5}
```

## 📊 rerank

Réordonne les chunks avec un Cross-Encoder.

```python
from mcp_server.tools.rerank import get_rerank_tool

tool = get_rerank_tool()
result = tool.execute({
    "query": "security measures",
    "chunks": [...],  # chunks from retrieve
    "top_k": 3
})
# {"chunks": [...], "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
```

## 🤖 Modèles utilisés

| Outil | Modèle | Usage |
|-------|--------|-------|
| embed_text | `all-MiniLM-L6-v2` | Embedding rapide (384 dims) |
| retrieve_chunks | FAISS + embedding | Recherche vectorielle |
| rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Scoring de pertinence |

## ⚡ Performance

- Premier appel : ~5-15s (chargement modèles)
- Appels suivants : ~10-100ms
- Les modèles sont gardés en mémoire (singletons)

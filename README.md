# 🔍 RAG System from Scratch avec MCP

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un système **RAG (Retrieval-Augmented Generation)** complet, construit de zéro avec architecture **MCP (Model Context Protocol)**.

## ✨ Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS et SentenceTransformers
- 📊 **Reranking** avec Cross-Encoder pour une pertinence maximale
- 🤖 **Agent RAG** orchestrant le pipeline complet
- 🔌 **API MCP** unifiée pour tous les outils
- 🛡️ **Zero hallucination** — réponses basées uniquement sur les documents
- 📈 **Évaluation LLM-as-a-Judge** avec métriques standardisées
- 💻 **CLI + UI Streamlit** pour la démo

## 🚀 Démarrage rapide

### 1. Cloner et installer
```bash
git clone https://github.com/FEDI-HASSINE/RAG-System-from-Scratch-avec-MCP.git
cd RAG-System-from-Scratch-avec-MCP
pip install -r rag_system/mcp_server/requirements.txt
pip install typer rich streamlit pandas matplotlib
```

### 2. Démarrer le serveur MCP
```bash
cd rag_system/mcp_server
uvicorn main:app --reload --port 8000
```

### 3. Tester via CLI
```bash
cd rag_system/demo
python rag_cli.py ask "What is system architecture?" --top-k 3
```

### 4. Lancer l'interface web
```bash
cd rag_system/demo
streamlit run app.py
# Ouvrez http://localhost:8501
```

## 📦 Architecture

```
RAG-System-from-Scratch-avec-MCP/
└── rag_system/
    ├── data/              # Phase 1: Ingestion & chunking
    ├── embeddings/        # Phase 2: Vectorisation
    ├── vector_store/      # Phase 2: Index FAISS
    ├── mcp_server/        # Phase 5: API MCP
    │   └── tools/         # Phases 3-4: Outils RAG
    ├── agents/            # Phase 6: Agent RAG
    ├── evaluation/        # Phase 7: Métriques
    └── demo/              # Phase 8: CLI + UI
```

## 🔄 Pipeline RAG

```
Question
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Embed      │ ──▶ │   Retrieve   │ ──▶ │   Rerank     │
│   (384 dims) │     │   (FAISS)    │     │ (CrossEnc)   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Réponse    │ ◀── │   LLM        │
                     │   + Sources  │     │   Generate   │
                     └──────────────┘     └──────────────┘
```

## 🛠️ Technologies

| Composant | Technologie |
|-----------|-------------|
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Reranking | Cross-Encoder `ms-marco-MiniLM-L-6-v2` |
| API | FastAPI + MCP Protocol |
| LLM | OpenAI / Ollama / Mock |
| CLI | Typer + Rich |
| UI | Streamlit |
| Evaluation | LLM-as-a-Judge |

## 📊 Métriques d'évaluation

| Métrique | Description |
|----------|-------------|
| **Groundedness** | Réponse basée sur le contexte |
| **Relevance** | Répond à la question |
| **Faithfulness** | Pas d'hallucination |

## 📖 Documentation

Chaque dossier contient un `README.md` détaillé :
- [rag_system/](rag_system/README.md) — Vue d'ensemble
- [data/](rag_system/data/README.md) — Ingestion
- [embeddings/](rag_system/embeddings/README.md) — Vectorisation
- [vector_store/](rag_system/vector_store/README.md) — Stockage FAISS
- [mcp_server/](rag_system/mcp_server/README.md) — API MCP
- [agents/](rag_system/agents/README.md) — Agent RAG
- [evaluation/](rag_system/evaluation/README.md) — Métriques
- [demo/](rag_system/demo/README.md) — Interfaces

## 🤝 Contribution

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## 📄 License

MIT License — voir [LICENSE](LICENSE) pour plus de détails.

---

Fait avec ❤️ par [FEDI-HASSINE](https://github.com/FEDI-HASSINE)
# 🔍 RAG System from Scratch avec MCP

Un système RAG (Retrieval-Augmented Generation) complet, construit de zéro avec architecture MCP (Model Context Protocol).

## 🎯 Objectifs

- **Zero hallucination** : Réponses basées uniquement sur les documents
- **Traçabilité complète** : Chaque étape du pipeline est visible
- **Architecture modulaire** : Composants réutilisables via MCP
- **Multi-agent ready** : Prêt pour orchestration multi-agents

## 📦 Phases du projet

| Phase | Dossier | Description |
|-------|---------|-------------|
| 1 | `data/` | Ingestion & chunking des documents |
| 2 | `embeddings/` + `vector_store/` | Vectorisation & indexation FAISS |
| 3-4 | `mcp_server/tools/` | Outils RAG (retrieve, rerank) |
| 5 | `mcp_server/` | API MCP unifiée |
| 6 | `agents/` | Agent RAG orchestrant le pipeline |
| 7 | `evaluation/` | LLM-as-a-Judge pour métriques |
| 8 | `demo/` | CLI Typer + UI Streamlit |

## 🚀 Démarrage rapide

### 1. Démarrer le serveur MCP
```bash
cd rag_system/mcp_server
uvicorn main:app --reload
```

### 2. Tester via CLI
```bash
cd rag_system/demo
python rag_cli.py ask "What is system architecture?"
```

### 3. Lancer l'interface web
```bash
cd rag_system/demo
streamlit run app.py
```

## 📁 Structure

```
rag_system/
├── data/              # Phase 1: Ingestion
│   ├── raw_docs/      # Documents sources
│   ├── loaders.py     # Chargement multi-format
│   ├── chunker.py     # Découpage intelligent
│   └── chunks.json    # Sortie
├── embeddings/        # Phase 2: Vectorisation
│   └── embedding_models.py
├── vector_store/      # Phase 2: Stockage
│   ├── faiss_store.py
│   └── index.faiss
├── mcp_server/        # Phase 5: API MCP
│   ├── main.py        # FastAPI
│   └── tools/         # Phases 3-4: Outils
├── agents/            # Phase 6: Agent RAG
│   ├── rag_agent.py
│   └── llm_service.py
├── evaluation/        # Phase 7: Métriques
│   └── eval_pipeline.py
└── demo/              # Phase 8: Interfaces
    ├── rag_cli.py
    └── app.py
```

## 🔧 Technologies

| Composant | Technologie |
|-----------|-------------|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| API | FastAPI + MCP |
| LLM | OpenAI / Ollama / Mock |
| CLI | Typer + Rich |
| UI | Streamlit |

## 📊 Pipeline RAG

```
Question → Embed → Retrieve → Rerank → Context → LLM → Réponse
              │         │          │                    │
              └─────────┴──────────┴────────────────────┘
                      Tout exposé via MCP
```

## 🛡️ Zero Hallucination

Le système garantit des réponses fiables :
- Prompts stricts imposant l'utilisation du contexte
- Citation explicite des sources
- Aveu d'ignorance si info absente

## 📖 Documentation par dossier

Chaque dossier contient un `README.md` détaillant :
- 🎯 Objectif
- 🔧 Problèmes résolus
- 📁 Fichiers
- 🚀 Utilisation

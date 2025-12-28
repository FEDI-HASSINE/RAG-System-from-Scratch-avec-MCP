# RAG System Demo — Phase 8

Cette phase propose deux interfaces pour interagir avec le RAG System :

| Mode | Fichier | Usage |
|------|---------|-------|
| CLI (Typer) | `rag_cli.py` | Dev / Jury technique |
| UI (Streamlit) | `app.py` | Démonstration visuelle |

---

## Pré-requis

1. **MCP Server** en cours d'exécution :
   ```bash
   cd rag_system/mcp_server
   uvicorn main:app --reload
   ```

2. **Dépendances Python** (déjà installées par l'agent) :
   ```bash
   pip install typer rich streamlit pandas matplotlib
   ```

3. *(Optionnel)* Clé OpenAI pour un LLM réel :
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

---

## Mode CLI

### Poser une question
```bash
cd rag_system/demo
python rag_cli.py ask "What is A2A protocol?"
```

### Options
| Flag | Description |
|------|-------------|
| `--top-k` / `-k` | Nombre de chunks affichés (défaut : 5) |
| `--trace / --no-trace` | Afficher la trace du pipeline |
| `--mcp URL` | URL du serveur MCP |

### Autres commandes
```bash
python rag_cli.py health      # État du système
python rag_cli.py stats       # Statistiques agent
python rag_cli.py export "Ma question" -o reponse.md
```

### Exemple de sortie
```
❓ Question: What is A2A protocol?

🔍 Retrieved Chunks:
┏━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Score  ┃ Source        ┃ Extrait                                           ┃
┡━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ 0.92   │ a2a_spec.md   │ A2A allows agents to publish Agent Cards...       │
│ 2 │ 0.87   │ protocols.md  │ Agent interoperability enables...                 │
└───┴────────┴───────────────┴───────────────────────────────────────────────────┘

╭──────────────────────── 🧠 Final Answer ─────────────────────────╮
│ A2A is an open protocol enabling agents to discover and         │
│ collaborate using Agent Cards.                                   │
╰──────────────────────────────────────────────────────────────────╯

📚 Sources: a2a_spec.md, protocols.md

📊 Pipeline Trace:
   ✅ retrieve: 3.5ms — found 10 chunks
   ✅ rerank: 120ms — top 5 after rerank
   ✅ build_context: 0.2ms — context=1024 chars
   ✅ llm_generate: 450ms — response=156 chars
   ⏱️  Total: 574ms
```

---

## Mode UI (Streamlit)

### Lancer l'interface
```bash
cd rag_system/demo
streamlit run app.py
```

Ouvrez ensuite `http://localhost:8501` dans votre navigateur.

### Fonctionnalités

| Zone UI | Fonction |
|---------|----------|
| **Input** | Question utilisateur |
| **Bouton** | ▶️ Run RAG |
| **Expandable panel** | Chunks récupérés avec scores |
| **Toggle** | Voir raisonnement RAG (pipeline trace) |
| **Download** | Export réponse Markdown |

### Capture d'écran (à ajouter)
Placez vos captures dans `rag_system/demo/screenshots/`.

---

## Bouton "Voir raisonnement RAG"

Affiche les étapes du pipeline :

1. **Embedding query** — Vectorisation de la question
2. **Retrieving top chunks** — Recherche dans FAISS
3. **Reranking** — Réordonnancement par Cross-Encoder
4. **Prompt injection** — Construction du prompt avec contexte
5. **LLM generation** — Génération de la réponse

> ⚠️ Pas de chain-of-thought réel — seulement la logique du pipeline.

---

## Sécurité

- Les clés API (ex. `OPENAI_API_KEY`) sont chargées via **variables d'environnement** et ne sont jamais affichées.
- Le serveur MCP est en local (`localhost:8000`) par défaut.

---

## Fichiers

| Fichier | Description |
|---------|-------------|
| `rag_cli.py` | CLI Typer avec commandes ask, health, stats, export |
| `app.py` | Interface Streamlit |
| `demo.md` | Ce document |
| `screenshots/` | Captures d'écran pour portfolio |

---

## Troubleshooting

| Problème | Solution |
|----------|----------|
| MCP Server ❌ | `cd rag_system/mcp_server && uvicorn main:app --reload` |
| LLM ⚠️ (mock) | Définir `OPENAI_API_KEY` ou utiliser Ollama |
| Aucun chunk | Vérifier que `vector_store/index.faiss` existe |

---

*Phase 8 — RAG System from Scratch avec MCP*

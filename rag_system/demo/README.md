# 🎬 Demo — Interface CLI & UI (Phase 8)

## 🎯 Objectif

Permettre à quiconque de tester le RAG System et comprendre son fonctionnement.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Accès technique uniquement | `app.py` — UI Streamlit accessible |
| Pas de visibilité pipeline | Toggle "Voir raisonnement RAG" |
| Export des réponses | Bouton Download Markdown |
| Tests rapides | `rag_cli.py` — CLI Typer |

## 📁 Fichiers

```
demo/
├── rag_cli.py      # CLI Typer
├── app.py          # Interface Streamlit
├── demo.md         # Documentation complète
└── screenshots/    # Captures pour portfolio
```

## 🚀 Pré-requis

1. **MCP Server actif** :
```bash
cd rag_system/mcp_server
uvicorn main:app --reload
```

2. *(Optionnel)* Clé OpenAI pour LLM réel :
```bash
export OPENAI_API_KEY=sk-...
```

## 💻 Mode CLI

### Poser une question
```bash
cd rag_system/demo
python rag_cli.py ask "What is system architecture?" --top-k 3
```

### Autres commandes
```bash
python rag_cli.py health          # État du système
python rag_cli.py stats           # Statistiques
python rag_cli.py export "Q?" -o response.md
```

### Exemple de sortie
```
❓ Question: system architecture

🔍 Retrieved Chunks:
┏━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Score ┃ Source    ┃ Extrait                        ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ 5.59  │ notes.txt │ Technical Notes - System...    │
└───┴───────┴───────────┴────────────────────────────────┘

🧠 Final Answer:
The system uses a distributed architecture with...

📊 Pipeline Trace:
   ✅ retrieve: 50ms
   ✅ rerank: 120ms
   ✅ llm_generate: 450ms
```

## 🌐 Mode UI (Streamlit)

```bash
cd rag_system/demo
streamlit run app.py
# Ouvrez http://localhost:8501
```

### Fonctionnalités

| Zone | Fonction |
|------|----------|
| Input | Question utilisateur |
| Bouton | ▶️ Run RAG |
| Panel | Chunks récupérés + scores |
| Toggle | Voir raisonnement RAG |
| Download | Export Markdown |

## 🔍 Raisonnement RAG

Le toggle affiche les étapes :
1. **Embedding query** — Vectorisation
2. **Retrieving chunks** — Recherche FAISS
3. **Reranking** — Cross-encoder
4. **Prompt injection** — Construction contexte
5. **LLM generation** — Réponse finale

> ⚠️ C'est la logique du pipeline, pas un chain-of-thought LLM.

## 🔒 Sécurité

- Clés API via variables d'environnement (jamais affichées)
- Serveur MCP en localhost par défaut

# 🔌 MCP Server — API Unifiée (Phase 5)

## 🎯 Objectif

Exposer tous les outils RAG via une API REST unifiée suivant le protocole MCP (Model Context Protocol).

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Outils dispersés | Endpoint unique `/mcp` avec registry |
| Découverte des outils | `/tools` liste tous les outils disponibles |
| Appels multiples inefficaces | `/mcp/batch` pour appels groupés |
| Traçabilité des requêtes | Logging centralisé avec request IDs |
| Erreurs non standardisées | Format de réponse MCP uniforme |

## 📁 Fichiers

```
mcp_server/
├── main.py              # FastAPI app + endpoints
├── tools_registry.py    # Registry des outils MCP
├── retrieval_service.py # Service de recherche
├── reranking_service.py # Service de reranking
├── tools/               # Implémentations des outils
│   ├── embed_text.py
│   ├── retrieve_chunks.py
│   └── rerank.py
├── logs/                # Logs centralisés
└── requirements.txt
```

## 🚀 Démarrage

```bash
cd rag_system/mcp_server
uvicorn main:app --reload --port 8000
```

## 📡 Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | État du serveur |
| `/status` | GET | Statistiques et uptime |
| `/tools` | GET | Liste des outils disponibles |
| `/tools/{name}` | GET | Schéma d'un outil |
| `/mcp` | POST | Appel d'un outil |
| `/mcp/batch` | POST | Appels multiples |

## 🛠️ Outils disponibles

### `embed_text`
Vectorise un texte.
```json
{
  "tool": "embed_text",
  "params": { "text": "Hello world" }
}
```

### `retrieve_chunks`
Recherche sémantique dans les documents.
```json
{
  "tool": "retrieve_chunks",
  "params": { "query": "security measures", "top_k": 5 }
}
```

### `rerank`
Réordonne les chunks par pertinence.
```json
{
  "tool": "rerank",
  "params": { "query": "...", "chunks": [...], "top_k": 3 }
}
```

## 📋 Exemple de requête

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool": "retrieve_chunks", "params": {"query": "data protection", "top_k": 3}}'
```

Réponse :
```json
{
  "success": true,
  "result": {
    "chunks": [...],
    "total_found": 3
  },
  "execution_time_ms": 45.2
}
```

## 📊 Logs

Les logs sont écrits dans `logs/` :
- `mcp.log` — Logs généraux du serveur
- `requests.log` — Détails des requêtes avec timings

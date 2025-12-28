# 🧠 Embeddings — Vectorisation (Phase 2)

## 🎯 Objectif

Convertir les chunks textuels en vecteurs numériques pour la recherche sémantique.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Choix du modèle d'embedding | `embedding_models.py` — Support multi-modèles (SentenceTransformers, OpenAI) |
| Vectorisation batch inefficace | Traitement par lots optimisé |
| Stockage des embeddings | Intégration avec FAISS via `indexing_pipeline.py` |
| Cohérence modèle/index | Métadonnées sauvegardées avec le modèle utilisé |

## 📁 Fichiers

```
embeddings/
├── embedding_models.py    # Service d'embedding unifié
└── indexing_pipeline.py   # Pipeline de création d'index
```

## 🚀 Utilisation

### Générer un embedding
```python
from embeddings.embedding_models import get_embedding_service

service = get_embedding_service("sentence-transformers")
vector = service.embed("Texte à vectoriser")
print(f"Dimension: {len(vector)}")  # 384 pour all-MiniLM-L6-v2
```

### Créer l'index complet
```python
from embeddings.indexing_pipeline import IndexingPipeline

pipeline = IndexingPipeline()
result = pipeline.run()
print(f"Vecteurs indexés: {result.total_vectors}")
```

Ou en ligne de commande :
```bash
cd rag_system
python run_indexing.py
```

## 🤖 Modèles supportés

| Modèle | Type | Dimensions | Vitesse |
|--------|------|------------|---------|
| `all-MiniLM-L6-v2` | SentenceTransformers | 384 | ⚡ Rapide |
| `all-mpnet-base-v2` | SentenceTransformers | 768 | 🎯 Précis |
| `text-embedding-ada-002` | OpenAI | 1536 | ☁️ API |

## ⚙️ Configuration

```python
from embeddings.embedding_models import EmbeddingService

service = EmbeddingService(
    model_type="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)
```

Variable d'environnement pour OpenAI :
```bash
export OPENAI_API_KEY=sk-...
```

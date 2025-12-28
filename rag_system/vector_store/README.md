# 🗄️ Vector Store — Stockage FAISS (Phase 2)

## 🎯 Objectif

Stocker et rechercher efficacement les vecteurs d'embeddings avec leurs métadonnées.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Recherche vectorielle rapide | FAISS avec index optimisé (Flat, IVF) |
| Persistance des vecteurs | Sauvegarde/chargement `index.faiss` + `metadata.json` |
| Association vecteur ↔ texte | Métadonnées stockées séparément |
| Filtrage par source/section | Support des filtres dans la recherche |

## 📁 Fichiers

```
vector_store/
├── faiss_store.py     # Wrapper FAISS avec métadonnées
├── index.faiss        # Index FAISS binaire (généré)
└── metadata.json      # Métadonnées des chunks (généré)
```

## 🚀 Utilisation

### Charger et rechercher
```python
from vector_store.faiss_store import FAISSVectorStore
from embeddings.embedding_models import get_embedding_service

# Charger l'index existant
store = FAISSVectorStore.load("vector_store/")
embedding_service = get_embedding_service("sentence-transformers")

# Rechercher
results = store.search_by_text(
    query_text="system architecture",
    embedding_service=embedding_service,
    k=5
)

for r in results:
    print(f"Score: {r.score:.2f} | {r.source}: {r.text[:100]}")
```

### Créer un nouvel index
```python
store = FAISSVectorStore(dimension=384, metric="l2")
store.add_vectors(vectors, chunks_metadata)
store.save("vector_store/")
```

## 📊 Structure des métadonnées

`metadata.json` :
```json
{
  "dimension": 384,
  "index_type": "flat",
  "metric": "l2",
  "model_name": "all-MiniLM-L6-v2",
  "total_vectors": 21,
  "chunks": [
    {
      "chunk_id": "notes_001",
      "source": "notes.txt",
      "text": "...",
      "section": "Introduction"
    }
  ]
}
```

## ⚙️ Types d'index

| Type | Usage | Performance |
|------|-------|-------------|
| `flat` | < 10K vecteurs | Exact, lent |
| `ivf` | 10K-1M vecteurs | Approx, rapide |
| `hnsw` | > 1M vecteurs | Approx, très rapide |

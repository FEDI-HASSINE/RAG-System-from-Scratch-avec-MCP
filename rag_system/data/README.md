# 📂 Data — Ingestion & Preprocessing (Phase 1)

## 🎯 Objectif

Transformer des documents bruts (TXT, MD, PDF) en chunks structurés prêts pour l'embedding.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Documents de formats variés | `loaders.py` — Loaders unifiés (TXT, MD, PDF) |
| Texte bruité (HTML, espaces) | `cleaner.py` — Nettoyage et normalisation |
| Perte de structure (titres, sections) | `structure_detector.py` — Détection automatique des sections |
| Chunks trop grands/petits | `chunker.py` — Découpage intelligent avec overlap |
| Pipeline manuelle répétitive | `ingestion_pipeline.py` — Orchestration automatique |

## 📁 Fichiers

```
data/
├── raw_docs/              # Documents sources (TXT, MD, PDF)
├── loaders.py             # Chargement multi-format
├── cleaner.py             # Nettoyage du texte
├── structure_detector.py  # Détection de structure
├── chunker.py             # Découpage en chunks
├── ingestion_pipeline.py  # Pipeline complète
└── chunks.json            # Sortie: chunks prêts pour embedding
```

## 🚀 Utilisation

```python
from data.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline()
result = pipeline.run()
print(f"Chunks créés: {result.total_chunks}")
```

Ou en ligne de commande :
```bash
cd rag_system
python run_ingestion.py
```

## 📊 Sortie

`chunks.json` contient :
```json
[
  {
    "chunk_id": "notes_001",
    "text": "Contenu du chunk...",
    "source": "notes.txt",
    "section": "Introduction",
    "tokens": 128
  }
]
```

## ⚙️ Configuration

Dans `ingestion_pipeline.py` :
- `chunk_size`: Taille cible des chunks (défaut: 512 tokens)
- `chunk_overlap`: Chevauchement entre chunks (défaut: 50 tokens)
- `min_chunk_size`: Taille minimale (défaut: 100 tokens)

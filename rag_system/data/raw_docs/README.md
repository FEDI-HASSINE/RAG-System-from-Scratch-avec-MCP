# 📄 Raw Documents — Sources brutes

## 🎯 Objectif

Stocker les documents sources à ingérer dans le système RAG.

## 📁 Formats supportés

| Format | Extension | Loader |
|--------|-----------|--------|
| Texte brut | `.txt` | `TextLoader` |
| Markdown | `.md` | `MarkdownLoader` |
| PDF | `.pdf` | `PDFLoader` |

## 📋 Documents actuels

| Fichier | Description |
|---------|-------------|
| `notes.txt` | Notes techniques sur l'architecture système |
| `privacy_policy.md` | Politique de confidentialité |
| `finance_report.txt` | Rapport financier |

## ➕ Ajouter un document

1. Placez votre fichier dans ce dossier
2. Relancez l'ingestion :
   ```bash
   cd rag_system
   python run_ingestion.py
   ```
3. Réindexez :
   ```bash
   python run_indexing.py
   ```

## ⚠️ Bonnes pratiques

- **Encodage** : UTF-8 recommandé
- **Taille** : Pas de limite, le chunker découpe automatiquement
- **Nommage** : Évitez les caractères spéciaux
- **Structure** : Utilisez des titres/sections pour une meilleure détection

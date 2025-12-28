# 📊 Evaluation — LLM-as-a-Judge (Phase 7)

## 🎯 Objectif

Mesurer objectivement la qualité du RAG Agent avec un "juge" LLM.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Évaluation subjective | LLM-as-a-Judge avec scores 0-1 |
| Métriques non standardisées | Groundedness, Relevance, Faithfulness |
| Suivi dans le temps | `history.csv` pour baseline |
| Visualisation | Dashboard PNG automatique |

## 📁 Fichiers

```
evaluation/
├── eval_pipeline.py    # Pipeline d'évaluation
├── eval_dataset.json   # Questions + réponses attendues
├── eval_results.csv    # Résultats par question (généré)
├── dashboard.png       # Graphique des scores (généré)
└── history.csv         # Historique des runs (généré)
```

## 🚀 Pré-requis

- MCP server démarré: `cd rag_system/mcp_server && uvicorn main:app --reload`
- Optionnel: `OPENAI_API_KEY` pour un Judge plus fiable (sinon Mock)
- Dépendances: `pandas`, `matplotlib`

## 🚀 Lancer l'évaluation

```bash
cd rag_system
python -c "from evaluation.eval_pipeline import run_evaluation; run_evaluation()"
```

Ou avec Python :
```python
from evaluation.eval_pipeline import run_evaluation

result = run_evaluation(
    dataset_path="evaluation/eval_dataset.json",
    output_dir="evaluation/"
)

print(f"Groundedness: {result['aggregate']['groundedness']:.2f}")
print(f"Relevance: {result['aggregate']['relevance']:.2f}")
print(f"Faithfulness: {result['aggregate']['faithfulness']:.2f}")
```

## 📋 Format du dataset

`eval_dataset.json` :
```json
[
  {
    "question": "What is Agent2Agent protocol?",
    "expected_answer": "A2A is an open protocol enabling agents to discover and collaborate."
  }
]
```

## 📊 Métriques

| Métrique | Description | Score idéal |
|----------|-------------|-------------|
| **Groundedness** | Réponse basée sur le contexte récupéré | 1.0 |
| **Relevance** | Répond à la question posée | 1.0 |
| **Faithfulness** | Pas d'hallucination | 1.0 |

## 🤖 Judge LLM

Le prompt du juge :
```
Evaluate the assistant answer.

Question: {question}
Expected: {expected}
Assistant: {answer}

Give scores between 0 and 1:
- groundedness
- relevance  
- faithfulness

Return JSON only.
```

## 📈 Sorties

- `eval_results.csv` : scores par question
- `dashboard.png` : moyennes (0–1) en graphique
- `history.csv` : historique des agrégats pour baseline

## ⚙️ Personnaliser

- Modifiez le dataset: `eval_dataset.json`
- Changez le Judge: éditez `judge_provider` / `judge_model` dans `run_evaluation()`

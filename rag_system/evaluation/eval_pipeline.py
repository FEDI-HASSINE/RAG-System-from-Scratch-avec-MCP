"""
Evaluation pipeline for RAG Agent (Phase 7)

Outputs:
- eval_results.csv: per-question scores
- dashboard.png: average scores bar chart
- history.csv: timestamped aggregate metrics for baseline tracking
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Ensure rag_system is importable when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_SYSTEM_DIR = os.path.dirname(CURRENT_DIR)
if RAG_SYSTEM_DIR not in sys.path:
    sys.path.insert(0, RAG_SYSTEM_DIR)

from agents.rag_agent import RAGAgent
from agents.llm_service import call_llm, LLMConfig, LLMProvider, get_llm_service


def _ensure_packages():
    """Lazy check for required packages and guide installation if missing."""
    try:
        import pandas  # noqa: F401
        import matplotlib  # noqa: F401
    except Exception:
        print("[!] Dépendances manquantes: installez pandas et matplotlib")
        print("    pip install pandas matplotlib")


def judge_answer(question: str, expected: str, answer: str,
                 provider: LLMProvider = LLMProvider.OPENAI,
                 model: str = "gpt-4o-mini") -> Dict[str, float]:
    """
    LLM-as-a-Judge: returns scores in [0,1] for groundedness, relevance, faithfulness.
    Falls back to mock provider if the target provider is unavailable.
    """
    prompt = f"""
    Evaluate the assistant answer.

    Question: {question}
    Expected: {expected}
    Assistant: {answer}

    Give scores between 0 and 1:
    - groundedness
    - relevance
    - faithfulness

    Return JSON only.
    """

    # Configure judge LLM
    cfg = LLMConfig(provider=provider, model=model, temperature=0.0, max_tokens=256)
    # Initialize specific service instance to avoid global state collisions
    get_llm_service(cfg)

    raw = call_llm(prompt)

    def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(s)
        except Exception:
            # Try to extract JSON blob
            import re
            match = re.search(r"\{[\s\S]*\}", s)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
            return None

    data = _safe_parse_json(raw) or {}
    # Normalize and clamp scores
    def _clamp(v: Any) -> float:
        try:
            x = float(v)
        except Exception:
            x = 0.0
        return max(0.0, min(1.0, x))

    return {
        "groundedness": _clamp(data.get("groundedness")),
        "relevance": _clamp(data.get("relevance")),
        "faithfulness": _clamp(data.get("faithfulness")),
    }


def run_evaluation(dataset_path: Optional[str] = None,
                   output_dir: Optional[str] = None,
                   judge_provider: LLMProvider = LLMProvider.OPENAI,
                   judge_model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Run evaluation loop and persist results & dashboard.

    Returns aggregate metrics.
    """
    _ensure_packages()
    import pandas as pd
    import matplotlib.pyplot as plt

    dataset_path = dataset_path or os.path.join(CURRENT_DIR, "eval_dataset.json")
    output_dir = output_dir or CURRENT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset non trouvé: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    # Initialize agent
    agent = RAGAgent()
    health = agent.health_check()
    if not health.get("mcp_server", False):
        print("[!] MCP Server indisponible — l'évaluation utilisera les fallback et peut être moins pertinente.")

    results: List[Dict[str, Any]] = []
    start = time.time()

    for row in dataset:
        q = row.get("question", "").strip()
        expected = row.get("expected_answer", "").strip()
        if not q:
            continue

        t0 = time.time()
        resp = agent.answer(q, include_trace=False)
        ans = resp.answer

        scores = judge_answer(q, expected, ans, provider=judge_provider, model=judge_model)

        results.append({
            "question": q,
            "groundedness": scores["groundedness"],
            "relevance": scores["relevance"],
            "faithfulness": scores["faithfulness"],
            "uses_sources": 1 if (resp.sources or resp.chunks_used > 0) else 0,
            "answer_len": len(ans or ""),
            "duration_ms": (time.time() - t0) * 1000.0,
        })

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "eval_results.csv")
    df.to_csv(csv_path, index=False)

    # Aggregate metrics
    agg = {
        "groundedness": float(df["groundedness"].mean()) if not df.empty else 0.0,
        "relevance": float(df["relevance"].mean()) if not df.empty else 0.0,
        "faithfulness": float(df["faithfulness"].mean()) if not df.empty else 0.0,
        "uses_sources_rate": float(df["uses_sources"].mean()) if not df.empty else 0.0,
        "count": int(len(df)),
        "total_ms": (time.time() - start) * 1000.0,
    }

    # Dashboard plot
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["groundedness", "relevance", "faithfulness"]
    values = [agg[m] for m in metrics]
    ax.bar(metrics, values, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_ylim(0, 1)
    ax.set_title("Scores moyens (0-1)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    dash_path = os.path.join(output_dir, "dashboard.png")
    fig.savefig(dash_path)
    plt.close(fig)

    # Baseline history
    hist_path = os.path.join(output_dir, "history.csv")
    hist_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **agg,
    }
    if os.path.exists(hist_path):
        hist_df = pd.read_csv(hist_path)
        hist_df = pd.concat([hist_df, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        hist_df = pd.DataFrame([hist_row])
    hist_df.to_csv(hist_path, index=False)

    print(f"✅ Évaluation terminée. Résultats: {csv_path}, dashboard: {dash_path}")
    return {"csv": csv_path, "dashboard": dash_path, "aggregate": agg}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()

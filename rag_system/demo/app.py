#!/usr/bin/env python3
"""
RAG System — Interface Streamlit

Lancez avec:
    streamlit run app.py
"""

import os
import sys
import time

# Ensure rag_system is importable
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_SYSTEM_DIR = os.path.dirname(DEMO_DIR)
if RAG_SYSTEM_DIR not in sys.path:
    sys.path.insert(0, RAG_SYSTEM_DIR)

import streamlit as st
import pandas as pd

from agents.rag_agent import RAGAgent, RAGConfig, RetrievalStrategy

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="RAG System Demo",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def get_agent():
    """Charge l'agent RAG (cache pour performance)."""
    config = RAGConfig(
        mcp_base_url="http://localhost:8000",
        retrieval_strategy=RetrievalStrategy.RERANK,
        initial_top_k=10,
        final_top_k=5,
        include_trace=True,
    )
    return RAGAgent(config)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/search--v1.png", width=64)
    st.title("RAG System")
    st.markdown("---")

    # Health check
    agent = get_agent()
    health = agent.health_check()

    st.subheader("🩺 État du système")
    col1, col2 = st.columns(2)
    with col1:
        if health.get("mcp_server"):
            st.success("MCP ✅")
        else:
            st.error("MCP ❌")
    with col2:
        if health.get("llm_service"):
            st.success("LLM ✅")
        else:
            st.warning("LLM ⚠️")

    st.markdown("---")

    # Options
    st.subheader("⚙️ Options")
    top_k = st.slider("Chunks à afficher", 1, 10, 5)
    show_trace = st.checkbox("Voir raisonnement RAG", value=True)

    st.markdown("---")
    st.caption("Phase 8 — Démo RAG System")


# ============================================================
# Main Content
# ============================================================

st.title("🔍 RAG System Demo")
st.markdown(
    """
    Posez une question et observez comment le système RAG récupère et exploite vos documents.
    """
)

# Input
question = st.text_input(
    "💬 Votre question",
    placeholder="Ex: What is Agent2Agent protocol?",
    key="question_input",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    run_btn = st.button("▶️ Run RAG", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear"):
        st.session_state.pop("last_response", None)
        st.rerun()

# ============================================================
# RAG Execution
# ============================================================

if run_btn and question.strip():
    if not health.get("mcp_server"):
        st.error("❌ MCP Server indisponible. Démarrez-le avec:")
        st.code("cd rag_system/mcp_server && uvicorn main:app --reload", language="bash")
    else:
        with st.spinner("🧠 RAG en cours..."):
            start = time.time()
            response = agent.answer(question.strip(), include_trace=show_trace)
            elapsed = (time.time() - start) * 1000

        st.session_state["last_response"] = response
        st.session_state["last_question"] = question.strip()
        st.session_state["last_elapsed"] = elapsed
        st.session_state["last_top_k"] = top_k


# ============================================================
# Display Results
# ============================================================

if "last_response" in st.session_state:
    response = st.session_state["last_response"]
    question_used = st.session_state.get("last_question", "")
    elapsed = st.session_state.get("last_elapsed", 0)
    top_k_used = st.session_state.get("last_top_k", 5)

    st.markdown("---")

    # Chunks récupérés
    with st.expander("🔍 Chunks récupérés", expanded=True):
        chunks = agent.retrieve_and_rerank(question_used, initial_k=10, final_k=top_k_used)
        if chunks:
            data = []
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get("rerank_score") or chunk.get("score", 0)
                source = chunk.get("source", "?")
                text = (chunk.get("text") or chunk.get("content", ""))[:200]
                data.append({"#": i, "Score": f"{score:.2f}", "Source": source, "Extrait": text})
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun chunk trouvé.")

    # Réponse finale
    st.subheader("🧠 Réponse")
    st.markdown(response.answer)

    # Sources
    if response.sources:
        st.caption(f"📚 Sources: {', '.join(response.sources)}")

    # Trace du pipeline
    if show_trace and response.trace:
        with st.expander("📊 Raisonnement RAG (Pipeline Trace)", expanded=False):
            st.markdown(
                """
                **Étapes du pipeline RAG :**
                1. **Embedding query** — Vectorisation de la question
                2. **Retrieving top chunks** — Recherche dans FAISS
                3. **Reranking** — Réordonnancement par Cross-Encoder
                4. **Prompt injection** — Construction du prompt avec contexte
                5. **LLM generation** — Génération de la réponse
                """
            )
            st.markdown("---")
            trace_data = []
            for step in response.trace.steps:
                status = "✅" if step.success else "❌"
                trace_data.append({
                    "Étape": f"{status} {step.name}",
                    "Durée (ms)": f"{step.duration_ms:.1f}",
                    "Détails": step.output_summary,
                })
            st.table(pd.DataFrame(trace_data))
            st.metric("⏱️ Temps total", f"{response.trace.total_duration_ms:.0f} ms")

    # Export
    st.markdown("---")
    md_export = f"""# RAG Response

## Question
{question_used}

## Answer
{response.answer}

## Sources
{', '.join(response.sources) if response.sources else 'Aucune'}

## Trace
"""
    if response.trace:
        for step in response.trace.steps:
            status = "✅" if step.success else "❌"
            md_export += f"- {status} **{step.name}**: {step.duration_ms:.1f}ms — {step.output_summary}\n"
        md_export += f"\n**Total:** {response.trace.total_duration_ms:.1f}ms\n"

    st.download_button(
        label="📥 Export Markdown",
        data=md_export,
        file_name="rag_response.md",
        mime="text/markdown",
    )

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("RAG System from Scratch avec MCP — Phase 8 Demo")

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
from agents.llm_service import LLMProvider # Added missing import
from data.ingestion_pipeline import DataIngestionPipeline
from embeddings.indexing_pipeline import IndexingPipeline
import shutil

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
    
    # Configuration par défaut (OpenRouter / Llama 3.3)
    # Assurez-vous que OPENAI_API_KEY est défini dans le terminal avant de lancer
    config = RAGConfig(
        mcp_base_url="http://localhost:8000",
        retrieval_strategy=RetrievalStrategy.RERANK,
        initial_top_k=10,
        final_top_k=5,
        include_trace=True,
        llm_provider=LLMProvider.OPENAI,
        llm_model="meta-llama/llama-3.3-70b-instruct:free",
        llm_base_url="https://openrouter.ai/api/v1",
        temperature=0.1
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
    if not os.getenv("OPENAI_API_KEY"):
         st.error("⚠️ OPENAI_API_KEY manquante!")
         st.info("export OPENAI_API_KEY='...' avant de lancer")
            
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
    
    # Upload Documents
    st.subheader("📂 Ajouter des documents")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT/MD", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Ingérer & Indexer"):
            with st.status("Traitement en cours...", expanded=True) as status:
                try:
                    # 1. Save files
                    st.write("💾 Sauvegarde des fichiers...")
                    raw_docs_path = os.path.join(RAG_SYSTEM_DIR, "data", "raw_docs")
                    os.makedirs(raw_docs_path, exist_ok=True)
                    
                    saved_files = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(raw_docs_path, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(uploaded_file.name)
                    st.write(f"✅ {len(saved_files)} fichiers sauvegardés.")
                    
                    # 2. Ingestion
                    st.write("⚙️ Exécution Pipeline Ingestion (Cleaning + Chunking)...")
                    from data.ingestion_pipeline import IngestionConfig
                    from embeddings.indexing_pipeline import IndexingConfig
                    
                    ingestion_config = IngestionConfig(
                        input_dir=raw_docs_path,
                        output_file=os.path.join(RAG_SYSTEM_DIR, "data", "chunks.json")
                    )
                    ingestion = DataIngestionPipeline(ingestion_config)
                    ingestion_result = ingestion.run() 
                    st.write(f"✅ Ingestion terminée: {ingestion_result.total_chunks} chunks générés.")
                    
                    # 3. Indexing
                    st.write("🧠 Exécution Pipeline Indexation (Embedding + FAISS)...")
                    indexing_config = IndexingConfig(
                        chunks_file=os.path.join(RAG_SYSTEM_DIR, "data", "chunks.json"),
                        vector_store_dir=os.path.join(RAG_SYSTEM_DIR, "vector_store")
                    )
                    indexing = IndexingPipeline(indexing_config)
                    indexing_result = indexing.run()
                    st.write(f"✅ Indexation terminée: {indexing_result.chunks_indexed} vecteurs créés.")
                    
                    # 4. Reload MCP Index
                    st.write("🔄 Rechargement de la base vectors dans le serveur MCP...")
                    if agent.reload_knowledge_base():
                        st.success("Base de connaissance rechargée avec succès !")
                    else:
                        st.warning("⚠️ Impossible de recharger automatiquement le serveur. Veuillez redémarrer manuellement.")

                    status.update(label="Processus terminé !", state="complete", expanded=False)
                    st.success("La base de connaissance a été mise à jour ! Vous pouvez poser vos questions sur les nouveaux documents.")
                    
                except Exception as e:
                    st.error(f"Erreur durant le processus: {e}")
                    status.update(label="Erreur", state="error")

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

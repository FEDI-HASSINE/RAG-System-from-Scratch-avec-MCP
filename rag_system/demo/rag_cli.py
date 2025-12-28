#!/usr/bin/env python3
"""
RAG CLI — Démo ligne de commande pour le RAG System

Usage:
    python rag_cli.py ask "What is A2A protocol?"
    python rag_cli.py health
    python rag_cli.py stats
"""

import os
import sys

# Ensure rag_system is importable
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_SYSTEM_DIR = os.path.dirname(DEMO_DIR)
if RAG_SYSTEM_DIR not in sys.path:
    sys.path.insert(0, RAG_SYSTEM_DIR)

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from typing import Optional

from agents.rag_agent import RAGAgent, RAGConfig, RetrievalStrategy

app = typer.Typer(help="RAG System CLI — Interrogez vos documents via MCP")
console = Console()


def _get_agent(mcp_url: str = "http://localhost:8000") -> RAGAgent:
    config = RAGConfig(
        mcp_base_url=mcp_url,
        retrieval_strategy=RetrievalStrategy.RERANK,
        initial_top_k=10,
        final_top_k=5,
        include_trace=True,
    )
    return RAGAgent(config)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question à poser au RAG"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Nombre de chunks à afficher"),
    show_trace: bool = typer.Option(True, "--trace/--no-trace", help="Afficher la trace du pipeline"),
    mcp_url: str = typer.Option("http://localhost:8000", "--mcp", help="URL du serveur MCP"),
):
    """Pose une question au RAG System et affiche la réponse."""
    agent = _get_agent(mcp_url)

    # Health check
    health = agent.health_check()
    if not health.get("mcp_server"):
        console.print("[bold red]⚠ MCP Server indisponible.[/bold red] Démarrez-le avec:")
        console.print("  cd rag_system/mcp_server && uvicorn main:app --reload")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]❓ Question:[/bold cyan] {question}\n")

    with console.status("[bold green]RAG en cours...[/bold green]"):
        response = agent.answer(question, include_trace=show_trace)

    # --- Chunks récupérés ---
    if response.trace:
        retrieve_step = next((s for s in response.trace.steps if s.name == "retrieve"), None)
        rerank_step = next((s for s in response.trace.steps if s.name == "rerank"), None)

        # Afficher les chunks utilisés (via retrieve_and_rerank pour avoir les scores)
        chunks = agent.retrieve_and_rerank(question, initial_k=10, final_k=top_k)
        if chunks:
            console.print("[bold yellow]🔍 Retrieved Chunks:[/bold yellow]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim", width=3)
            table.add_column("Score", width=8)
            table.add_column("Source", width=15)
            table.add_column("Extrait", overflow="fold")

            for i, chunk in enumerate(chunks[:top_k], 1):
                score = chunk.get("rerank_score") or chunk.get("score", 0)
                source = chunk.get("source", "?")
                text = (chunk.get("text") or chunk.get("content", ""))[:120] + "..."
                table.add_row(str(i), f"{score:.2f}", source, text)

            console.print(table)
            console.print()

    # --- Réponse finale ---
    console.print(Panel(
        Markdown(response.answer),
        title="[bold green]🧠 Final Answer[/bold green]",
        border_style="green",
    ))

    # --- Sources ---
    if response.sources:
        console.print(f"\n[bold blue]📚 Sources:[/bold blue] {', '.join(response.sources)}")

    # --- Trace du pipeline ---
    if show_trace and response.trace:
        console.print("\n[bold magenta]📊 Pipeline Trace:[/bold magenta]")
        for step in response.trace.steps:
            status = "✅" if step.success else "❌"
            console.print(f"   {status} [cyan]{step.name}[/cyan]: {step.duration_ms:.1f}ms — {step.output_summary}")
        console.print(f"   ⏱️  Total: {response.trace.total_duration_ms:.1f}ms")


@app.command()
def health(
    mcp_url: str = typer.Option("http://localhost:8000", "--mcp", help="URL du serveur MCP"),
):
    """Vérifie l'état du RAG System (MCP + LLM)."""
    agent = _get_agent(mcp_url)
    h = agent.health_check()

    console.print("\n[bold]🩺 Health Check[/bold]")
    console.print(f"   MCP Server: {'✅' if h.get('mcp_server') else '❌'}")
    console.print(f"   LLM Service: {'✅' if h.get('llm_service') else '⚠️ (mock)'}")


@app.command()
def stats(
    mcp_url: str = typer.Option("http://localhost:8000", "--mcp", help="URL du serveur MCP"),
):
    """Affiche les statistiques de l'agent RAG."""
    agent = _get_agent(mcp_url)
    s = agent.get_stats()

    console.print("\n[bold]📈 Agent Stats[/bold]")
    console.print(f"   MCP calls: {s['mcp_stats'].get('total_calls', 0)}")
    console.print(f"   Memory size: {s['memory_size']}")
    console.print(f"   Strategy: {s['config']['retrieval_strategy']}")
    console.print(f"   LLM model: {s['config']['llm_model']}")


@app.command()
def export(
    question: str = typer.Argument(..., help="Question à poser"),
    output: str = typer.Option("response.md", "--output", "-o", help="Fichier de sortie"),
    mcp_url: str = typer.Option("http://localhost:8000", "--mcp", help="URL du serveur MCP"),
):
    """Exporte la réponse RAG en Markdown."""
    agent = _get_agent(mcp_url)

    health = agent.health_check()
    if not health.get("mcp_server"):
        console.print("[bold red]⚠ MCP Server indisponible.[/bold red]")
        raise typer.Exit(1)

    response = agent.answer(question, include_trace=True)

    md_content = f"""# RAG Response

## Question
{question}

## Answer
{response.answer}

## Sources
{', '.join(response.sources) if response.sources else 'Aucune source'}

## Pipeline Trace
"""
    if response.trace:
        for step in response.trace.steps:
            status = "✅" if step.success else "❌"
            md_content += f"- {status} **{step.name}**: {step.duration_ms:.1f}ms — {step.output_summary}\n"
        md_content += f"\n**Total:** {response.trace.total_duration_ms:.1f}ms\n"

    with open(output, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print(f"[green]✅ Réponse exportée vers {output}[/green]")


if __name__ == "__main__":
    app()

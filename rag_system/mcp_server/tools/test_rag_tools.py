"""
Script de Test - Phases 3 & 4 : Outils RAG (Embed, Retrieve, Rerank)

Ce script teste les trois outils fondamentaux qui permettent au système de finding
la bonne information. Il simule le pipeline sans passer par le serveur API.

Usage:
    python test_rag_tools.py
"""

import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ajouter la racine du projet au path
# Si on est dans rag_system/mcp_server/tools, la racine du projet qui contient 'rag_system' est 3 niveaux plus haut
# Mais ici on veut importer depuis le dossier PARENT de rag_system (pour faire 'from rag_system.embeddings...') 
# OU BIEN depuis rag_system lui-même si les imports sont relatifs.

# Structure:
# /workspaces/Projet/rag_system/mcp_server/tools/test.py

current_file = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_file)          # .../tools
server_dir = os.path.dirname(tools_dir)            # .../mcp_server
rag_system_dir = os.path.dirname(server_dir)       # .../rag_system (où se trouve embeddings/, vector_store/)

# Ajouter rag_system au path pour pouvoir importer les modules
sys.path.insert(0, rag_system_dir)

# Import des composants réels
# Note: On doit adapter les imports car on a ajouté rag_system_dir au path
# Donc on importe directement 'embeddings.xxx' et non 'rag_system.embeddings.xxx'
try:
    from embeddings.embedding_models import SentenceTransformerEmbedding
    from vector_store.faiss_store import FAISSVectorStore
except ImportError:
    # Fallback: si on lance depuis la racine, le path est différent
    root_dir = os.path.dirname(rag_system_dir)
    sys.path.insert(0, root_dir)
    from rag_system.embeddings.embedding_models import SentenceTransformerEmbedding
    from rag_system.vector_store.faiss_store import FAISSVectorStore

from sentence_transformers import CrossEncoder

console = Console()

def main():
    console.print("\n[bold magenta]🚀 Test des Outils RAG (Phases 3-4)[/bold magenta]\n")

    # 1. Définir une question test
    query = "What is Message Queue and what is his role?"
    console.print(f"[bold]❓ Question :[/bold] [yellow]'{query}'[/yellow]\n")

    # --- PHASE 3 : Embed & Retrieve ---

    # A. Embed (Transformer en vecteur)
    try:
        console.print("[dim]1. Chargement du modèle d'embedding (all-MiniLM-L6-v2)...[/dim]")
        embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
        query_vector = embedder.embed(query)
        console.print(f"   ✅ Vecteur généré (dimension {len(query_vector)})")
    except Exception as e:
        console.print(f"[red]❌ Erreur Embedding : {e}[/red]")
        return

    # B. Retrieve (Chercher dans FAISS)
    try:
        # Chemin vers vector_store (relatif à rag_system_dir)
        index_path = os.path.join(rag_system_dir, "vector_store")
        console.print(f"[dim]2. Chargement de l'index FAISS depuis {index_path}...[/dim]")
        
        if not os.path.exists(os.path.join(index_path, "index.faiss")):
            console.print(f"[red]❌ Index introuvable dans {index_path}. Lancez d'abord la Phase 2 ![/red]")
            return

        # Correction : load() est une factory method qui retourne une nouvelle instance
        store = FAISSVectorStore.load(index_path)
        console.print(f"   ✅ Index chargé ({store.index.ntotal} vecteurs)")
        
        # Récupérer plus de chunks que nécessaire pour le reranking (ex: top 10)
        initial_results = store.search(query_vector, k=10)
        console.print(f"   ✅ {len(initial_results)} chunks trouvés par similarité (FAISS)")
    except Exception as e:
        console.print(f"[red]❌ Erreur Retrieval : {e}[/red]")
        return

    if not initial_results:
        console.print("[yellow]⚠️ Aucun résultat trouvé. Fin du test.[/yellow]")
        return

    # Afficher les résultats bruts (avant reranking)
    table = Table(title="🔍 Résultats FAISS (Recherche Vectorielle)")
    table.add_column("#", style="cyan")
    table.add_column("Score (Dist)", style="dim")
    table.add_column("Texte (extrait)", style="white")
    
    for i, res in enumerate(initial_results[:3], 1): # Montrer les 3 premiers
        text_preview = res.text[:80].replace('\n', ' ') + "..."
        table.add_row(str(i), f"{res.score:.4f}", text_preview)
    console.print(table)


    # --- PHASE 4 : Rerank ---

    # C. Rerank (Réordonner par pertinence)
    try:
        console.print("\n[dim]3. Reranking avec Cross-Encoder (ms-marco-MiniLM-L-6-v2)...[/dim]")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Préparer les paires [Question, Chunk]
        pairs = [(query, res.text) for res in initial_results]
        
        # Prédire les scores de pertinence
        scores = reranker.predict(pairs)
        
        # Associer scores aux résultats
        reranked_results = []
        for i, res in enumerate(initial_results):
            res_copy = res.__dict__.copy()
            res_copy['rerank_score'] = float(scores[i])
            reranked_results.append(res_copy)
            
        # Trier par score de reranking décroissant (plus haut = plus pertinent)
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        console.print(f"   ✅ {len(reranked_results)} chunks ré-ordonnés\n")

    except Exception as e:
        console.print(f"[red]❌ Erreur Reranking : {e}[/red]")
        return

    # Afficher les résultats finaux
    final_table = Table(title="🏆 Résultats Finaux (Après Reranking)")
    final_table.add_column("Rang", style="bold green")
    final_table.add_column("Score", style="magenta")
    final_table.add_column("Texte (extrait)", style="white")
    final_table.add_column("Source", style="blue")

    for i, res in enumerate(reranked_results[:5], 1): # Top 5 final
        text_preview = res['text'][:100].replace('\n', ' ') + "..."
        # Correction: source est un attribut direct, pas dans metadata
        source = res.get('source', 'Unknown')
        final_table.add_row(f"#{i}", f"{res.get('rerank_score', 0):.4f}", text_preview, source)

    console.print(final_table)
    
    # Conclusion
    best_chunk = reranked_results[0]
    console.print(Panel(
        f"[bold green]Meilleure réponse trouvée :[/bold green]\n\n"
        f"{best_chunk.get('text', best_chunk.text if hasattr(best_chunk, 'text') else '')}\n\n"
        f"[dim]Source: {best_chunk.get('source', 'Unknown')}[/dim]",
        title="✨ Résultat Final",
        border_style="green"
    ))

if __name__ == "__main__":
    main()

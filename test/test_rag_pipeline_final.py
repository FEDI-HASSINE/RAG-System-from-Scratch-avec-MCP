import os
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.ERROR) # On veut voir surtout les prints

# Ajouter la racine du projet (un niveau au-dessus)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from rag_system.agents.rag_agent import RAGAgent, RAGConfig, LLMProvider

def test_full_rag_pipeline():
    print("🚀 Démarrage du test RAG Complet (Agent + MCP + OpenRouter)...\n")
    
    # 1. Configuration de l'Agent avec OpenRouter
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Erreur: OPENAI_API_KEY manquante.")
        return

    config = RAGConfig(
        mcp_base_url="http://localhost:8000",
        llm_provider=LLMProvider.OPENAI,
        llm_model="meta-llama/llama-3.3-70b-instruct:free",
        llm_base_url="https://openrouter.ai/api/v1",
        temperature=0.1
    )
    
    agent = RAGAgent(config)
    
    # 2. Poser une question sur vos documents
    question = "What id Message Queue and what is his role?"
    print(f"👤 Question Utilisateur : '{question}'")
    print("🤖 L'Agent réfléchit (Récupération -> Reranking -> Génération)...")
    
    try:
        # Exécution du pipeline
        response = agent.answer(question)
        
        # 3. Affichage du résultat
        print("\n" + "="*60)
        print("📝 RÉPONSE GÉNÉRÉE :")
        print("="*60)
        print(response.answer)
        print("="*60)
        
        print("\n📚 Sources utilisées :")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. {source}")
            
        print(f"\n✅ Succès ! (Chunks utilisés : {response.chunks_used})")
        
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        # Hint: Si c'est une erreur de connexion MCP, dire de vérifier Uvicorn

if __name__ == "__main__":
    test_full_rag_pipeline()
import os
import sys

# Ajouter la racine du projet au path pour résoudre 'rag_system'
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../rag_system
project_root = os.path.dirname(current_dir) # .../RAG-System-from-Scratch-avec-MCP
sys.path.insert(0, project_root)

from rag_system.agents.llm_service import LLMConfig, LLMProvider, OpenAIProvider

def test_openrouter():
    print("🚀 Test de connexion OpenRouter (Llama 3.3)...")
    
    # 1. Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Erreur: La variable d'environnement OPENAI_API_KEY n'est pas définie.")
        print("export OPENAI_API_KEY='sk-or-...'")
        return

    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="meta-llama/llama-3.3-70b-instruct:free", # Modèle gratuit choisi
        temperature=0.7
    )
    
    # 2. Initialisation du Provider
    try:
        llm = OpenAIProvider(config)
        print("✅ Provider initialisé.")
    except Exception as e:
        print(f"❌ Erreur init: {e}")
        return

    # 3. Test de Génération
    prompt = "Explique le concept de RAG (Retrieval Augmented Generation) en une phrase simple."
    print(f"\n📝 Prompt: '{prompt}'")
    print("⏳ Génération en cours...")
    
    try:
        response = llm.generate(prompt)
        print(f"\n🤖 Réponse Llama 3.3 :\n{response.content}")
        print(f"\n✅ Terminé (Tokens: {response.tokens_used})")
    except Exception as e:
        print(f"❌ Erreur lors de la génération : {e}")

if __name__ == "__main__":
    test_openrouter()
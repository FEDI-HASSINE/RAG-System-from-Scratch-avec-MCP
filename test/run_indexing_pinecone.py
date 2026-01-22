#!/usr/bin/env python3
"""
Indexer les documents dans Pinecone.

Usage:
    # 1. Définir les variables d'environnement
    export PINECONE_API_KEY="votre-clé-pinecone"
    export PINECONE_INDEX="rag-index"          # optionnel, défaut: rag-index
    export PINECONE_NAMESPACE="default"         # optionnel, défaut: default
    export PINECONE_CLOUD="aws"                 # optionnel, défaut: aws
    export PINECONE_REGION="us-east-1"          # optionnel, défaut: us-east-1
    
    # 2. Lancer l'indexation
    python run_indexing_pinecone.py
    
    # 3. Vérifier l'index
    python run_indexing_pinecone.py --check-only
"""

import sys
import os
import json
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_pinecone_index():
    """Vérifie l'état de l'index Pinecone."""
    try:
        from pinecone import Pinecone
    except ImportError:
        print("❌ Pinecone n'est pas installé. Installez avec: pip install pinecone")
        return False
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY non définie!")
        print("   export PINECONE_API_KEY='votre-clé'")
        return False
    
    index_name = os.getenv("PINECONE_INDEX", "rag-index")
    
    print("\n" + "=" * 60)
    print("🔍 VÉRIFICATION DE L'INDEX PINECONE")
    print("=" * 60)
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Lister les index existants
        indexes = list(pc.list_indexes())
        print(f"\n📋 Index existants: {len(indexes)}")
        for idx in indexes:
            print(f"   - {idx.name} (dimension: {idx.dimension}, metric: {idx.metric})")
        
        # Vérifier si notre index existe
        index_names = [idx.name for idx in indexes]
        if index_name in index_names:
            print(f"\n✅ Index '{index_name}' existe!")
            
            # Stats de l'index
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"\n📊 Statistiques de l'index:")
            print(f"   - Vecteurs total: {stats.total_vector_count}")
            print(f"   - Dimension: {stats.dimension}")
            print(f"   - Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else ['(vide)']}")
            
            if stats.namespaces:
                for ns, ns_stats in stats.namespaces.items():
                    print(f"   - Namespace '{ns}': {ns_stats.vector_count} vecteurs")
            
            return True
        else:
            print(f"\n⚠️  Index '{index_name}' n'existe pas encore.")
            print("   Il sera créé automatiquement lors de l'indexation.")
            return False
            
    except Exception as e:
        print(f"\n❌ Erreur Pinecone: {e}")
        return False


def run_pinecone_indexing():
    """Indexe les documents dans Pinecone."""
    from embeddings.indexing_pipeline import IndexingPipeline, IndexingConfig
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration pour Pinecone
    config = IndexingConfig(
        chunks_file=os.path.join(base_path, "data", "chunks.json"),
        vector_store_dir=os.path.join(base_path, "vector_store"),
        vector_backend="pinecone",  # <- Utiliser Pinecone
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        metric="cosine",  # Pinecone préfère cosine
        batch_size=100,   # Pinecone gère bien les gros batches
        show_progress=True,
        pinecone_index=os.getenv("PINECONE_INDEX", "rag-index"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
    )
    
    print("\n" + "=" * 60)
    print("🚀 INDEXATION DANS PINECONE")
    print("=" * 60)
    print(f"Index: {config.pinecone_index}")
    print(f"Namespace: {config.pinecone_namespace}")
    print(f"Cloud: {config.pinecone_cloud}")
    print(f"Region: {config.pinecone_region}")
    print(f"Modèle: {config.model_name}")
    print("=" * 60)
    
    # Créer et exécuter le pipeline
    pipeline = IndexingPipeline(config)
    result = pipeline.run()
    
    # Résumé
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DE L'INDEXATION")
    print("=" * 60)
    print(f"Status: {'✅ SUCCÈS' if result.success else '❌ ÉCHEC'}")
    print(f"Chunks indexés: {result.chunks_indexed}")
    print(f"Dimension: {result.dimension}")
    print(f"Modèle: {result.model}")
    print(f"Temps: {result.processing_time}")
    
    if result.errors:
        print(f"\n⚠️  Erreurs ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    
    return result.success


def main():
    parser = argparse.ArgumentParser(description="Indexer les documents dans Pinecone")
    parser.add_argument("--check-only", action="store_true", 
                       help="Vérifier l'index sans indexer")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🌲 PINECONE INDEXATION TOOL")
    print("=" * 60)
    
    # Vérifier les variables d'environnement
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("\n❌ PINECONE_API_KEY non définie!")
        print("\nUtilisation:")
        print('  export PINECONE_API_KEY="votre-clé"')
        print('  export PINECONE_INDEX="rag-index"  # optionnel')
        print("  python run_indexing_pinecone.py")
        return 1
    
    print(f"✅ PINECONE_API_KEY: {api_key[:10]}...")
    print(f"📦 PINECONE_INDEX: {os.getenv('PINECONE_INDEX', 'rag-index')}")
    
    if args.check_only:
        # Juste vérifier l'index
        success = check_pinecone_index()
        return 0 if success else 1
    else:
        # Vérifier d'abord, puis indexer
        check_pinecone_index()
        
        print("\n" + "-" * 60)
        response = input("Voulez-vous indexer les documents maintenant? (o/n): ")
        if response.lower() in ['o', 'oui', 'y', 'yes']:
            success = run_pinecone_indexing()
            
            # Vérifier après indexation
            print("\n")
            check_pinecone_index()
            
            return 0 if success else 1
        else:
            print("Indexation annulée.")
            return 0


if __name__ == "__main__":
    sys.exit(main())

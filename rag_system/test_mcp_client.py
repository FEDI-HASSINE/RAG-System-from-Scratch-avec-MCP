import requests
import json
import time

BASE_URL = "http://localhost:8000"

def log(msg, color="\033[0m"):
    print(f"{color}{msg}\033[0m")

def test_mcp_pipeline():
    log("\n🚀 Testing MCP Server RAG Pipeline (Phase 5)...", "\033[1;35m")

    # 1. Health Check
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            log("✅ Server is reachable", "\033[32m")
        else:
            log(f"❌ Server health check failed: {resp.text}", "\033[31m")
            return
    except requests.exceptions.ConnectionError:
        log("❌ Could not connect to server. Is it running? (uvicorn main:app ...)", "\033[31m")
        return

    # 2. List Tools
    resp = requests.get(f"{BASE_URL}/tools")
    tools = resp.json()
    log(f"✅ Discovered {len(tools)} tools: {[t['name'] for t in tools]}", "\033[32m")
    
    # 3. Retrieve chunks (Search in FAISS via MCP)
    query = "What is the privacy policy about?"
    log(f"\nQUERY: '{query}'", "\033[1;33m")
    
    payload = {
        "tool": "retrieve_chunks",
        "params": {
            "query": query,
            "top_k": 5
        }
    }
    
    log("📡 Sending 'retrieve_chunks' request...", "\033[36m")
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/mcp", json=payload)
    t1 = time.time()
    
    if resp.status_code == 200:
        data = resp.json()
        chunks = data['result']['chunks']
        log(f"✅ Retrieved {len(chunks)} chunks in {(t1-t0):.2f}s", "\033[32m")
        for i, chunk in enumerate(chunks[:2]):
            print(f"   Result {i+1}: {chunk['text'][:100]}...")
            
        # 4. Rerank the results (via MCP)
        if chunks:
            log("\n📡 Sending 'rerank' request...", "\033[36m")
            # Correction: 'chunks' doit être une liste de dictionnaires, pas une liste de strings
            rerank_payload = {
                "tool": "rerank",
                "params": {
                    "query": query,
                    "chunks": chunks  # On passe les objets chunks complets (qui sont déjà des dicts avec 'text')
                }
            }
            
            resp = requests.post(f"{BASE_URL}/mcp", json=rerank_payload)
            if resp.status_code == 200:
                rerank_data = resp.json()
                
                # Check for success
                if not rerank_data['result'].get('success', True):
                     log(f"❌ Rerank failed: {rerank_data['result'].get('error')}", "\033[31m")
                     return

                ranked_chunks = rerank_data['result']['chunks']
                
                if ranked_chunks:
                    log(f"✅ Reranking complete. Best score: {ranked_chunks[0]['rerank_score']:.4f}", "\033[32m")
                    best_chunk = ranked_chunks[0]
                    log(f"🏆 Best Result: {best_chunk['text'][:150]}...", "\033[1;32m")
                else:
                    log("⚠️ Reranking returned no chunks", "\033[33m")
            else:
                log(f"❌ Rerank failed: {resp.text}", "\033[31m")
    else:
        log(f"❌ Retrieval failed: {resp.text}", "\033[31m")

if __name__ == "__main__":
    test_mcp_pipeline()
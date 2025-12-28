# 🤖 Agents — RAG Agent & LLM (Phase 6)

## 🎯 Objectif

Orchestrer le pipeline RAG complet : récupération → reranking → génération de réponse.

## 🔧 Problèmes résolus

| Problème | Solution |
|----------|----------|
| Appels MCP manuels | `mcp_client.py` — Client HTTP avec retries |
| Multi-providers LLM | `llm_service.py` — OpenAI, Ollama, Mock |
| Hallucinations | `prompts.py` — Prompts stricts "basé sur le contexte" |
| Pipeline complexe | `rag_agent.py` — Orchestration avec trace |
| Pas de fallback | Mock LLM si API indisponible |

## 📁 Fichiers

```
agents/
├── mcp_client.py    # Client HTTP pour MCP
├── llm_service.py   # Service LLM unifié
├── prompts.py       # Templates de prompts
├── rag_agent.py     # Agent RAG principal
└── requirements.txt
```

## 🚀 Utilisation

### Réponse simple
```python
from agents.rag_agent import rag_answer

answer = rag_answer("What are the security measures?")
print(answer)
```

### Avec trace complète
```python
from agents.rag_agent import RAGAgent

agent = RAGAgent()
response = agent.answer("Explain the system architecture")

print(response.answer)
print(response.sources)

for step in response.trace.steps:
    print(f"{step.name}: {step.duration_ms}ms")
```

## 🔄 Pipeline RAG

```
Question
   │
   ▼
┌─────────────────┐
│  1. Retrieve    │ → MCP retrieve_chunks
└─────────────────┘
   │
   ▼
┌─────────────────┐
│  2. Rerank      │ → MCP rerank
└─────────────────┘
   │
   ▼
┌─────────────────┐
│  3. Build       │ → Prompt avec contexte
│     Context     │
└─────────────────┘
   │
   ▼
┌─────────────────┐
│  4. LLM         │ → OpenAI / Ollama / Mock
│     Generate    │
└─────────────────┘
   │
   ▼
Réponse + Sources + Trace
```

## 🤖 Providers LLM

| Provider | Configuration | Usage |
|----------|---------------|-------|
| OpenAI | `OPENAI_API_KEY` | Production |
| Ollama | Local `localhost:11434` | Dev/Offline |
| Mock | Aucune | Tests |

```python
from agents.llm_service import LLMConfig, LLMProvider

config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    temperature=0.1
)
```

## 🛡️ Zero Hallucination

Les prompts dans `prompts.py` imposent :
- Réponse **uniquement** basée sur le contexte fourni
- Citation explicite des sources
- Aveu d'ignorance si info absente

```python
SYSTEM_PROMPT = """
Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni.
Si l'information n'est pas dans le contexte, dis "Je n'ai pas cette information".
Ne jamais inventer de faits.
"""
```

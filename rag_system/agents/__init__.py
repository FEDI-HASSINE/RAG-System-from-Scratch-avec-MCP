"""
RAG System Agents Module

This module provides:
- MCPClient: HTTP client for MCP server communication
- LLMService: Unified interface for LLM providers
- RAGAgent: Complete RAG pipeline agent
- PromptLibrary: Template management for prompts
"""

from agents.mcp_client import (
    MCPClient,
    MCPClientConfig,
    MCPToolResult,
    get_mcp_client,
    call_mcp
)

from agents.llm_service import (
    LLMService,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    get_llm_service,
    call_llm
)

from agents.prompts import (
    PromptLibrary,
    PromptTemplate,
    get_prompt_library
)

from agents.rag_agent import (
    RAGAgent,
    RAGConfig,
    RAGResponse,
    RAGTrace,
    RetrievalStrategy,
    get_rag_agent,
    rag_answer
)

__all__ = [
    # MCP Client
    "MCPClient",
    "MCPClientConfig", 
    "MCPToolResult",
    "get_mcp_client",
    "call_mcp",
    
    # LLM Service
    "LLMService",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "get_llm_service",
    "call_llm",
    
    # Prompts
    "PromptLibrary",
    "PromptTemplate",
    "get_prompt_library",
    
    # RAG Agent
    "RAGAgent",
    "RAGConfig",
    "RAGResponse",
    "RAGTrace",
    "RetrievalStrategy",
    "get_rag_agent",
    "rag_answer"
]

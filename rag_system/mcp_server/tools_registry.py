"""
MCP Tools Registry - Central mapping of tool names to functions

This registry provides:
- Unified tool lookup by name
- Tool metadata (description, parameters schema)
- Easy extension mechanism for new tools
- Tool validation and introspection
"""

import os
import sys
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field, asdict
from functools import wraps
import inspect
import time
import logging

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tool classes
from mcp_server.tools.embed_text import EmbedTextTool, get_embed_text_tool
from mcp_server.tools.retrieve_chunks import RetrieveChunksTool, get_retrieve_chunks_tool
from mcp_server.tools.rerank import RerankTool, get_rerank_tool


# Create tool wrapper functions that match the expected signature
def embed_text_tool(**kwargs) -> Dict[str, Any]:
    """Wrapper for embed_text tool."""
    tool = get_embed_text_tool()
    return tool(kwargs)


def retrieve_chunks_tool(**kwargs) -> Dict[str, Any]:
    """Wrapper for retrieve_chunks tool."""
    tool = get_retrieve_chunks_tool()
    return tool(kwargs)


def rerank_tool(**kwargs) -> Dict[str, Any]:
    """Wrapper for rerank tool."""
    tool = get_rerank_tool()
    return tool(kwargs)


# ============================================================
# Tool Definition Schema
# ============================================================

@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ToolDefinition:
    """Complete definition of an MCP tool."""
    name: str
    description: str
    function: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "parameters": [p.to_dict() for p in self.parameters]
        }


# ============================================================
# Tool Wrapper with Timing and Error Handling
# ============================================================

def wrap_tool(func: Callable, tool_name: str) -> Callable:
    """Wrap a tool function with timing and error handling."""
    @wraps(func)
    def wrapper(**kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            result = func(**kwargs)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"result": result}
            
            # Add execution metadata
            result["_meta"] = {
                "tool": tool_name,
                "execution_time_ms": round(execution_time_ms, 2),
                "success": True
            }
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "_meta": {
                    "tool": tool_name,
                    "execution_time_ms": round(execution_time_ms, 2),
                    "success": False
                }
            }
    
    return wrapper


# ============================================================
# Tool Definitions
# ============================================================

TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "embed_text": ToolDefinition(
        name="embed_text",
        description="Generate embeddings for text using sentence-transformers or OpenAI models",
        function=embed_text_tool,
        category="embeddings",
        parameters=[
            ToolParameter(
                name="text",
                type="string | array",
                description="Single text or list of texts to embed",
                required=True
            ),
            ToolParameter(
                name="model_type",
                type="string",
                description="Model type: 'sentence-transformers' or 'openai'",
                required=False,
                default="sentence-transformers"
            ),
            ToolParameter(
                name="model_name",
                type="string",
                description="Specific model name (optional)",
                required=False,
                default=None
            )
        ]
    ),
    
    "retrieve_chunks": ToolDefinition(
        name="retrieve_chunks",
        description="Retrieve relevant chunks from the vector store using semantic search",
        function=retrieve_chunks_tool,
        category="retrieval",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query text",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of results to return",
                required=False,
                default=5
            ),
            ToolParameter(
                name="threshold",
                type="float",
                description="Maximum distance threshold for filtering",
                required=False,
                default=None
            ),
            ToolParameter(
                name="source_filter",
                type="string",
                description="Filter results by source file",
                required=False,
                default=None
            ),
            ToolParameter(
                name="section_filter",
                type="string",
                description="Filter results by section (partial match)",
                required=False,
                default=None
            )
        ]
    ),
    
    "rerank": ToolDefinition(
        name="rerank",
        description="Rerank retrieved chunks using a cross-encoder model for improved relevance",
        function=rerank_tool,
        category="retrieval",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Original search query",
                required=True
            ),
            ToolParameter(
                name="chunks",
                type="array",
                description="List of chunks to rerank (each with 'text' field)",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Number of top results to return after reranking",
                required=False,
                default=5
            ),
            ToolParameter(
                name="model_name",
                type="string",
                description="Cross-encoder model name",
                required=False,
                default="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        ]
    )
}


# ============================================================
# Tools Registry Class
# ============================================================

class ToolsRegistry:
    """
    Central registry for MCP tools.
    
    Provides:
    - Tool registration and lookup
    - Tool execution with error handling
    - Tool introspection and documentation
    """
    
    def __init__(self):
        """Initialize the registry with default tools."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._wrapped_tools: Dict[str, Callable] = {}
        self.logger = logging.getLogger("mcp.registry")
        
        # Register default tools
        for name, definition in TOOL_DEFINITIONS.items():
            self.register(definition)
    
    def register(self, definition: ToolDefinition) -> None:
        """
        Register a new tool.
        
        Args:
            definition: ToolDefinition with function and metadata
        """
        self._tools[definition.name] = definition
        self._wrapped_tools[definition.name] = wrap_tool(
            definition.function, 
            definition.name
        )
        self.logger.info(f"Registered tool: {definition.name}")
    
    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            del self._wrapped_tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def get(self, tool_name: str) -> Optional[Callable]:
        """
        Get a wrapped tool function by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Wrapped tool function or None if not found
        """
        return self._wrapped_tools.get(tool_name)
    
    def get_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get tool definition by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolDefinition or None if not found
        """
        return self._tools.get(tool_name)
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool result dictionary
            
        Raises:
            KeyError: If tool not found
        """
        tool_func = self.get(tool_name)
        if tool_func is None:
            raise KeyError(f"Tool not found: {tool_name}")
        
        return tool_func(**params)
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())
    
    def list_definitions(self) -> List[Dict]:
        """Get list of all tool definitions as dictionaries."""
        return [d.to_dict() for d in self._tools.values()]
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tool names in a specific category."""
        return [
            name for name, defn in self._tools.items()
            if defn.category == category
        ]
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
    
    def __contains__(self, tool_name: str) -> bool:
        """Support 'in' operator for tool lookup."""
        return self.has_tool(tool_name)
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)


# ============================================================
# Global Registry Instance
# ============================================================

# Create global registry instance
_registry: Optional[ToolsRegistry] = None


def get_registry() -> ToolsRegistry:
    """Get or create the global tools registry."""
    global _registry
    if _registry is None:
        _registry = ToolsRegistry()
    return _registry


# Legacy compatibility - dict-like access
TOOLS: Dict[str, Callable] = {}


def _init_legacy_tools():
    """Initialize legacy TOOLS dict for backward compatibility."""
    global TOOLS
    registry = get_registry()
    for name in registry.list_tools():
        TOOLS[name] = registry.get(name)


# Initialize on import
_init_legacy_tools()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing Tools Registry...")
    print("=" * 60)
    
    registry = get_registry()
    
    # List tools
    print("\n📦 Registered Tools:")
    for tool_name in registry.list_tools():
        defn = registry.get_definition(tool_name)
        print(f"  • {tool_name} [{defn.category}]: {defn.description[:50]}...")
    
    # Test tool execution
    print("\n🧪 Test 1: embed_text")
    result = registry.execute("embed_text", {
        "text": "Test sentence for embedding"
    })
    print(f"   Success: {result.get('_meta', {}).get('success')}")
    print(f"   Dimension: {result.get('dimension')}")
    print(f"   Time: {result.get('_meta', {}).get('execution_time_ms')}ms")
    
    print("\n🧪 Test 2: retrieve_chunks")
    result = registry.execute("retrieve_chunks", {
        "query": "data protection",
        "top_k": 2
    })
    print(f"   Success: {result.get('_meta', {}).get('success')}")
    print(f"   Found: {result.get('total_found', 0)} chunks")
    
    print("\n🧪 Test 3: Tool not found")
    try:
        registry.execute("unknown_tool", {})
    except KeyError as e:
        print(f"   ✅ Correctly raised KeyError: {e}")
    
    print("\n🧪 Test 4: Legacy TOOLS dict")
    print(f"   TOOLS keys: {list(TOOLS.keys())}")
    
    print("\n✅ Registry tests complete!")

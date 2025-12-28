"""
MCP Client - HTTP client for communicating with MCP Server

This client provides:
- Easy-to-use interface for MCP tool calls
- Automatic retries and error handling
- Request/response logging
- Batch operations support
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import httpx
    HTTP_CLIENT = "httpx"
except ImportError:
    import requests
    HTTP_CLIENT = "requests"


# ============================================================
# Configuration
# ============================================================

@dataclass
class MCPClientConfig:
    """Configuration for MCP Client."""
    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    log_requests: bool = True


# ============================================================
# Response Models
# ============================================================

@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""
    tool: str
    success: bool
    result: Dict[str, Any]
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    request_id: Optional[str] = None
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to result."""
        return self.result.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from result with default."""
        return self.result.get(key, default)


@dataclass
class MCPBatchResult:
    """Result from batch MCP calls."""
    results: List[MCPToolResult]
    total: int
    successful: int
    failed: int
    total_time_ms: float


# ============================================================
# MCP Client
# ============================================================

class MCPClient:
    """
    Client for communicating with MCP Server.
    
    Usage:
        client = MCPClient()
        
        # Single tool call
        result = client.call("embed_text", {"text": "Hello world"})
        
        # Retrieve chunks
        chunks = client.retrieve_chunks("What is data protection?", top_k=5)
        
        # Full RAG pipeline
        answer = client.rag_pipeline("What are the security measures?")
    """
    
    def __init__(self, config: Optional[MCPClientConfig] = None):
        """Initialize the MCP client."""
        self.config = config or MCPClientConfig()
        self.logger = logging.getLogger("mcp.client")
        
        # Initialize HTTP client
        if HTTP_CLIENT == "httpx":
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        else:
            self._session = requests.Session()
        
        # Track statistics
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time_ms": 0.0
        }
    
    def _make_request(self, 
                      method: str, 
                      endpoint: str, 
                      json_data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to MCP server."""
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if HTTP_CLIENT == "httpx":
                    if method == "GET":
                        response = self._client.get(endpoint)
                    else:
                        response = self._client.post(endpoint, json=json_data)
                    response.raise_for_status()
                    return response.json()
                else:
                    if method == "GET":
                        response = self._session.get(url, timeout=self.config.timeout)
                    else:
                        response = self._session.post(
                            url, 
                            json=json_data, 
                            timeout=self.config.timeout
                        )
                    response.raise_for_status()
                    return response.json()
                    
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}): {e}. Retrying..."
                    )
                    time.sleep(self.config.retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to MCP server: {e}")
    
    def call(self, tool: str, params: Optional[Dict[str, Any]] = None) -> MCPToolResult:
        """
        Call an MCP tool.
        
        Args:
            tool: Name of the tool to call
            params: Parameters for the tool
            
        Returns:
            MCPToolResult with tool output
        """
        params = params or {}
        start_time = time.time()
        
        self._stats["total_calls"] += 1
        
        if self.config.log_requests:
            self.logger.info(f"Calling tool: {tool} | Params: {str(params)[:100]}...")
        
        try:
            response = self._make_request("POST", "/mcp", {
                "tool": tool,
                "params": params
            })
            
            execution_time = (time.time() - start_time) * 1000
            self._stats["total_time_ms"] += execution_time
            
            result = response.get("result", {})
            meta = result.get("_meta", {})
            success = meta.get("success", True)
            
            if success:
                self._stats["successful_calls"] += 1
            else:
                self._stats["failed_calls"] += 1
            
            return MCPToolResult(
                tool=tool,
                success=success,
                result=result,
                execution_time_ms=meta.get("execution_time_ms", execution_time),
                error=result.get("error"),
                request_id=response.get("request_id")
            )
            
        except Exception as e:
            self._stats["failed_calls"] += 1
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Tool call failed: {tool} | Error: {e}")
            
            return MCPToolResult(
                tool=tool,
                success=False,
                result={},
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    # ============================================================
    # Convenience Methods for RAG Tools
    # ============================================================
    
    def embed_text(self, text: str, model_type: str = None) -> MCPToolResult:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            model_type: Optional model type
            
        Returns:
            MCPToolResult with embedding vector
        """
        params = {"text": text}
        if model_type:
            params["model_type"] = model_type
        return self.call("embed_text", params)
    
    def retrieve_chunks(self, 
                        query: str, 
                        top_k: int = 5,
                        source_filter: Optional[str] = None,
                        threshold: Optional[float] = None) -> MCPToolResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            source_filter: Filter by source file
            threshold: Minimum similarity threshold
            
        Returns:
            MCPToolResult with retrieved chunks
        """
        params = {"query": query, "top_k": top_k}
        if source_filter:
            params["source_filter"] = source_filter
        if threshold is not None:
            params["threshold"] = threshold
        return self.call("retrieve_chunks", params)
    
    def rerank(self, 
               query: str, 
               chunks: List[Dict],
               top_k: Optional[int] = None) -> MCPToolResult:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: Original query
            chunks: Chunks to rerank
            top_k: Number of top results to return
            
        Returns:
            MCPToolResult with reranked chunks
        """
        params = {"query": query, "chunks": chunks}
        if top_k:
            params["top_k"] = top_k
        return self.call("rerank", params)
    
    # ============================================================
    # Batch Operations
    # ============================================================
    
    def call_batch(self, calls: List[Dict[str, Any]]) -> MCPBatchResult:
        """
        Execute multiple tool calls in batch.
        
        Args:
            calls: List of {"tool": str, "params": dict}
            
        Returns:
            MCPBatchResult with all results
        """
        start_time = time.time()
        results = []
        
        for call_spec in calls:
            result = self.call(call_spec["tool"], call_spec.get("params", {}))
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        
        return MCPBatchResult(
            results=results,
            total=len(results),
            successful=successful,
            failed=len(results) - successful,
            total_time_ms=total_time
        )
    
    # ============================================================
    # Server Info
    # ============================================================
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self._make_request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception:
            return False
    
    def get_status(self) -> Dict:
        """Get server status."""
        return self._make_request("GET", "/status")
    
    def list_tools(self) -> List[Dict]:
        """List all available tools."""
        return self._make_request("GET", "/tools")
    
    def get_stats(self) -> Dict:
        """Get client statistics."""
        return {
            **self._stats,
            "avg_time_ms": (
                self._stats["total_time_ms"] / self._stats["total_calls"]
                if self._stats["total_calls"] > 0 else 0
            )
        }
    
    # ============================================================
    # Context Manager
    # ============================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the HTTP client."""
        if HTTP_CLIENT == "httpx":
            self._client.close()


# ============================================================
# Global Client Instance
# ============================================================

_client_instance: Optional[MCPClient] = None


def get_mcp_client(config: Optional[MCPClientConfig] = None) -> MCPClient:
    """Get or create global MCP client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = MCPClient(config)
    return _client_instance


# ============================================================
# Convenience Functions
# ============================================================

def call_mcp(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple function to call MCP tool.
    
    Args:
        tool: Tool name
        params: Tool parameters
        
    Returns:
        Tool result dictionary
    """
    client = get_mcp_client()
    result = client.call(tool, params)
    return result.result


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing MCP Client...")
    print("=" * 60)
    
    # Check if server is running
    client = MCPClient()
    
    print("\n🔍 Checking server health...")
    if client.health_check():
        print("   ✅ Server is healthy")
    else:
        print("   ❌ Server is not reachable")
        print("   Start the server with: uvicorn mcp_server.main:app --reload")
        exit(1)
    
    print("\n📊 Server status:")
    status = client.get_status()
    print(f"   Tools: {status.get('tools')}")
    print(f"   Uptime: {status.get('uptime_seconds')}s")
    
    print("\n🧪 Test 1: embed_text")
    result = client.embed_text("Test embedding via MCP client")
    print(f"   Success: {result.success}")
    print(f"   Dimension: {result.get('dimension')}")
    print(f"   Time: {result.execution_time_ms:.2f}ms")
    
    print("\n🧪 Test 2: retrieve_chunks")
    result = client.retrieve_chunks("data protection", top_k=2)
    print(f"   Success: {result.success}")
    print(f"   Found: {result.get('total_found', 0)} chunks")
    
    if result.success and result.get("chunks"):
        print("\n🧪 Test 3: rerank")
        chunks = result.get("chunks", [])
        rerank_result = client.rerank("data protection", chunks, top_k=2)
        print(f"   Success: {rerank_result.success}")
        if rerank_result.success:
            for chunk in rerank_result.get("chunks", [])[:2]:
                print(f"   • [{chunk.get('rank')}] Score: {chunk.get('rerank_score', 0):.2f}")
    
    print("\n📈 Client stats:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ MCP Client tests complete!")

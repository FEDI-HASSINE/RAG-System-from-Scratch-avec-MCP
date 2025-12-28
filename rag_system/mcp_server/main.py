"""
MCP Server - FastAPI server for RAG system tools

This server provides:
- Unified /mcp endpoint for all tool calls
- Tool discovery and documentation
- Centralized logging and observability
- Error handling and validation
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.tools_registry import get_registry, ToolsRegistry


# ============================================================
# Logging Configuration
# ============================================================

def setup_logging():
    """Configure centralized logging."""
    # Create logs directory
    logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            # File handler for all logs
            logging.FileHandler(logs_dir / "mcp.log", encoding="utf-8"),
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    # Create MCP-specific logger
    mcp_logger = logging.getLogger("mcp")
    mcp_logger.setLevel(logging.INFO)
    
    # Request/response logger (more detailed)
    request_handler = logging.FileHandler(logs_dir / "requests.log", encoding="utf-8")
    request_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    request_logger = logging.getLogger("mcp.requests")
    request_logger.addHandler(request_handler)
    request_logger.setLevel(logging.INFO)
    
    return mcp_logger


# Initialize logging
logger = setup_logging()


# ============================================================
# Pydantic Models
# ============================================================

class MCPRequest(BaseModel):
    """Request model for MCP tool calls."""
    tool: str = Field(..., description="Name of the tool to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool": "retrieve_chunks",
                "params": {
                    "query": "What is data protection?",
                    "top_k": 5
                },
                "request_id": "req-12345"
            }
        }


class MCPResponse(BaseModel):
    """Response model for MCP tool calls."""
    tool: str
    result: Dict[str, Any]
    request_id: Optional[str] = None
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool": "retrieve_chunks",
                "result": {
                    "chunks": [],
                    "total_found": 5,
                    "_meta": {"success": True, "execution_time_ms": 45.2}
                },
                "request_id": "req-12345",
                "timestamp": "2025-12-28T10:30:00"
            }
        }


class MCPError(BaseModel):
    """Error response model."""
    error: str
    error_type: str
    tool: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: str


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str
    category: str
    version: str
    parameters: List[Dict[str, Any]]


class ServerStatus(BaseModel):
    """Server status model."""
    status: str
    version: str
    tools_count: int
    tools: List[str]
    uptime_seconds: float


# ============================================================
# Application Lifecycle
# ============================================================

# Track server start time
server_start_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global server_start_time
    
    # Startup
    server_start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 MCP Server starting...")
    
    # Initialize registry
    registry = get_registry()
    logger.info(f"📦 Loaded {len(registry)} tools: {registry.list_tools()}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("🛑 MCP Server shutting down...")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="MCP RAG Server",
    description="""
    Model Context Protocol (MCP) Server for RAG System.
    
    This server exposes all RAG tools (embeddings, retrieval, reranking) 
    through a unified JSON API for agent interactions.
    
    ## Tools Available
    - **embed_text**: Generate text embeddings
    - **retrieve_chunks**: Semantic search in vector store
    - **rerank**: Cross-encoder reranking for improved relevance
    
    ## Usage
    Send POST requests to `/mcp` with tool name and parameters.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request Logging Middleware
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    request_logger = logging.getLogger("mcp.requests")
    
    # Generate request ID if not provided
    request_id = request.headers.get("X-Request-ID", f"req-{int(time.time() * 1000)}")
    
    # Log request
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log
    request_logger.info(
        f"{request_id} | {request.method} {request.url.path} | "
        f"Status: {response.status_code} | Duration: {duration_ms:.2f}ms"
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with server info."""
    return {
        "name": "MCP RAG Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "mcp_endpoint": "/mcp"
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=ServerStatus, tags=["Info"])
async def server_status():
    """Get server status and statistics."""
    registry = get_registry()
    
    uptime = 0.0
    if server_start_time:
        uptime = (datetime.now() - server_start_time).total_seconds()
    
    return ServerStatus(
        status="running",
        version="1.0.0",
        tools_count=len(registry),
        tools=registry.list_tools(),
        uptime_seconds=round(uptime, 2)
    )


@app.get("/tools", response_model=List[ToolInfo], tags=["Tools"])
async def list_tools():
    """List all available MCP tools."""
    registry = get_registry()
    return registry.list_definitions()


@app.get("/tools/{tool_name}", response_model=ToolInfo, tags=["Tools"])
async def get_tool(tool_name: str):
    """Get information about a specific tool."""
    registry = get_registry()
    definition = registry.get_definition(tool_name)
    
    if definition is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {tool_name}"
        )
    
    return definition.to_dict()


@app.post("/mcp", response_model=MCPResponse, tags=["MCP"])
async def call_tool(request: MCPRequest):
    """
    Execute an MCP tool.
    
    This is the main endpoint for agent-tool communication.
    Send the tool name and parameters to execute any registered tool.
    
    ## Example Request
    ```json
    {
        "tool": "retrieve_chunks",
        "params": {
            "query": "What is data protection?",
            "top_k": 5
        }
    }
    ```
    
    ## Example Response
    ```json
    {
        "tool": "retrieve_chunks",
        "result": {
            "chunks": [...],
            "total_found": 5,
            "_meta": {"success": true, "execution_time_ms": 45.2}
        },
        "timestamp": "2025-12-28T10:30:00"
    }
    ```
    """
    registry = get_registry()
    
    # Check if tool exists
    if not registry.has_tool(request.tool):
        logger.warning(f"Tool not found: {request.tool}")
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {request.tool}. Available tools: {registry.list_tools()}"
        )
    
    # Log the request
    logger.info(f"Executing tool: {request.tool} | Params: {json.dumps(request.params)[:200]}")
    
    try:
        # Execute the tool
        result = registry.execute(request.tool, request.params)
        
        # Check for tool-level errors
        if result.get("_meta", {}).get("success") is False:
            logger.error(f"Tool error: {request.tool} | Error: {result.get('error')}")
        else:
            logger.info(
                f"Tool success: {request.tool} | "
                f"Time: {result.get('_meta', {}).get('execution_time_ms', 0)}ms"
            )
        
        return MCPResponse(
            tool=request.tool,
            result=result,
            request_id=request.request_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error executing {request.tool}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing tool: {str(e)}"
        )


@app.post("/mcp/batch", tags=["MCP"])
async def call_tools_batch(requests: List[MCPRequest]):
    """
    Execute multiple MCP tools in sequence.
    
    Useful for chaining operations (e.g., retrieve then rerank).
    """
    results = []
    
    for req in requests:
        try:
            response = await call_tool(req)
            results.append({
                "tool": req.tool,
                "success": True,
                "result": response.result
            })
        except HTTPException as e:
            results.append({
                "tool": req.tool,
                "success": False,
                "error": e.detail
            })
    
    return {
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r["success"]),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=MCPError(
            error=exc.detail,
            error_type="HTTPException",
            request_id=request.headers.get("X-Request-ID"),
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=MCPError(
            error="Internal server error",
            error_type=type(exc).__name__,
            request_id=request.headers.get("X-Request-ID"),
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting MCP RAG Server")
    print("=" * 60)
    print("📍 URL: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔧 MCP Endpoint: http://localhost:8000/mcp")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


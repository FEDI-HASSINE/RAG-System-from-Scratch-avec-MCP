# MCP Server Logs
This directory contains log files for the MCP server.

## Log Files
- `mcp.log` - Main server log (all operations)
- `requests.log` - HTTP request/response log

## Log Format
```
{timestamp} | {level} | {logger_name} | {message}
```

## Note
Log files are created automatically when the server starts.
These files are gitignored to avoid committing logs.

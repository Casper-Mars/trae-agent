# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""MCP Tool implementation for integrating with MCP servers."""

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPToolDef, CallToolResult

from .base import Tool, ToolParameter, ToolExecResult, ToolCallArguments


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    type: str  # "stdio", "http", "websocket"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True
    env: Optional[Dict[str, str]] = None


class MCPTool(Tool):
    """A tool that wraps an MCP server tool."""
    
    def __init__(self, mcp_tool_def: MCPToolDef, server_config: MCPServerConfig, session: ClientSession):
        self.mcp_tool_def = mcp_tool_def
        self.server_config = server_config
        self.session = session
        self._name = f"mcp_{server_config.name}_{mcp_tool_def.name}"
        self._description = mcp_tool_def.description or f"MCP tool {mcp_tool_def.name} from {server_config.name}"
        self._parameters = self._convert_mcp_parameters(mcp_tool_def.inputSchema)
        super().__init__()
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_parameters(self) -> List[ToolParameter]:
        return self._parameters
    
    def _convert_mcp_parameters(self, input_schema: Optional[Dict[str, Any]]) -> List[ToolParameter]:
        """Convert MCP input schema to ToolParameter list."""
        if not input_schema or "properties" not in input_schema:
            return []
        
        parameters = []
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])
        
        for param_name, param_def in properties.items():
            param_type = param_def.get("type", "string")
            description = param_def.get("description", f"Parameter {param_name}")
            enum_values = param_def.get("enum")
            items = param_def.get("items")
            is_required = param_name in required_fields
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=description,
                enum=enum_values,
                items=items,
                required=is_required
            ))
        
        return parameters
    
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the MCP tool."""
        try:
            # Call the MCP tool through the session
            result = await self.session.call_tool(self.mcp_tool_def.name, arguments)
            
            if result.isError:
                return ToolExecResult(
                    output=None,
                    error=f"MCP tool error: {result.content[0].text if result.content else 'Unknown error'}",
                    error_code=1
                )
            
            # Extract output from result
            output_parts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    output_parts.append(content.text)
                elif hasattr(content, 'data'):
                    output_parts.append(str(content.data))
            
            output = "\n".join(output_parts) if output_parts else "Tool executed successfully"
            
            return ToolExecResult(
                output=output,
                error=None,
                error_code=0
            )
            
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=f"Error executing MCP tool {self.mcp_tool_def.name}: {str(e)}",
                error_code=1
            )
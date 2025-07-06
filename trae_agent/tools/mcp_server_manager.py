# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""MCP Server Manager for handling MCP server lifecycle and connections."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPToolDef

from .mcp_tool import MCPTool, MCPServerConfig
from .base import Tool

logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages MCP server connections and tool discovery."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.stdio_contexts: Dict[str, Any] = {}
        self._running = False
    
    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        if not config.enabled:
            logger.info(f"MCP server {config.name} is disabled, skipping")
            return
        
        self.servers[config.name] = config
        logger.info(f"Added MCP server configuration: {config.name}")
    
    async def start_all_servers(self) -> None:
        """Start all configured MCP servers."""
        if self._running:
            logger.warning("MCP servers are already running")
            return
        
        self._running = True
        
        for server_name, config in self.servers.items():
            try:
                await self._start_server(server_name, config)
            except Exception as e:
                logger.error(f"Failed to start MCP server {server_name}: {e}")
    
    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers."""
        if not self._running:
            return
        
        for server_name in list(self.sessions.keys()):
            await self._stop_server(server_name)
        
        self._running = False
    
    async def _start_server(self, server_name: str, config: MCPServerConfig) -> None:
        """Start a single MCP server."""
        try:
            if config.type == "stdio":
                await self._start_stdio_server(server_name, config)
            elif config.type == "http":
                await self._start_http_server(server_name, config)
            elif config.type == "websocket":
                await self._start_websocket_server(server_name, config)
            else:
                raise ValueError(f"Unsupported MCP server type: {config.type}")
            
            logger.info(f"Successfully started MCP server: {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            raise
    
    async def _start_stdio_server(self, server_name: str, config: MCPServerConfig) -> None:
        """Start a stdio-based MCP server."""
        logger.info(f"Starting stdio MCP server: {server_name}")
        
        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env=config.env
            )
            
            # Create session with stdio transport
            stdio_context = stdio_client(server_params)
            read, write = await stdio_context.__aenter__()
            session = ClientSession(read, write)
            
            # Store the context for cleanup
            self.stdio_contexts[server_name] = stdio_context
            
            # Initialize the session
            await session.initialize()
            
            # Store the session
            self.sessions[server_name] = session
            
            # Discover tools
            await self._discover_tools(server_name, config, session)
            
        except Exception as e:
            logger.error(f"Failed to start stdio MCP server {server_name}: {e}")
            raise
    
    async def _start_http_server(self, server_name: str, config: MCPServerConfig) -> None:
        """Start an HTTP-based MCP server."""
        # TODO: Implement HTTP transport in phase 3
        raise NotImplementedError("HTTP transport will be implemented in phase 3")
    
    async def _start_websocket_server(self, server_name: str, config: MCPServerConfig) -> None:
        """Start a WebSocket-based MCP server."""
        # TODO: Implement WebSocket transport in phase 3
        raise NotImplementedError("WebSocket transport will be implemented in phase 3")
    
    async def _stop_server(self, server_name: str) -> None:
        """Stop a single MCP server."""
        if server_name in self.sessions:
            try:
                session = self.sessions[server_name]
                # Close the session if it has a close method
                if hasattr(session, 'close'):
                    await session.close()
                del self.sessions[server_name]
                
                # Clean up stdio context if exists
                if server_name in self.stdio_contexts:
                    try:
                        stdio_context = self.stdio_contexts[server_name]
                        await stdio_context.__aexit__(None, None, None)
                        del self.stdio_contexts[server_name]
                    except Exception as e:
                        logger.error(f"Error cleaning up stdio context for {server_name}: {e}")
                
                # Remove tools from this server
                tools_to_remove = [tool_name for tool_name, tool in self.tools.items() 
                                 if tool.server_config.name == server_name]
                for tool_name in tools_to_remove:
                    del self.tools[tool_name]
                
                logger.info(f"Stopped MCP server: {server_name}")
                
            except Exception as e:
                logger.error(f"Error stopping MCP server {server_name}: {e}")
    
    async def _discover_tools(self, server_name: str, config: MCPServerConfig, session: ClientSession) -> None:
        """Discover tools from an MCP server."""
        try:
            # List available tools
            tools_result = await session.list_tools()
            
            for mcp_tool_def in tools_result.tools:
                # Create MCPTool wrapper
                mcp_tool = MCPTool(mcp_tool_def, config, session)
                tool_name = mcp_tool.get_name()
                
                self.tools[tool_name] = mcp_tool
                logger.info(f"Discovered MCP tool: {tool_name} from server {server_name}")
            
            logger.info(f"Discovered {len(tools_result.tools)} tools from server {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from server {server_name}: {e}")
            raise
    
    def get_all_tools(self) -> List[Tool]:
        """Get all discovered MCP tools."""
        return list(self.tools.values())
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific MCP tool by name."""
        return self.tools.get(tool_name)
    
    def get_server_tools(self, server_name: str) -> List[MCPTool]:
        """Get all tools from a specific server."""
        return [tool for tool in self.tools.values() 
                if tool.server_config.name == server_name]
    
    def is_running(self) -> bool:
        """Check if the manager is running."""
        return self._running
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all servers."""
        health_status = {}
        
        for server_name, session in self.sessions.items():
            try:
                # Try to list tools as a health check
                await session.list_tools()
                health_status[server_name] = True
            except Exception as e:
                logger.warning(f"Health check failed for server {server_name}: {e}")
                health_status[server_name] = False
        
        return health_status
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for automatic server lifecycle management."""
        try:
            await self.start_all_servers()
            yield self
        finally:
            await self.stop_all_servers()
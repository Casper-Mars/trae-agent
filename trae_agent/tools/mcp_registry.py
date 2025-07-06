# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""MCP Tool Registry for integrating MCP tools with the existing tool system."""

import logging
from typing import Dict, List, Optional, Type

from .base import Tool
from .mcp_tool import MCPTool
from .mcp_server_manager import MCPServerManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


class MCPToolRegistry:
    """Registry for managing MCP tools integration with existing tools."""
    
    def __init__(self, config: Config):
        self.config = config
        self.server_manager = MCPServerManager()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the MCP tool registry."""
        if self._initialized:
            logger.warning("MCP tool registry is already initialized")
            return
        
        if not self.config.mcp_servers:
            logger.info("No MCP servers configured, skipping initialization")
            return
        
        try:
            # Add all configured servers
            for server_name, server_config in self.config.mcp_servers.items():
                if not server_config.enabled:
                    continue
                self.server_manager.add_server(server_config)
            
            # Start all servers and discover tools
            await self.server_manager.start_all_servers()
            
            self._initialized = True
            logger.info(f"MCP tool registry initialized with {len(self.get_mcp_tools())} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tool registry: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the MCP tool registry."""
        if not self._initialized:
            return
        
        try:
            await self.server_manager.stop_all_servers()
            self._initialized = False
            logger.info("MCP tool registry shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during MCP tool registry shutdown: {e}")
    
    def get_mcp_tools(self) -> List[MCPTool]:
        """Get all available MCP tools."""
        if not self._initialized:
            return []
        
        return self.server_manager.get_all_tools()
    
    def get_tool_by_name(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific MCP tool by name."""
        if not self._initialized:
            return None
        
        return self.server_manager.get_tool(tool_name)
    
    def get_tools_by_server(self, server_name: str) -> List[MCPTool]:
        """Get all tools from a specific MCP server."""
        if not self._initialized:
            return []
        
        return self.server_manager.get_server_tools(server_name)
    
    def is_initialized(self) -> bool:
        """Check if the registry is initialized."""
        return self._initialized
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all MCP servers."""
        if not self._initialized:
            return {}
        
        return await self.server_manager.health_check()
    
    def get_server_status(self) -> Dict[str, Dict[str, any]]:
        """Get status information for all MCP servers."""
        status = {}
        
        for server_name, server_config in self.config.mcp_servers.items():
            tools = self.get_tools_by_server(server_name)
            status[server_name] = {
                "type": server_config.type,
                "tool_count": len(tools),
                "tools": [tool.get_name() for tool in tools],
                "running": server_name in self.server_manager.sessions
            }
        
        return status


def create_enhanced_tool_registry(config: Config, base_tools: Dict[str, Type[Tool]]) -> Dict[str, Type[Tool]]:
    """Create an enhanced tool registry that includes both base tools and MCP tools.
    
    Note: This function returns tool classes for the base tools, but MCP tools
    need to be instantiated through the MCPToolRegistry after initialization.
    """
    enhanced_registry = base_tools.copy()
    
    # MCP tools will be added dynamically after initialization
    # This function just prepares the registry structure
    
    return enhanced_registry
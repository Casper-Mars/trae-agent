# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tools module for Trae Agent."""

from .base import Tool, ToolResult, ToolCall, ToolExecutor
from .bash_tool import BashTool
from .edit_tool import TextEditorTool
from .sequential_thinking_tool import SequentialThinkingTool
from .task_done_tool import TaskDoneTool
from .mcp_tool import MCPTool, MCPServerConfig
from .mcp_server_manager import MCPServerManager
from .mcp_registry import MCPToolRegistry, create_enhanced_tool_registry

__all__ = [
    "Tool",
    "ToolResult",
    "ToolCall",
    "ToolExecutor",
    "BashTool",
    "TextEditorTool",
    "SequentialThinkingTool",
    "TaskDoneTool",
    "MCPTool",
    "MCPServerConfig",
    "MCPServerManager",
    "MCPToolRegistry",
    "create_enhanced_tool_registry"
]

tools_registry = {
    "bash": BashTool,
    "str_replace_based_edit_tool": TextEditorTool,
    "sequentialthinking": SequentialThinkingTool,
    "task_done": TaskDoneTool
}
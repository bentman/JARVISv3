"""
Integration tests for MCP Dispatcher
"""
import pytest
import asyncio
from backend.mcp_servers.base_server import mcp_dispatcher


@pytest.mark.asyncio
async def test_mcp_dispatcher_enhancements():
    """Test MCP dispatcher enhancements"""
    # Test basic tool calls
    read_result = await mcp_dispatcher.call_tool("read_file", {"path": "core/voice.py"})
    assert "success" in read_result

    list_result = await mcp_dispatcher.call_tool("list_files", {"directory": "."})
    assert "success" in list_result

    # Test system info tool
    system_info = await mcp_dispatcher.call_tool("system_info", {})
    assert "success" in system_info
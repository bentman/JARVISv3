"""
Unit tests for Search System (SearchNode, Providers, MCP)
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from backend.ai.workflows.search_node import search_node
from backend.mcp_servers.base_server import mcp_dispatcher
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, BudgetState, UserPreferences, HardwareState, ContextBudget

@pytest.mark.asyncio
async def test_search_node():
    """Test search node execution (Unified Search)"""
    context = TaskContext(
        system_context=SystemContext(
            user_id="test", session_id="test_sess",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=8, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0), 
            user_preferences=UserPreferences(privacy_level="medium")
        ),
        workflow_context=WorkflowContext(
            workflow_id="test_wf", workflow_name="test", initiating_query="JARVISv3 architecture",
            user_intent=UserIntent(type=TaskType.RESEARCH, confidence=1.0, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )
    
    # Mock privacy service to allow web search
    with patch("backend.ai.workflows.search_node.privacy_service") as mock_privacy:
        mock_privacy.classify_data.return_value = "public"
        mock_privacy.should_process_locally.return_value = False
        mock_privacy.redact_sensitive_data.side_effect = lambda x: x
        
        # Mock search providers
        with patch("backend.ai.workflows.search_node.DuckDuckGoProvider") as MockDDG:
            instance = MockDDG.return_value
            instance.search = AsyncMock(return_value=[{"title": "Test", "url": "http://test.com", "snippet": "test"}])
            search_node.providers = {"duckduckgo": instance}
            
            result = await search_node.execute(context, {})
            assert result["success"] is True
            assert len(result["web"]) > 0

@pytest.mark.asyncio
async def test_mcp_dispatcher():
    """Test MCP tool dispatching"""
    # 1. Read File
    # Mock open is tricky in async, testing list_files instead which uses os
    res_list = await mcp_dispatcher.call_tool("list_files", {"directory": "."})
    assert res_list["success"] is True
    
    # 2. Web Search Tool
    with patch("ddgs.DDGS") as MockDDGS:
        instance = MockDDGS.return_value
        instance.text.return_value = [{"title": "Test", "href": "http://test.com", "body": "test"}]

        res_search = await mcp_dispatcher.call_tool("web_search", {"query": "test"})
        assert res_search["success"] is True
        assert res_search["count"] > 0

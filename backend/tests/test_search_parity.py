import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from backend.ai.workflows.search_node import SearchNode
from backend.ai.context.schemas import TaskContext, WorkflowContext, SystemContext, UserPreferences, HardwareState, BudgetState, UserIntent, TaskType

@pytest.fixture
def mock_task_context():
    return TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="test_session",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=8, cpu_usage=10, available_tiers=["cpu"], current_load=0.1),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences(privacy_level="medium")
        ),
        workflow_context=WorkflowContext(
            workflow_id="test_wf",
            workflow_name="test_search",
            initiating_query="Where is the best pizza in New York?",
            user_intent=UserIntent(type=TaskType.RESEARCH, confidence=1.0, description="Find pizza", priority=1)
        )
    )

@pytest.mark.asyncio
async def test_search_node_privacy_redaction(mock_task_context):
    """Verify that SearchNode redacts PII before calling providers"""
    node = SearchNode()
    
    # Mock privacy service
    with patch("backend.ai.workflows.search_node.privacy_service") as mock_privacy:
        mock_privacy.classify_data.return_value = MagicMock(sensitive=False)
        mock_privacy.should_process_locally.return_value = False
        mock_privacy.redact_sensitive_data.return_value = "redacted query"
        
        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.search.return_value = [{"title": "Pizza", "url": "http://pizza.com", "snippet": "Best pizza", "source": "test"}]
        node.providers = {"test_provider": mock_provider}
        
        await node.execute(mock_task_context, {})
        
        # Check if redact_sensitive_data was called
        mock_privacy.redact_sensitive_data.assert_called_with(mock_task_context.workflow_context.initiating_query)
        # Check if provider was called with redacted query
        mock_provider.search.assert_called_with("redacted query", max_results=ANY)

@pytest.mark.asyncio
async def test_search_node_caching(mock_task_context):
    """Verify that SearchNode uses cache_service"""
    node = SearchNode()
    
    # Mock cache service
    with patch("backend.ai.workflows.search_node.cache_service") as mock_cache:
        mock_cache.get_json = AsyncMock(return_value={"success": True, "cached": True})
        
        result = await node.execute(mock_task_context, {})
        
        # Check if cache was checked
        assert mock_cache.get_json.called
        # Check if cached result was returned
        assert result["cached"] is True

@pytest.mark.asyncio
async def test_search_node_provider_fallback(mock_task_context):
    """Verify that SearchNode falls back to next provider if one fails"""
    node = SearchNode()
    
    # Mock providers
    mock_p1 = AsyncMock()
    mock_p1.search.side_effect = Exception("Failed")
    
    mock_p2 = AsyncMock()
    mock_p2.search.return_value = [{"title": "Success", "url": "http://ok.com", "snippet": "Found it", "source": "p2"}]
    
    node.providers = {
        "p1": mock_p1,
        "p2": mock_p2
    }
    
    # Mock privacy to allow web search
    with patch("backend.ai.workflows.search_node.privacy_service") as mock_privacy:
        mock_privacy.should_process_locally.return_value = False
        mock_privacy.redact_sensitive_data.return_value = mock_task_context.workflow_context.initiating_query
        
        # Disable cache for this test
        with patch("backend.ai.workflows.search_node.cache_service.get_json", return_value=None):
            result = await node.execute(mock_task_context, {})
            
            assert mock_p1.search.called
            assert mock_p2.search.called
            assert result["web"][0]["source"] == "p2"
            assert result["success"] is True

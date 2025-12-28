"""
Integration Test for Memory & Workflows
Verifies that memory can be actively used during a multi-step workflow.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, BudgetState, UserPreferences, HardwareState, ContextBudget

@pytest.mark.asyncio
async def test_memory_integration_in_workflow():
    """Test that a workflow can store and retrieve data from memory"""
    
    engine = WorkflowEngine()
    
    # 1. Store Node
    engine.add_node(WorkflowNode(
        id="store_fact",
        type=NodeType.ACTIVE_MEMORY,
        description="Store a fact",
        conditions={"operation": "store", "content": "The sky is blue"}
    ))
    
    # 2. Retrieve Node
    engine.add_node(WorkflowNode(
        id="retrieve_fact",
        type=NodeType.ACTIVE_MEMORY,
        description="Retrieve the fact",
        dependencies=["store_fact"],
        conditions={"operation": "retrieve", "query": "sky"}
    ))
    
    context = TaskContext(
        system_context=SystemContext(
            user_id="test", session_id="test_sess",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=8, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0), user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="memory_integration_test", workflow_name="test", initiating_query="test",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=1.0, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )
    
    # Mock the underlying memory service to avoid DB/VectorStore hits in unit tests
    with patch("backend.ai.workflows.active_memory.memory_service") as mock_memory:
        mock_memory.add_message = AsyncMock(return_value="msg_123")
        mock_memory.semantic_search = AsyncMock(return_value=[{"content": "The sky is blue", "score": 0.9}])
        
        result = await engine.execute_workflow(context)
        
        # Verify
        assert result["status"] == "completed"
        
        # Check store result
        store_res = result["results"]["store_fact"]
        assert store_res["status"] == "stored"
        assert store_res["message_id"] == "msg_123"
        
        # Check retrieve result
        retr_res = result["results"]["retrieve_fact"]
        assert retr_res["status"] == "retrieved"
        assert retr_res["results"][0]["content"] == "The sky is blue"

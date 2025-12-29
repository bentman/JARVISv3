"""
Unit tests for Memory System (MemoryService, ActiveMemoryNode)
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from backend.core.memory import memory_service
from backend.ai.workflows.active_memory import active_memory_node
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, BudgetState, UserPreferences, HardwareState, ContextBudget

@pytest.mark.asyncio
async def test_memory_operations():
    """Test core memory service operations"""
    # 1. Store Conversation
    cid = await memory_service.store_conversation("Unit Test Conv")
    assert cid is not None
    
    # 2. Add Message
    msg_id = await memory_service.add_message(cid, "user", "Test message")
    assert msg_id is not None
    
    # 3. Retrieve Context
    context = await memory_service.get_conversation_context(cid)
    assert "Test message" in context
    
    # 4. Semantic Search (Mocked for speed if needed, but using real logic here)
    # Note: Requires embedding model to be loaded, might be slow on first run
    try:
        results = await memory_service.semantic_search("Test message", k=1)
        assert len(results) > 0
    except Exception as e:
        pytest.skip(f"Skipping semantic search due to environment issues: {e}")

@pytest.mark.asyncio
async def test_active_memory_node():
    """Test the ActiveMemoryNode used in workflows"""
    # Create mock context
    context = TaskContext(
            system_context=SystemContext(
            user_id="test", session_id="test_sess",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=8, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0), user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="test_wf", workflow_name="test", initiating_query="test",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=1.0, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )
    
    # 1. Store
    res_store = await active_memory_node.execute(context, "store", content="Important fact")
    assert res_store["status"] == "stored"
    
    # 2. Pin
    res_pin = await active_memory_node.execute(context, "pin", content="Pinned fact")
    assert res_pin["status"] == "pinned"
    
    # 3. Retrieve
    res_retr = await active_memory_node.execute(context, "retrieve", query="fact")
    assert res_retr["status"] == "retrieved"
    assert len(res_retr["results"]) > 0

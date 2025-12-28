"""
Integration tests for Search Node
"""
import pytest
import asyncio
from backend.ai.workflows.search_node import search_node
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.mark.asyncio
async def test_search_node_enhancements():
    """Test search node enhancements"""
    # Create a test context
    system_context = SystemContext(
        user_id="test_user",
        session_id="test_session",
        hardware_state=HardwareState(
            gpu_usage=0.0,
            memory_available_gb=16.0,
            cpu_usage=20.0,
            current_load=0.1
        ),
        budget_state=BudgetState(
            cloud_spend_usd=0.0,
            monthly_limit_usd=100.0,
            remaining_pct=100.0
        ),
        user_preferences=UserPreferences(privacy_level="medium")
    )

    workflow_context = WorkflowContext(
        workflow_id="test_workflow",
        workflow_name="test_workflow",
        initiating_query="JARVISv3 architecture",
        user_intent=UserIntent(
            type=TaskType.RESEARCH,
            confidence=0.9,
            description="Research query",
            priority=3
        ),
        context_budget=ContextBudget()
    )

    task_context = TaskContext(
        system_context=system_context,
        workflow_context=workflow_context,
        additional_context={}
    )

    # Test unified search
    search_results = await search_node.execute(task_context, {})
    assert "success" in search_results
    assert "privacy_assessment" in search_results
    assert "retrieval_stats" in search_results
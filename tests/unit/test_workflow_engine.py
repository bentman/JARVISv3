"""
Unit tests for Workflow Engine
"""
import pytest
import asyncio
from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.mark.asyncio
async def test_workflow_engine_basic():
    """Test basic workflow engine functionality"""
    engine = WorkflowEngine()
    
    # Add a simple test node
    async def test_node_func(context, results):
        return {"status": "completed", "data": "test_result"}
    
    test_node = WorkflowNode(
        id="test_node",
        type=NodeType.LLM_WORKER,
        description="Test node for workflow engine",
        execute_func=test_node_func
    )
    
    engine.add_node(test_node)
    
    # Create a test context
    system_context = SystemContext(
        user_id="test_user_123",
        session_id="test_session_123",
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
        user_preferences=UserPreferences()
    )
    
    workflow_context = WorkflowContext(
        workflow_id="test_workflow_123",
        workflow_name="test_workflow",
        initiating_query="Test workflow execution",
        user_intent=UserIntent(
            type=TaskType.CHAT,
            confidence=0.9,
            description="Test workflow query",
            priority=3
        ),
        context_budget=ContextBudget()
    )
    
    task_context = TaskContext(
        system_context=system_context,
        workflow_context=workflow_context,
        additional_context={}
    )
    
    # Execute the workflow
    result = await engine.execute_workflow(task_context)
    assert result["status"] == "completed"
    assert "test_node" in result["results"]
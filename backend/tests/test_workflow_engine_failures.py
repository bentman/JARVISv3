"""
Unit tests for JARVISv3 WorkflowEngine failure modes and edge cases.
"""
import pytest
import asyncio
from datetime import datetime
from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType, NodeStatus
from backend.ai.context.schemas import (
    TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType,
    HardwareState, BudgetState, UserPreferences, ContextBudget
)

@pytest.fixture
def base_context():
    return TaskContext(
        system_context=SystemContext(
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
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="test_workflow",
            workflow_name="test_workflow",
            initiating_query="Test query",
            user_intent=UserIntent(
                type=TaskType.CHAT,
                confidence=0.9,
                description="Test query",
                priority=3
            ),
            context_budget=ContextBudget(max_tokens=1000, consumed_tokens=0)
        )
    )

@pytest.mark.asyncio
async def test_node_retry_success(base_context):
    """Test that a node retries and eventually succeeds"""
    engine = WorkflowEngine()
    attempt_count = 0
    
    async def failing_then_succeeding_node(context, results):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ValueError("First attempt failed")
        return {"result": "success on attempt 2"}
    
    engine.add_node(WorkflowNode(
        id="retry_node",
        type=NodeType.LLM_WORKER,
        description="Node that fails once then succeeds",
        execute_func=failing_then_succeeding_node,
        max_retries=3
    ))
    
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "completed"
    assert attempt_count == 2
    assert result["results"]["retry_node"]["result"] == "success on attempt 2"

@pytest.mark.asyncio
async def test_node_max_retries_exceeded(base_context):
    """Test that a node fails after exceeding max retries"""
    engine = WorkflowEngine()
    attempt_count = 0
    
    async def always_failing_node(context, results):
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError("Permanent failure")
    
    engine.add_node(WorkflowNode(
        id="failing_node",
        type=NodeType.LLM_WORKER,
        description="Node that always fails",
        execute_func=always_failing_node,
        max_retries=2
    ))
    
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "failed"
    assert attempt_count == 3  # Initial + 2 retries
    assert "Permanent failure" in result["error"]

@pytest.mark.asyncio
async def test_node_timeout(base_context):
    """Test that a node times out if it takes too long"""
    engine = WorkflowEngine()
    
    async def slow_node(context, results):
        await asyncio.sleep(2)
        return {"result": "too late"}
    
    engine.add_node(WorkflowNode(
        id="timeout_node",
        type=NodeType.LLM_WORKER,
        description="Slow node",
        execute_func=slow_node,
        timeout_seconds=1,
        max_retries=0
    ))
    
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "failed"
    # The error might be a TimeoutError or wrapped
    assert "timeout" in result["error"].lower()

@pytest.mark.asyncio
async def test_context_budget_enforcement(base_context):
    """Test that exceeding context budget stops execution"""
    engine = WorkflowEngine()
    
    # Set low budget
    base_context.workflow_context.context_budget.max_tokens = 50
    
    async def token_heavy_node(context, results):
        # Consume tokens
        context.update_tokens_consumed(100)
        return {"result": "consumed 100"}
    
    engine.add_node(WorkflowNode(
        id="heavy_node",
        type=NodeType.LLM_WORKER,
        description="Node that consumes many tokens",
        execute_func=token_heavy_node
    ))
    
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "failed"
    assert "Context budget exceeded" in result["error"]

@pytest.mark.asyncio
async def test_dependency_execution_order(base_context):
    """Test that nodes are executed in the correct dependency order"""
    engine = WorkflowEngine()
    execution_order = []
    
    async def node_a(context, results):
        execution_order.append("A")
        return {"data": "A"}
        
    async def node_b(context, results):
        execution_order.append("B")
        return {"data": "B"}
    
    engine.add_node(WorkflowNode(
        id="node_a",
        type=NodeType.ROUTER,
        description="First node",
        execute_func=node_a
    ))
    
    engine.add_node(WorkflowNode(
        id="node_b",
        type=NodeType.LLM_WORKER,
        description="Second node",
        dependencies=["node_a"],
        execute_func=node_b
    ))
    
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "completed"
    assert execution_order == ["A", "B"]

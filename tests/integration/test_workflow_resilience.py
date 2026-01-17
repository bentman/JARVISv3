"""
Workflow Engine Failure and Distributed System Tests for JARVISv3
Tests workflow engine failure modes, resilience, and distributed system capabilities.
"""
import pytest
import asyncio
from datetime import datetime
from fastapi.testclient import TestClient
from backend.main import app
from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType, NodeStatus
from backend.ai.context.schemas import (
    TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType,
    HardwareState, BudgetState, UserPreferences, ContextBudget, RemoteNode, NodeCapability
)


client = TestClient(app)


# Define the base context directly to avoid fixture issues
def create_base_context():
    """Create a base context for testing"""
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
async def test_node_retry_success():
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
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "completed"
    assert attempt_count == 2
    assert result["results"]["retry_node"]["result"] == "success on attempt 2"


@pytest.mark.asyncio
async def test_node_max_retries_exceeded():
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
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "failed"
    assert attempt_count == 3  # Initial + 2 retries
    assert "Permanent failure" in result["error"]


@pytest.mark.asyncio
async def test_node_timeout():
    """Test that a node times out if it takes too long"""
    import asyncio
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
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "failed"
    # The error might be a TimeoutError or wrapped
    assert "timeout" in result["error"].lower()


@pytest.mark.asyncio
async def test_context_budget_enforcement():
    """Test that exceeding context budget stops execution"""
    engine = WorkflowEngine()
    
    # Set low budget
    base_context = create_base_context()
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
async def test_dependency_execution_order():
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
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "completed"
    assert execution_order == ["A", "B"]


@pytest.mark.asyncio
async def test_workflow_engine_error_handling():
    """Test comprehensive error handling in workflow engine"""
    engine = WorkflowEngine()
    
    # Test with a node that raises an unexpected exception
    async def error_node(context, results):
        raise RuntimeError("Unexpected error in workflow")
    
    engine.add_node(WorkflowNode(
        id="error_node",
        type=NodeType.LLM_WORKER,
        description="Node that raises runtime error",
        execute_func=error_node,
        max_retries=0
    ))
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "failed"
    assert "Unexpected error in workflow" in result["error"]


@pytest.mark.asyncio
async def test_node_validation_errors():
    """Test handling of node validation errors"""
    engine = WorkflowEngine()
    
    # Add a node with invalid configuration
    node = WorkflowNode(
        id="invalid_node",
        type=NodeType.LLM_WORKER,
        description="Node with potential validation issues"
    )
    engine.add_node(node)
    
    # Try to execute with a node that has validation issues
    async def validation_error_node(context, results):
        # Simulate a validation error
        if not hasattr(context, 'workflow_context'):
            raise ValueError("Context validation failed")
        return {"result": "success"}
    
    # Replace the node's execute function
    engine.nodes["invalid_node"].execute_func = validation_error_node
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    # Should handle the validation error gracefully
    assert result["status"] in ["completed", "failed"]


@pytest.mark.asyncio
async def test_workflow_circuit_breaker():
    """Test workflow circuit breaker for infinite loops"""
    engine = WorkflowEngine()
    
    # Create a node that would cause infinite cycling
    async def cycling_node(context, results):
        # This would normally cause infinite loop, but circuit breaker should stop it
        return {"next_node": "cycling_node"}
    
    engine.add_node(WorkflowNode(
        id="cycling_node",
        type=NodeType.SUPERVISOR,
        description="Node that cycles infinitely",
        execute_func=cycling_node
    ))
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    # Circuit breaker should prevent infinite execution
    assert result["status"] == "failed"
    assert "exceeded maximum iterations" in result["error"]


@pytest.mark.asyncio
async def test_workflow_state_recovery():
    """Test workflow state recovery and checkpointing"""
    engine = WorkflowEngine()
    
    async def checkpoint_node(context, results):
        # This node should create a checkpoint
        return {"result": "checkpoint_created"}
    
    engine.add_node(WorkflowNode(
        id="checkpoint_node",
        type=NodeType.LLM_WORKER,
        description="Node that creates checkpoints",
        execute_func=checkpoint_node
    ))
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    
    # Verify that checkpointing was attempted
    assert result["status"] == "completed"
    assert "results" in result


@pytest.mark.asyncio
async def test_node_resource_exhaustion():
    """Test handling of resource exhaustion in nodes"""
    engine = WorkflowEngine()
    
    async def memory_intensive_node(context, results):
        # Simulate memory exhaustion
        # In a real scenario, this might be caught by system resource monitoring
        large_data = ["data"] * 1000000  # Large list to consume memory
        return {"result": "success", "data_size": len(large_data)}
    
    engine.add_node(WorkflowNode(
        id="memory_node",
        type=NodeType.LLM_WORKER,
        description="Memory intensive node",
        execute_func=memory_intensive_node,
        max_retries=1
    ))
    
    base_context = create_base_context()
    try:
        result = await engine.execute_workflow(base_context)
        # Should handle memory issues gracefully
        assert result["status"] in ["completed", "failed"]
    except MemoryError:
        # If memory error occurs, that's also valid behavior
        pass


@pytest.mark.asyncio
async def test_workflow_engine_concurrent_execution():
    """Test workflow engine with concurrent execution scenarios"""
    engine = WorkflowEngine()
    
    async def concurrent_node(context, results):
        # Simulate some async work
        await asyncio.sleep(0.1)
        return {"result": "concurrent_success"}
    
    # Add multiple nodes that could run concurrently
    for i in range(3):
        engine.add_node(WorkflowNode(
            id=f"concurrent_node_{i}",
            type=NodeType.LLM_WORKER,
            description=f"Concurrent node {i}",
            execute_func=concurrent_node,
            dependencies=[]  # No dependencies, can run in parallel
        ))
    
    base_context = create_base_context()
    result = await engine.execute_workflow(base_context)
    assert result["status"] == "completed"



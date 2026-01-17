"""
Test Cyclic Workflow Execution
Verifies the State Machine capabilities of the WorkflowEngine
"""
import pytest
import asyncio
from datetime import datetime, UTC
from typing import Dict, Any

from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget

@pytest.mark.asyncio
async def test_cyclic_workflow_execution():
    """Test that a workflow can loop back and self-correct"""
    
    engine = WorkflowEngine()
    
    # State to track attempts
    execution_state = {"attempts": 0}
    
    # 1. Generator Node (Fails first time, succeeds second time)
    async def generator_func(context, results):
        execution_state["attempts"] += 1
        print(f"Generator attempt: {execution_state['attempts']}")
        if execution_state["attempts"] < 2:
            return {"output": "bad_code"}
        return {"output": "good_code"}
        
    engine.add_node(WorkflowNode(
        id="generator",
        type=NodeType.LLM_WORKER,
        description="Generates code",
        execute_func=generator_func
    ))
    
    # 2. Validator Node
    async def validator_func(context, results):
        gen_output = results.get("generator", {}).get("output")
        print(f"Validating: {gen_output}")
        if gen_output == "good_code":
            return {"is_valid": True, "errors": []}
        return {"is_valid": False, "errors": ["Syntax error"]}
        
    engine.add_node(WorkflowNode(
        id="validator",
        type=NodeType.VALIDATOR,
        description="Validates code",
        dependencies=["generator"],
        execute_func=validator_func
    ))
    
    # 3. Reflector Node
    # Uses the standard ReflectorNode implementation via NodeType.REFLECTOR
    # We don't need a custom func, just configuration
    engine.add_node(WorkflowNode(
        id="reflector",
        type=NodeType.REFLECTOR,
        description="Checks validation",
        dependencies=["validator"],
        conditions={
            "target_node_id": "generator",
            "criteria": "Must be good_code"
        }
    ))
    
    # Create context
    context = TaskContext(
        system_context=SystemContext(
            user_id="test",
            session_id="test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="cyclic_test",
            workflow_name="cyclic_test",
            initiating_query="Test cycle",
            user_intent=UserIntent(type=TaskType.CODING, confidence=1.0, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )
    
    # Execute
    print("Starting cyclic execution...")
    result = await engine.execute_workflow(context)
    
    print(f"Final Result: {result}")
    
    # Verification
    assert result["status"] == "completed"
    assert execution_state["attempts"] == 2
    assert engine.state
    assert engine.state.iteration_count >= 3 # generator -> validator -> reflector -> generator -> validator -> reflector (approved)
    
    # Check execution history
    history = engine.state.execution_history
    # Should see generator twice
    gen_runs = [h for h in history if h["node_id"] == "generator"]
    assert len(gen_runs) == 2
    assert gen_runs[0]["result"]["output"] == "bad_code"
    assert gen_runs[1]["result"]["output"] == "good_code"

if __name__ == "__main__":
    asyncio.run(test_cyclic_workflow_execution())

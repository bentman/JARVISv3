"""
Test Dynamic Planning with Supervisor
Verifies that the Supervisor Agent can generate execution plans at runtime.
"""
import pytest
import asyncio
from typing import Dict, Any

from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget

@pytest.mark.asyncio
async def test_dynamic_planning():
    """Test that a supervisor can plan and execute a sequence"""
    
    engine = WorkflowEngine()
    
    # 1. Supervisor Node
    # Uses the standard SupervisorNode implementation via NodeType.SUPERVISOR
    engine.add_node(WorkflowNode(
        id="supervisor",
        type=NodeType.SUPERVISOR,
        description="Plans the workflow"
    ))
    
    # 2. Worker Nodes (that will be called dynamically)
    execution_log = []
    
    async def search_func(context, results):
        execution_log.append("search_web")
        return {"results": "AI Trends 2025..."}
        
    engine.add_node(WorkflowNode(
        id="search_web",
        type=NodeType.SEARCH_WEB,
        description="Search",
        execute_func=search_func
    ))
    
    async def llm_func(context, results):
        execution_log.append("llm_worker")
        return {"summary": "AI is growing."}
        
    engine.add_node(WorkflowNode(
        id="llm_worker",
        type=NodeType.LLM_WORKER,
        description="LLM",
        execute_func=llm_func
    ))
    
    # Create context with a query that triggers planning
    context = TaskContext(
        system_context=SystemContext(
            user_id="test",
            session_id="test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="dynamic_test",
            workflow_name="dynamic_test",
            initiating_query="Research AI trends and summarize them",
            user_intent=UserIntent(type=TaskType.RESEARCH, confidence=1.0, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )
    
    # Execute
    print("Starting dynamic execution...")
    result = await engine.execute_workflow(context)
    
    print(f"Final Result: {result}")
    
    # Verification
    # 1. Supervisor runs first (implicit start)
    # 2. Supervisor adds [search_web, llm_worker] to plan_queue
    # 3. Engine executes search_web
    # 4. Engine executes llm_worker
    
    assert "search_web" in execution_log
    assert "llm_worker" in execution_log
    assert execution_log == ["search_web", "llm_worker"]
    
    # Check plan queue was consumed
    assert engine.state
    print(f"Plan Queue at end: {engine.state.plan_queue}")
    print(f"Execution History: {engine.state.execution_history}")
    assert len(engine.state.plan_queue) == 0

if __name__ == "__main__":
    asyncio.run(test_dynamic_planning())

"""
Unit tests for Context System (Schemas, Builder, Lifecycle)
"""
import pytest
import asyncio
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget
from backend.ai.generators.context_builder import ContextBuilder
from backend.core.lifecycle import context_lifecycle_manager


@pytest.mark.asyncio
async def test_context_schemas_validation():
    """Test that context schemas work properly with validation"""
    # Create a complete context packet
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
        initiating_query="Hello, what can you do?",
        user_intent=UserIntent(
            type=TaskType.CHAT,
            confidence=0.9,
            description="Simple chat query",
            priority=3
        ),
        context_budget=ContextBudget()
    )
    
    task_context = TaskContext(
        system_context=system_context,
        workflow_context=workflow_context,
        additional_context={}
    )
    
    # Validate the context
    validation_errors = task_context.validate_context()
    assert len(validation_errors) == 0  # Should have no validation errors


@pytest.mark.asyncio
async def test_context_builder():
    """Test context builder functionality"""
    builder = ContextBuilder()
    
    # Build a test context
    task_context = await builder.build_task_context(
        user_id="test_user_123",
        session_id="test_session_123",
        workflow_id="test_build_context_123",
        workflow_name="test_build",
        initiating_query="Testing context building",
        task_type=TaskType.CHAT
    )
    
    assert task_context is not None
    assert task_context.system_context.user_id == "test_user_123"
    assert task_context.workflow_context.initiating_query == "Testing context building"
    
    # Test context size calculation
    context_size = await builder.get_context_size(task_context)
    assert context_size > 0


@pytest.mark.asyncio
async def test_context_lifecycle():
    """Test context lifecycle management"""
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
        initiating_query="Testing context lifecycle",
        user_intent=UserIntent(
            type=TaskType.CHAT,
            confidence=0.9,
            description="Test lifecycle management",
            priority=3
        ),
        context_budget=ContextBudget()
    )
    
    task_context = TaskContext(
        system_context=system_context,
        workflow_context=workflow_context,
        additional_context={}
    )
    
    # Test lifecycle management
    managed_context = await context_lifecycle_manager.manage_context_lifecycle(task_context)
    assert managed_context is not None
    
    # Test context metrics
    context_size = task_context.get_context_size()
    assert context_size > 0
    assert hasattr(task_context.workflow_context, 'accumulated_artifacts')
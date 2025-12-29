"""
Full System Integration Test for JARVISv3
Tests the complete workflow from API to backend services
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import asyncio
from backend.main import app
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_context_flow_integration():
    """Test the complete context flow from creation to validation"""
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
    
    # Test context size calculation
    context_size = task_context.get_context_size()
    assert context_size > 0

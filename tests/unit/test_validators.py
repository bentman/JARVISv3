"""
Unit tests for Validation System
"""
import pytest
import asyncio
from backend.ai.validators.code_check import ValidatorPipeline
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget
from backend.core.security import security_validator


@pytest.mark.asyncio
async def test_validator_pipeline():
    """Test validation pipeline"""
    validator = ValidatorPipeline()
    
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
        initiating_query="Hello, how are you?",
        user_intent=UserIntent(
            type=TaskType.CHAT,
            confidence=0.9,
            description="Test validation",
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
    validation_result = await validator.validate_task_context(task_context)
    assert validation_result.is_valid is True
    assert len(validation_result.errors) == 0


@pytest.mark.asyncio
async def test_security_validation():
    """Test security validation functionality"""
    # Test clean input
    clean_input = "What can you help me with?"
    security_result = await security_validator.validate_input(clean_input)
    assert security_result['is_valid'] is True
    assert security_result['has_critical'] is False
    assert len(security_result['issues']) == 0
    
    # Test input with PII (should be flagged but not critical)
    pii_input = "My email is test@example.com and my phone is 123-456-7890"
    pii_result = await security_validator.validate_input(pii_input)
    assert pii_result['is_valid'] is False  # Should flag PII
"""
Unit tests for Budget Service
"""
import pytest
import asyncio
from backend.core.budget import budget_service, BudgetService


@pytest.mark.asyncio
async def test_budget_service_get_state():
    """Test getting budget state for a user"""
    # Test with a fresh user ID that should return default budget state
    state = await budget_service.get_budget_state("fresh_test_user_123")

    assert state is not None
    assert hasattr(state, 'cloud_spend_usd')
    assert hasattr(state, 'monthly_limit_usd')
    assert hasattr(state, 'remaining_pct')
    assert hasattr(state, 'daily_spending')

    # Should return default state for non-existent user
    assert state.monthly_limit_usd == 100.0
    assert state.remaining_pct == 100.0
    assert state.cloud_spend_usd == 0.0


@pytest.mark.asyncio
async def test_budget_service_check_budget():
    """Test budget checking functionality"""
    # Test with default budget (should allow small costs)
    can_afford = await budget_service.check_budget("test_user_456", 1.0)
    assert can_afford is True

    # Test with large cost (should deny)
    can_afford_large = await budget_service.check_budget("test_user_456", 200.0)
    assert can_afford_large is False


@pytest.mark.asyncio
async def test_budget_service_log_usage():
    """Test logging usage and budget updates"""
    # Use a truly unique user ID to avoid interference from other tests
    import uuid
    unique_user = f"budget_test_user_{uuid.uuid4().hex[:8]}"

    # Log some usage
    await budget_service.log_usage(unique_user, "workflow_123", 100, 0.50)

    # Check that budget state reflects the usage
    state = await budget_service.get_budget_state(unique_user)
    assert state.cloud_spend_usd == 0.50  # Should reflect the logged cost

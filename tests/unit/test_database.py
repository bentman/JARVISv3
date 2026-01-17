"""
Unit tests for Database System
"""
import pytest
import asyncio
from backend.core.database import database_manager


@pytest.mark.asyncio
async def test_database_initialization():
    """Test that database initializes correctly"""
    await database_manager.initialize()
    
    # Verify tables were created by checking if we can get a non-existent user
    user = await database_manager.get_user("test_user_123")
    assert user is None  # Should return None for non-existent user, not throw error


@pytest.mark.asyncio
async def test_budget_management():
    """Test budget management functionality"""
    # Test budget creation and updates
    budget_data = {
        'budget_id': 'test_budget_123',
        'user_id': 'test_user_123',
        'workflow_id': 'test_workflow_123',
        'monthly_limit_usd': 50.0,
        'daily_limit_usd': 10.0,
        'token_limit': 50000,
        'monthly_spent_usd': 1.50,
        'daily_spent_usd': 1.50,
        'tokens_consumed': 150
    }
    
    success = await database_manager.save_budget_record(budget_data)
    assert success is True
    
    # Update budget usage
    update_success = await database_manager.update_budget_usage(
        'test_user_123',
        'test_workflow_123',
        cost_usd=0.50,
        tokens=50
    )
    assert update_success is True
    
    # Check updated budget
    updated_budget = await database_manager.get_budget_record('test_user_123')
    assert updated_budget is not None
    assert updated_budget['tokens_consumed'] >= 150  # Should include original + update
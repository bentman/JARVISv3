"""
Unit tests for Core Services (Auth, Database, Hardware, Budget, Privacy)
Refactored from monolithic test_production_readiness.py
"""
import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from backend.core.auth import auth_manager, User
from backend.core.database import database_manager
from backend.core.privacy import privacy_service
from backend.core.budget import budget_manager

@pytest.mark.asyncio
async def test_database_initialization():
    """Test that database initializes correctly"""
    await database_manager.initialize()
    user = await database_manager.get_user("non_existent_user")
    assert user is None

@pytest.mark.asyncio
async def test_auth_manager():
    """Test authentication manager functionality"""
    user = await auth_manager.create_user(
        username="unit_test_user",
        email="unit_test@JARVISv3.local",
        password="test_password",
        role="user"
    )
    assert user is not None
    assert user.username == "unit_test_user"
    
    api_key = await auth_manager.generate_api_key(user.user_id)
    assert len(api_key) > 0
    
    has_perm = await auth_manager.check_permission(user, "read")
    assert has_perm is True

@pytest.mark.asyncio
async def test_budget_management():
    """Test budget management functionality"""
    budget_data = {
        'budget_id': 'unit_test_budget',
        'user_id': 'unit_test_user_budget',
        'monthly_limit_usd': 50.0,
        'tokens_consumed': 100
    }
    success = await database_manager.save_budget_record(budget_data)
    assert success is True
    
    update_success = await database_manager.update_budget_usage(
        'unit_test_user_budget', 'test_wf', cost_usd=0.50, tokens=50
    )
    assert update_success is True

def test_privacy_service():
    """Test privacy service enhancements"""
    # Classification
    classification = privacy_service.classify_data("My email is test@example.com")
    assert str(classification) == "personal"
    
    # Redaction
    redacted = privacy_service.redact_sensitive_data("Contact 123-456-7890")
    assert "[PHONE_REDACTED]" in redacted
    
    # Anonymization
    anonymized = privacy_service.anonymize_data("John Doe lives here", "high")
    assert "[NAME_REDACTED]" in anonymized

"""
Unit tests for Authentication System
"""
import pytest
import asyncio
from backend.core.auth import auth_manager


@pytest.mark.asyncio
async def test_auth_manager():
    """Test authentication manager functionality"""
    # Test user creation
    user = await auth_manager.create_user(
        username="test_user",
        email="test@JARVISv3.local",
        password="test_password",
        role="user"
    )
    
    assert user is not None
    assert user.username == "test_user"
    
    # Test API key generation
    api_key = await auth_manager.generate_api_key(user.user_id)
    assert len(api_key) > 0
    
    # Test permission checking
    has_perm = await auth_manager.check_permission(user, "read")
    assert has_perm is True  # User should have read permission (default for user role)
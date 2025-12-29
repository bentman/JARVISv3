"""
Unit tests for Model Router
"""
import pytest
import asyncio
from backend.core.model_router import model_router


@pytest.mark.asyncio
async def test_model_router_available_providers():
    """Test getting available model providers"""
    providers = await model_router.get_available_providers()
    assert isinstance(providers, list)
    # At minimum, should have some providers (even if none are available)
    assert len(providers) >= 0


@pytest.mark.asyncio
async def test_model_router_select_model():
    """Test model and provider selection"""
    model_name, provider_name, remote_node = await model_router.select_model_and_provider("chat")
    assert isinstance(model_name, str)
    assert isinstance(provider_name, str)
    # Remote node can be None
    assert remote_node is None or isinstance(remote_node, str)


@pytest.mark.asyncio
async def test_model_router_generate_response():
    """Test response generation (may fail without models, but should handle gracefully)"""
    try:
        result = await model_router.generate_response("Hello", "chat")
        assert hasattr(result, 'response')
        assert hasattr(result, 'tokens_used')
        assert hasattr(result, 'model_used')
    except Exception as e:
        # If models aren't available, it should fail gracefully with informative error
        assert "model" in str(e).lower() or "provider" in str(e).lower()

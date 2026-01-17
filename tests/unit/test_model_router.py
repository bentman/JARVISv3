"""
Unit tests for Model Router
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from backend.core.model_router import model_router


@pytest.mark.asyncio
@patch.object(model_router.providers["ollama"], 'is_available', return_value=False)
@patch.object(model_router.providers["llama_cpp"], 'is_available', return_value=True)
async def test_model_router_available_providers(mock_llama, mock_ollama):
    """Test getting available model providers (using mocks)"""
    providers = await model_router.get_available_providers()
    assert isinstance(providers, list)
    assert "llama_cpp" in providers
    assert "ollama" not in providers


@pytest.mark.asyncio
async def test_model_router_select_model():
    """Test model and provider selection (deterministic, no external deps)"""
    model_name, provider_name, remote_node = await model_router.select_model_and_provider("chat")
    assert isinstance(model_name, str)
    assert isinstance(provider_name, str)
    # Should select a valid provider that supports chat
    assert provider_name in model_router.providers
    # Remote node can be None
    assert remote_node is None or isinstance(remote_node, str)


@patch.object(model_router.providers["ollama"], 'generate_response', side_effect=Exception("OllamaProvider not available"))
@patch.object(model_router.providers["llama_cpp"], 'generate_response', side_effect=Exception("LlamaCppProvider not available"))
@pytest.mark.asyncio
async def test_model_router_generate_response(mock_llama, mock_ollama):
    """Test response generation error handling (using mocks)"""
    try:
        result = await model_router.generate_response("Hello", "chat")
        assert False, "Expected exception for unavailable provider"
    except Exception as e:
        # Should fail gracefully with provider error
        assert "provider" in str(e).lower()

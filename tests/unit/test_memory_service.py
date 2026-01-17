"""
Unit tests for Memory Service
"""
import pytest
import asyncio
from backend.core.memory import memory_service


@pytest.mark.asyncio
async def test_memory_service_enhancements():
    """Test memory service enhancements"""
    # Test conversation storage
    conversation_id = await memory_service.store_conversation("Test Conversation")
    assert conversation_id is not None

    # Test message addition
    message_id = await memory_service.add_message(
        conversation_id, "user", "Hello, this is a test message", 10, "chat"
    )
    assert message_id is not None

    # Test conversation context retrieval
    context = await memory_service.get_conversation_context(conversation_id)
    assert isinstance(context, str)

    # Test combined search
    combined_results = await memory_service.search_and_retrieve_context("test", conversation_id)
    assert "semantic_matches" in combined_results
    assert "conversation_context" in combined_results
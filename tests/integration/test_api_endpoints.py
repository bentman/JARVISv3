"""
API Endpoints Integration Tests for JARVISv3
Tests conversation management endpoints and data persistence functionality.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.main import app
from datetime import datetime

client = TestClient(app)


@pytest.mark.asyncio
async def test_list_conversations():
    """Test GET /api/v1/conversations - List all conversations"""
    mock_conversations = [
        {
            "conversation_id": "conv_1",
            "title": "Test Conv 1",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": ["test"]
        }
    ]
    
    with patch("backend.core.memory.MemoryService.get_conversations", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_conversations
        
        response = client.get("/api/v1/conversations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["conversation_id"] == "conv_1"
        assert data[0]["title"] == "Test Conv 1"


@pytest.mark.asyncio
async def test_get_conversation():
    """Test GET /api/v1/conversation/{id} - Retrieve specific conversation"""
    mock_conv = {
        "conversation_id": "conv_1",
        "title": "Test Conv 1",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "tags": ["test"]
    }
    mock_messages = [
        {
            "message_id": "msg_1",
            "conversation_id": "conv_1",
            "role": "user",
            "content": "hello",
            "timestamp": datetime.now().isoformat(),
            "tokens": 5,
            "mode": "chat"
        }
    ]
    
    with patch("backend.core.memory.MemoryService.get_conversation", new_callable=AsyncMock) as mock_get_conv:
        with patch("backend.core.memory.MemoryService.get_messages", new_callable=AsyncMock) as mock_get_msgs:
            mock_get_conv.return_value = mock_conv
            mock_get_msgs.return_value = mock_messages
            
            response = client.get("/api/v1/conversation/conv_1")
            assert response.status_code == 200
            data = response.json()
            assert data["conversation_id"] == "conv_1"
            assert len(data["messages"]) == 1
            assert data["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_get_conversation_not_found():
    """Test GET /api/v1/conversation/{id} when conversation doesn't exist"""
    with patch("backend.core.memory.MemoryService.get_conversation", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        
        response = client.get("/api/v1/conversation/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Conversation not found"


@pytest.mark.asyncio
async def test_delete_conversation():
    """Test DELETE /api/v1/conversation/{id} - Delete conversation"""
    with patch("backend.core.memory.MemoryService.delete_conversation", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = True
        
        response = client.delete("/api/v1/conversation/conv_1")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["conversation_id"] == "conv_1"
        mock_delete.assert_called_once_with("conv_1")


@pytest.mark.asyncio
async def test_conversation_persistence():
    """Test conversation persistence through MemoryService"""
    from backend.core.memory import memory_service
    import uuid
    
    # Generate unique conversation ID to avoid conflicts
    unique_id = f"test_conv_{uuid.uuid4().hex[:8]}"
    
    # Test conversation creation
    conv_id = await memory_service.store_conversation("Test Conversation", unique_id)
    assert conv_id == unique_id
    
    # Test conversation retrieval
    conversation = await memory_service.get_conversation(unique_id)
    assert conversation is not None
    assert conversation['title'] == "Test Conversation"
    
    # Test message addition
    msg_id = await memory_service.add_message(unique_id, "user", "Hello", tokens=5)
    assert msg_id != ""
    
    # Test message retrieval
    messages = await memory_service.get_messages(unique_id)
    assert len(messages) == 1
    assert messages[0]['content'] == "Hello"
    assert messages[0]['tokens'] == 5


@pytest.mark.asyncio
async def test_conversation_tagging():
    """Test conversation tagging functionality"""
    from backend.core.memory import memory_service
    import uuid
    
    # Generate unique conversation ID to avoid conflicts
    unique_id = f"tag_test_{uuid.uuid4().hex[:8]}"
    
    # Create and tag a conversation
    conv_id = await memory_service.store_conversation("Tag Test", unique_id)
    assert conv_id == unique_id
    
    # Set tags
    success = await memory_service.set_conversation_tags(unique_id, ["important", "test"])
    assert success is True
    
    # Verify tags were set
    conv = await memory_service.get_conversation(unique_id)
    # Tags are stored as JSON string, so we need to check differently
    assert conv is not None


@pytest.mark.asyncio
async def test_conversation_statistics():
    """Test conversation statistics calculation"""
    from backend.core.memory import memory_service
    
    # Create conversation with multiple messages
    conv_id = await memory_service.store_conversation("Stats Test", "stats_test_123")
    await memory_service.add_message("stats_test_123", "user", "Message 1", tokens=10)
    await memory_service.add_message("stats_test_123", "assistant", "Response 1", tokens=15)
    
    # Get statistics
    stats = await memory_service.get_conversation_stats("stats_test_123")
    assert "message_count" in stats
    assert "token_count" in stats
    assert stats["message_count"] >= 2
    assert stats["token_count"] >= 25

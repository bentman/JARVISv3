import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.main import app
from datetime import datetime

client = TestClient(app)

@pytest.mark.asyncio
async def test_list_conversations():
    """Test GET /api/v1/conversations"""
    mock_conversations = [
        {
            "conversation_id": "conv_1",
            "title": "Test Conv 1",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
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
    """Test GET /api/v1/conversation/{id}"""
    mock_conv = {
        "conversation_id": "conv_1",
        "title": "Test Conv 1",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "tags": ["test"]
    }
    mock_messages = [
        {
            "message_id": "msg_1",
            "conversation_id": "conv_1",
            "role": "user",
            "content": "hello",
            "timestamp": datetime.now(),
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
    """Test GET /api/v1/conversation/{id} when not found"""
    with patch("backend.core.memory.MemoryService.get_conversation", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        
        response = client.get("/api/v1/conversation/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Conversation not found"

@pytest.mark.asyncio
async def test_delete_conversation():
    """Test DELETE /api/v1/conversation/{id}"""
    with patch("backend.core.memory.MemoryService.delete_conversation", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = True
        
        response = client.delete("/api/v1/conversation/conv_1")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["conversation_id"] == "conv_1"
        mock_delete.assert_called_once_with("conv_1")

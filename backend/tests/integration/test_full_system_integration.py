"""
Full System Integration Test for JARVISv3
Tests the complete workflow from API to backend services
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import asyncio
from backend.main import app
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_full_voice_session_integration(client):
    """Test the complete voice session flow: API -> Voice Service -> Chat Workflow -> Response"""
    audio_base64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="  # Empty WAV
    
    # Mock the services that would be called
    with patch("backend.main.voice_service") as mock_voice:
        with patch("backend.main.chat_workflow") as mock_chat:
            # Setup STT mock
            mock_voice.speech_to_text = AsyncMock(return_value=("Hello Jarvis", 0.9))
            
            # Setup Chat mock
            mock_chat.execute_chat = AsyncMock(return_value={
                "response": "Hello User",
                "conversation_id": "test_conv_id",
                "workflow_id": "test_wf_id"
            })
            
            # Setup TTS mock
            mock_voice.text_to_speech = AsyncMock(return_value=b"audio data")
            
            # Make request to the voice session endpoint
            response = client.post("/api/v1/voice/session", json={
                "audio_data": audio_base64,
                "conversation_id": "existing_conv",
                "mode": "chat"
            })
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            
            # Verify flow
            assert data["text_response"] == "Hello User"
            assert data["conversation_id"] == "test_conv_id"
            assert data["detected"] is True
            assert data["audio_data"] is not None  # Base64 encoded "audio data"
            
            # Verify that the services were called in the correct order
            mock_voice.speech_to_text.assert_called_once()
            mock_chat.execute_chat.assert_called_once()
            mock_voice.text_to_speech.assert_called_once_with("Hello User")


@pytest.mark.asyncio
async def test_conversation_management_integration(client):
    """Test conversation management API integration with memory service"""
    from backend.core.memory import MemoryService
    
    # Test listing conversations
    with patch.object(MemoryService, 'get_conversations', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = [
            {
                "conversation_id": "conv_1",
                "title": "Test Conv 1",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "tags": ["test"]
            }
        ]
        
        response = client.get("/api/v1/conversations")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["conversation_id"] == "conv_1"
        assert data[0]["title"] == "Test Conv 1"


@pytest.mark.asyncio
async def test_context_flow_integration():
    """Test the complete context flow from creation to validation"""
    # Create a complete context packet
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
        initiating_query="Hello, what can you do?",
        user_intent=UserIntent(
            type=TaskType.CHAT,
            confidence=0.9,
            description="Simple chat query",
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
    validation_errors = task_context.validate_context()
    assert len(validation_errors) == 0  # Should have no validation errors
    
    # Test context size calculation
    context_size = task_context.get_context_size()
    assert context_size > 0
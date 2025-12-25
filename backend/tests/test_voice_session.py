import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_voice_session_flow():
    """Test the unified voice session flow: STT -> Chat -> TTS"""
    
    # Mock data
    audio_base64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" # Empty WAV
    
    # Mock dependencies
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
            
            # Make request
            response = client.post("/api/v1/voice/session", json={
                "audio_data": audio_base64,
                "conversation_id": "existing_conv",
                "mode": "chat"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify flow
            assert data["text_response"] == "Hello User"
            assert data["conversation_id"] == "test_conv_id"
            assert data["detected"] is True
            assert data["audio_data"] is not None # Base64 encoded "audio data"
            
            # Verify calls
            mock_voice.speech_to_text.assert_called_once()
            mock_chat.execute_chat.assert_called_once()
            mock_voice.text_to_speech.assert_called_once_with("Hello User")

@pytest.mark.asyncio
async def test_voice_session_no_speech():
    """Test voice session when no speech is detected"""
    
    # Mock dependencies
    with patch("backend.main.voice_service") as mock_voice:
        # Setup STT mock to return empty
        mock_voice.speech_to_text = AsyncMock(return_value=("", 0.0))
        
        # Make request
        response = client.post("/api/v1/voice/session", json={
            "audio_data": "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
            "conversation_id": "existing_conv"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["detected"] is False
        assert data["text_response"] == ""

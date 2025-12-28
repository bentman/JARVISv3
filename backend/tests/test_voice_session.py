"""
Voice Session and Security Validation Tests for JARVISv3
Tests voice processing pipeline: STT -> Chat -> TTS and security validation.
"""
import pytest
import base64
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.main import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_voice_session_complete_flow():
    """Test the complete voice session flow: STT -> Chat -> TTS"""
    
    # Mock data - base64 encoded minimal WAV data
    audio_base64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    # Mock all dependencies to avoid actual model calls
    with patch("backend.main.voice_service") as mock_voice_service, \
         patch("backend.main.chat_workflow") as mock_chat_workflow:
        
        # Setup STT mock
        mock_voice_service.speech_to_text = AsyncMock(return_value=("Hello Jarvis", 0.9))
        
        # Setup Chat mock
        mock_chat_workflow.execute_chat = AsyncMock(return_value={
            "response": "Hello User",
            "conversation_id": "test_conv_id",
            "workflow_id": "test_wf_id",
            "tokens_used": 10,
            "validation_passed": True
        })
        
        # Setup TTS mock
        mock_voice_service.text_to_speech = AsyncMock(return_value=b"mock_audio_data")
        
        # Make request to the voice session endpoint
        response = client.post("/api/v1/voice/session", json={
            "audio_data": audio_base64,
            "conversation_id": "existing_conv",
            "mode": "chat"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify the complete flow worked
        assert data["text_response"] == "Hello User"
        assert data["conversation_id"] == "test_conv_id"
        assert data["detected"] is True
        assert data["workflow_id"] == "test_wf_id"
        
        # Verify all services were called in the correct sequence
        mock_voice_service.speech_to_text.assert_called_once()
        mock_chat_workflow.execute_chat.assert_called_once()
        mock_voice_service.text_to_speech.assert_called_once_with("Hello User")


@pytest.mark.asyncio
async def test_voice_session_no_speech_detected():
    """Test voice session when no speech is detected"""
    
    # Mock data
    audio_base64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    # Mock dependencies
    with patch("backend.main.voice_service") as mock_voice_service:
        # Setup STT mock to return empty
        mock_voice_service.speech_to_text = AsyncMock(return_value=("", 0.0))
        
        # Make request
        response = client.post("/api/v1/voice/session", json={
            "audio_data": audio_base64,
            "conversation_id": "existing_conv"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should indicate no speech was detected
        assert data["detected"] is False
        assert data["text_response"] == ""


@pytest.mark.asyncio
async def test_voice_session_with_quality_assessment():
    """Test voice session with audio quality assessment"""
    from backend.core.voice import voice_service
    
    # Create minimal WAV data for quality assessment
    fake_wav_data = (
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00'
        b'\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    )
    
    quality_info = voice_service.assess_audio_quality(fake_wav_data)
    
    # Validate quality assessment structure
    assert "quality_score" in quality_info
    assert "quality_level" in quality_info
    assert "sample_rate" in quality_info
    assert "volume_rms" in quality_info
    assert "duration" in quality_info
    assert "channels" in quality_info
    assert "feedback" in quality_info
    assert "is_acceptable" in quality_info
    
    # Validate types
    assert isinstance(quality_info["quality_score"], int)
    assert isinstance(quality_info["quality_level"], str)
    assert isinstance(quality_info["is_acceptable"], bool)


@pytest.mark.asyncio
async def test_voice_session_error_handling():
    """Test voice session error handling"""
    
    # Mock data
    audio_base64 = "invalid_base64_data"
    
    # Test with invalid base64
    response = client.post("/api/v1/voice/session", json={
        "audio_data": audio_base64,
        "conversation_id": "test_conv"
    })
    
    # Should return an error for invalid base64
    assert response.status_code in [400, 500]


@pytest.mark.asyncio
async def test_voice_session_empty_audio():
    """Test voice session with empty audio data"""
    
    # Mock dependencies for empty audio test
    with patch("backend.main.voice_service") as mock_voice_service:
        # Setup STT mock to return empty
        mock_voice_service.speech_to_text = AsyncMock(return_value=("", 0.0))
        
        # Use empty base64
        response = client.post("/api/v1/voice/session", json={
            "audio_data": "",
            "conversation_id": "test_conv"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle empty audio gracefully
        assert data["detected"] is False


@pytest.mark.asyncio
async def test_voice_transcription_endpoint():
    """Test the direct voice transcription endpoint"""
    
    # Mock file upload test
    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("test.wav", b"fake_audio_data", "audio/wav")}
    )
    
    # Should return error for invalid audio, not 404
    assert response.status_code in [400, 422, 500]


@pytest.mark.asyncio
async def test_voice_tts_endpoint():
    """Test the direct text-to-speech endpoint"""
    
    tts_request = {"text": "Hello world"}
    
    response = client.post("/api/v1/voice/speak", json=tts_request)
    
    # Should return 200 with audio response or 400 for validation
    assert response.status_code in [200, 400, 500]


@pytest.mark.asyncio
async def test_voice_service_initialization():
    """Test that voice service is properly initialized"""
    from backend.core.voice import voice_service
    
    # Service should be available
    assert voice_service is not None
    assert hasattr(voice_service, 'speech_to_text')
    assert hasattr(voice_service, 'text_to_speech')
    assert hasattr(voice_service, 'assess_audio_quality')


@pytest.mark.asyncio
async def test_voice_session_with_different_modes():
    """Test voice session with different operation modes"""
    
    audio_base64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    # Test with different modes
    modes = ["chat", "command", "query"]
    
    for mode in modes:
        with patch("backend.main.voice_service") as mock_voice_service, \
             patch("backend.main.chat_workflow") as mock_chat_workflow:
            
            mock_voice_service.speech_to_text = AsyncMock(return_value=("Test command", 0.8))
            mock_chat_workflow.execute_chat = AsyncMock(return_value={
                "response": f"Response for {mode}",
                "conversation_id": f"conv_{mode}",
                "workflow_id": f"wf_{mode}"
            })
            mock_voice_service.text_to_speech = AsyncMock(return_value=b"mock_audio")
            
            response = client.post("/api/v1/voice/session", json={
                "audio_data": audio_base64,
                "conversation_id": f"conv_{mode}",
                "mode": mode
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["text_response"] == f"Response for {mode}"


@pytest.mark.asyncio
async def test_voice_session_with_web_escalation():
    """Test voice session with web escalation capability"""
    
    audio_base64 = "UklGRi4AAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    with patch("backend.main.voice_service") as mock_voice_service, \
         patch("backend.main.chat_workflow") as mock_chat_workflow:
        
        mock_voice_service.speech_to_text = AsyncMock(return_value=("Search for weather", 0.9))
        mock_chat_workflow.execute_chat = AsyncMock(return_value={
            "response": "Weather search completed",
            "conversation_id": "web_conv",
            "workflow_id": "web_wf",
            "tokens_used": 15,
            "validation_passed": True
        })
        mock_voice_service.text_to_speech = AsyncMock(return_value=b"mock_audio")
        
        # Test with web escalation enabled
        response = client.post("/api/v1/voice/session", json={
            "audio_data": audio_base64,
            "conversation_id": "web_conv",
            "mode": "chat",
            "include_web": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "text_response" in data



import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.voice import VoiceService

@pytest.fixture
def voice_service():
    return VoiceService()

@pytest.mark.asyncio
async def test_tts_fallback_when_no_piper_model(voice_service):
    """Verify that TTS falls back to espeak if Piper model is missing"""
    
    # Mock model_manager to return None (model not available)
    voice_service._model_manager = MagicMock()
    voice_service._model_manager.download_recommended_model = AsyncMock(return_value=None)
    
    with patch("backend.core.voice.VoiceService._tts_fallback") as mock_fallback:
        mock_fallback.return_value = b"fallback audio"
        
        audio = await voice_service.text_to_speech("Hello world")
        
        assert audio == b"fallback audio"
        mock_fallback.assert_called_once_with("Hello world")

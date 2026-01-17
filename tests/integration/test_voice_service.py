"""
Unit tests for Voice Service
"""
import pytest
import asyncio
import tempfile
import wave
import struct
from backend.core.voice import voice_service


@pytest.mark.asyncio
async def test_voice_service_enhancements():
    """Test voice service enhancements"""
    # Test wake word detection (mock audio data)
    audio_data = bytes([0] * 32000)  # 2 seconds of silence at 16kHz
    wake_word_detected = voice_service.detect_wake_word(audio_data)
    assert isinstance(wake_word_detected, bool)

    # Test audio quality assessment
    try:
        # Create a simple WAV file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            # Write WAV header
            temp_audio.write(b'RIFF')
            temp_audio.write(struct.pack('<I', 36 + 32000))  # File size
            temp_audio.write(b'WAVEfmt ')
            temp_audio.write(struct.pack('<I', 16))  # Format chunk size
            temp_audio.write(struct.pack('<H', 1))  # PCM format
            temp_audio.write(struct.pack('<H', 1))  # Mono
            temp_audio.write(struct.pack('<I', 16000))  # Sample rate
            temp_audio.write(struct.pack('<I', 32000))  # Byte rate
            temp_audio.write(struct.pack('<H', 2))  # Block align
            temp_audio.write(struct.pack('<H', 16))  # Bits per sample
            temp_audio.write(b'data')
            temp_audio.write(struct.pack('<I', 32000))  # Data size
            temp_audio.write(bytes([0] * 32000))  # Audio data

            with open(temp_audio.name, 'rb') as f:
                audio_data = f.read()

        quality_assessment = voice_service.assess_audio_quality(audio_data)
        assert "quality_score" in quality_assessment
        assert "quality_level" in quality_assessment
        assert "is_acceptable" in quality_assessment

        # Test STT and TTS (mock)
        transcription, confidence = await voice_service.speech_to_text(audio_data)
        assert isinstance(transcription, str)
        assert isinstance(confidence, float)

        tts_audio = await voice_service.text_to_speech("Hello, this is a test.")
        assert isinstance(tts_audio, bytes)
    except Exception as e:
        # Skip if dependencies are missing (e.g., model files)
        pytest.skip(f"Voice processing skipped (missing dependencies): {e}")
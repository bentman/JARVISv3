"""Voice Service for JARVISv3
Ports JARVISv2 voice capabilities (Wake Word, STT, TTS).
"""
import os
import subprocess
import tempfile
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import wave
import audioop
import asyncio

# Optional openwakeword
try:
    from openwakeword.model import Model as OWWModel
    import openwakeword.utils
    HAS_OPENWAKEWORD = True
except ImportError:
    HAS_OPENWAKEWORD = False

from .config import settings

logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = settings.MODEL_PATH
OWW_MODELS_PATH = Path(MODEL_PATH) / "openwakeword"

class VoiceService:
    """
    Voice service for speech-to-text and text-to-speech conversion
    """

    def __init__(self):
        logger.info("Initializing VoiceService...")
        self.wake_word_model = self._init_wake_word()
        self.stt_model = None
        self.tts_model = None
        self.piper_voice = None
        self.models_path = Path(MODEL_PATH)
        self.models_path.mkdir(exist_ok=True)
        self._emotion_model = None
        self._model_manager = None
        logger.info("VoiceService initialized (lazy loading for executables)")

    @property
    def model_manager(self):
        if self._model_manager is None:
            from .model_manager import model_manager
            self._model_manager = model_manager
        return self._model_manager

    def _get_emotion_model(self):
        """Lazy load emotion detection model"""
        if self._emotion_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._emotion_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
        return self._emotion_model

    def detect_emotion(self, text: str) -> str:
        """Detect emotion from text using semantic similarity to emotion labels"""
        model = self._get_emotion_model()
        if not model:
            return "neutral"
            
        emotions = ["happy", "sad", "angry", "surprised", "neutral", "frustrated"]
        try:
            # Simple zero-shot classification via similarity
            text_emb = model.encode([text])
            emotion_embs = model.encode(emotions)
            
            # Calculate cosine similarities
            from numpy import dot
            from numpy.linalg import norm
            
            similarities = []
            for i in range(len(emotions)):
                sim = dot(text_emb[0], emotion_embs[i])/(norm(text_emb[0])*norm(emotion_embs[i]))
                similarities.append(sim)
                
            return emotions[similarities.index(max(similarities))]
        except Exception as e:
            logger.debug(f"Emotion detection failed: {e}")
            return "neutral"

    def _provision_oww_models(self):
        """Ensure default openwakeword models are present in persisted storage"""
        OWW_MODELS_PATH.mkdir(exist_ok=True, parents=True)
        
        # Default models to ensure presence (including required base models)
        models_to_ensure = ["alexa", "melspectrogram", "embedding_model"]
        
        # Check if we have at least one .onnx or .tflite for each requested model
        for model in models_to_ensure:
            has_onnx = list(OWW_MODELS_PATH.glob(f"{model}*.onnx"))
            has_tflite = list(OWW_MODELS_PATH.glob(f"{model}*.tflite"))
            
            if not has_onnx and not has_tflite:
                logger.info(f"Downloading openwakeword model: {model}...")
                try:
                    import openwakeword.utils
                    openwakeword.utils.download_models(
                        model_names=[model], 
                        target_directory=str(OWW_MODELS_PATH)
                    )
                    logger.info(f"Successfully provisioned {model}")
                except Exception as e:
                    logger.error(f"Failed to download openwakeword model {model}: {e}")

    def _init_wake_word(self):
        """Initialize openwakeword model if available"""
        if HAS_OPENWAKEWORD:
            try:
                # Provision models to persisted path
                self._provision_oww_models()
                
                # Load models from custom path (prefer ONNX)
                model_files = [str(p) for p in OWW_MODELS_PATH.glob("*.onnx")]
                inference_framework = "onnx"
                if not model_files:
                     # Fallback to TFLite if any exist
                     model_files = [str(p) for p in OWW_MODELS_PATH.glob("*.tflite")]
                     inference_framework = "tflite"
                
                if not model_files:
                     logger.warning("No openwakeword models found in persisted path")
                     return None
                     
                # Get paths for required base models
                melspec_path = next(OWW_MODELS_PATH.glob("melspectrogram*.onnx"), None)
                embedding_path = next(OWW_MODELS_PATH.glob("embedding_model*.onnx"), None)
                
                return OWWModel(
                    wakeword_models=model_files, 
                    inference_framework=inference_framework,
                    melspec_model_path=str(melspec_path) if melspec_path else None,
                    embedding_model_path=str(embedding_path) if embedding_path else None
                )
            except Exception as e:
                logger.warning(f"Failed to init openwakeword: {e}")
        return None

    def _find_whisper_model(self) -> str:
        """Find Whisper model executable"""
        possible_paths = [
            "/usr/local/bin/whisper",  # Docker build location
            "./whisper.cpp/whisper",
            "./backend/whisper.cpp/whisper",
            "whisper",
            "whisper-cpp"
        ]

        from shutil import which
        for path in possible_paths:
            found_path = which(path)
            if found_path:
                return str(found_path)
            if Path(path).exists():
                return str(Path(path).absolute())

        # If not found, return default (will likely fail later but handles 'not found')
        return "whisper"

    def _find_piper_model(self) -> str:
        """Find Piper TTS executable"""
        possible_paths = [
            "/usr/local/bin/piper",  # Docker build location
            "./piper/src/piper",
            "./backend/piper/src/piper",
            "piper"
        ]

        from shutil import which
        for path in possible_paths:
            found_path = which(path)
            if found_path:
                return str(found_path)
            if Path(path).exists():
                return str(Path(path).absolute())

        return "piper"

    def _find_whisper_weights(self) -> str:
        """Find Whisper weights file"""
        candidates = ["ggml-base.en.bin", "ggml-base.bin", "ggml-tiny.en.bin"]
        for name in candidates:
            p = self.models_path / name
            if p.exists():
                return str(p)
        # Fallback
        found = list(self.models_path.glob("ggml-*.bin"))
        if found:
            return str(found[0])
        return str(self.models_path / "ggml-base.en.bin")

    def detect_wake_word(self, audio_data: bytes) -> bool:
        """Detect wake word in audio stream"""
        if self.wake_word_model:
            try:
                # Basic conversion assuming 16kHz mono int16 PCM input in audio_data
                # OpenWakeWord expects float32 array
                pcm = np.frombuffer(audio_data, dtype=np.int16)
                # Normalize
                audio = (pcm.astype(np.float32) / 32768.0).reshape(1, -1)

                scores = self.wake_word_model.predict(audio)
                for _kw, score in scores.items():
                    val = float(np.max(score)) if isinstance(score, (list, tuple, np.ndarray)) else float(score)
                    if val >= 0.5:
                        return True
            except Exception as e:
                logger.debug(f"Wake word detection error: {e}")
                pass
        return False

    def assess_audio_quality(self, audio_data: bytes) -> Dict[str, Any]:
        """Assess audio quality and provide feedback"""
        try:
            # Parse WAV header to get audio properties
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            with wave.open(temp_audio_path, 'rb') as wav_file:
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()

            os.unlink(temp_audio_path)

            # Calculate RMS (volume level)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            rms = audioop.rms(audio_array.tobytes(), sample_width)

            # Quality assessment
            quality_score = 0
            feedback = []

            # Sample rate check
            if frame_rate >= 16000:
                quality_score += 30
            else:
                feedback.append("Low sample rate - use 16kHz or higher")

            # Volume check
            if rms > 1000:  # Reasonable volume threshold
                quality_score += 30
            else:
                feedback.append("Low volume - speak louder or move closer to microphone")

            # Duration check
            duration = n_frames / frame_rate
            if duration >= 1.0:  # At least 1 second of audio
                quality_score += 20
            else:
                feedback.append("Audio too short - speak for at least 1 second")

            # Channel check
            if n_channels == 1:  # Mono is preferred for speech
                quality_score += 20

            # Overall quality assessment
            if quality_score >= 80:
                quality_level = "excellent"
            elif quality_score >= 60:
                quality_level = "good"
            elif quality_score >= 40:
                quality_level = "fair"
            else:
                quality_level = "poor"

            return {
                "quality_score": quality_score,
                "quality_level": quality_level,
                "sample_rate": frame_rate,
                "volume_rms": rms,
                "duration": duration,
                "channels": n_channels,
                "feedback": feedback,
                "is_acceptable": quality_score >= 50
            }

        except Exception as e:
            logger.error(f"Audio quality assessment error: {e}")
            return {
                "quality_score": 0,
                "quality_level": "unknown",
                "feedback": ["Unable to assess audio quality"],
                "is_acceptable": False
            }

    async def speech_to_text(self, audio_data: bytes) -> Tuple[str, float]:
        """Convert speech to text using Whisper"""
        if not self.stt_model:
            self.stt_model = self._find_whisper_model()
            logger.info(f"Whisper executable: {self.stt_model}")

        # Ensure model is downloaded
        weights = await self.model_manager.download_recommended_model("stt")
        if not weights or not weights.exists():
            raise Exception("Voice STT unavailable - Whisper model weights not found. Use Docker or install locally.")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_output:
                temp_output_path = temp_output.name

            cmd = [
                self.stt_model,
                "-m", str(weights),
                "-f", temp_audio_path,
                "-otxt", temp_output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r') as f:
                    text = f.read().strip()
            else:
                text = ""

            os.unlink(temp_audio_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

            return text, 0.9 # Mock confidence for now

        except Exception as e:
            logger.error(f"STT Error: {e}")
            raise

    async def text_to_speech(self, text: str, speed: float = 1.0, pitch: float = 0.0) -> bytes:
        """
        Convert text to speech using Piper with prosody control.
        :param speed: 0.5 to 2.0
        :param pitch: -1.0 to 1.0 (if supported)
        """
        if not self.tts_model:
            self.tts_model = self._find_piper_model()
            logger.info(f"Piper executable: {self.tts_model}")

        # Ensure models are downloaded
        voice_model_path = await self.model_manager.download_recommended_model("tts")
        if not voice_model_path or not voice_model_path.exists():
            raise Exception("Voice TTS unavailable - Piper voice model not found. Use Docker or install locally.")

        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as temp_input:
                temp_input.write(text)
                temp_input_path = temp_input.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
                temp_output_path = temp_output.name

            cmd = [
                self.tts_model,
                "--model", str(voice_model_path),
                "--length_scale", str(1.0 / speed), # Piper uses length_scale (inverse of speed)
                "--output_file", temp_output_path
            ]

            # Use asyncio for non-blocking execution
            with open(temp_input_path, 'r', encoding='utf-8') as input_file:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=input_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'rb') as f:
                    audio = f.read()
            else:
                audio = b""

            # Cleanup
            for p in [temp_input_path, temp_output_path]:
                if os.path.exists(p):
                    os.unlink(p)

            if not audio:
                raise RuntimeError("Piper generated empty audio")

            return audio

        except Exception as e:
            logger.error(f"TTS Error: {e}")
            raise

    async def _tts_fallback(self, text: str) -> bytes:
        """Fallback to espeak-ng or espeak"""
        from shutil import which
        espeak = which("espeak-ng") or which("espeak")
        if not espeak:
            logger.error("No TTS engine available (Piper failed and espeak not found)")
            return b""
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
                temp_output_path = temp_output.name
            
            v = settings.TTS_PREFERRED_VOICE
            cmd = [espeak] + (["-v", v] if v else []) + ["-w", temp_output_path, text]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'rb') as f:
                    audio = f.read()
                os.unlink(temp_output_path)
                return audio
            return b""
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            return b""

# Global instance
voice_service = VoiceService()

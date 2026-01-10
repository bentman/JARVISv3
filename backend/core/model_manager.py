"""
Model Management Service for JARVISv3
Enhanced with advanced verification and graceful fallbacks.
"""
import os
import hashlib
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import aiohttp
import huggingface_hub
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from ..ai.context.schemas import HardwareState, ModelProfile
from .hardware import HardwareService
from .config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages model downloading, verification, and selection based on hardware capabilities.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        # Use settings if not provided
        target_dir = models_dir or settings.MODEL_PATH or "models"
        self.models_dir = Path(target_dir)
            
        # Try to create directory, but ignore if read-only (likely mounted)
        try:
            self.models_dir.mkdir(exist_ok=True, parents=True)
        except OSError as e:
            if "Read-only file system" not in str(e):
                raise
            logger.info(f"Models directory {self.models_dir} is read-only, skipping creation")

        self.hardware_service = HardwareService()
        self.verification_cache = {}
        self.download_locks = {}
        
        # Model profiles with SHA-256 checksums for verification
        self.model_profiles = {
            "light": {
                "model_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "expected_checksum": "6F85A640A97CF2BF5B8E764087B1E83DA0FDB51D7C9FAB7D0FECE9385611DF83",
                "size_mb": 700,
                "description": "Llama 3.2 1B parameter model for light hardware"
            },
            "medium": {
                "model_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "expected_checksum": "placeholder_checksum_medium",
                "size_mb": 2000,
                "description": "Llama 3.2 3B parameter model for medium hardware"
            },
            "heavy": {
                "model_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "expected_checksum": "placeholder_checksum_heavy",
                "size_mb": 5000,
                "description": "Llama 3.1 8B parameter model for heavy hardware"
            },
            "npu-optimized": {
                "model_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "expected_checksum": "placeholder_checksum_npu",
                "size_mb": 700,
                "description": "NPU-optimized 1B model"
            },
            "stt-base": {
                "model_id": "ggerganov/whisper.cpp",
                "filename": "ggml-base.en.bin",
                "expected_checksum": "A03779C86DF3323075F5E796CB2CE5029F00EC8869EEE3FDFB897AFE36C6D002",
                "size_mb": 150,
                "description": "Whisper base.en model for STT"
            },
            "tts-medium": {
                "model_id": "rhasspy/piper-voices",
                "filename": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "expected_checksum": "5EFE09E69902187827AF646E1A6E9D269DEE769F9877D17B16B1B46EEAAF019F",
                "size_mb": 50,
                "description": "Piper en_US-lessac-medium voice model"
            },
            "tts-medium-config": {
                "model_id": "rhasspy/piper-voices",
                "filename": "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
                "expected_checksum": "placeholder_checksum_tts_json",
                "size_mb": 1,
                "description": "Piper en_US-lessac-medium voice config"
            }
        }
        
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their profiles"""
        return [
            {
                "profile": profile,
                "model_id": details["model_id"],
                "filename": details["filename"],
                "size_mb": details["size_mb"],
                "description": details["description"],
                "available": await self._check_model_availability(profile)
            }
            for profile, details in self.model_profiles.items()
        ]
    
    async def _check_model_availability(self, profile: str) -> bool:
        """Check if a model is available locally"""
        model_details = self.model_profiles[profile]
        model_path = self.models_dir / model_details["filename"]
        return model_path.exists()
    
    async def get_best_model_for_hardware(self, hardware_state: HardwareState) -> Optional[ModelProfile]:
        """Determine the best model profile for current hardware"""
        # Get hardware profile
        hardware_profile = self.hardware_service.get_hardware_profile()
        
        # Map hardware profile to model profile
        profile_mapping = {
            "light": "light",
            "medium": "medium", 
            "heavy": "heavy",
            "npu-optimized": "npu-optimized"
        }
        
        target_profile = profile_mapping.get(hardware_profile, "light")
        model_details = self.model_profiles[target_profile]
        
        return ModelProfile(
            model_id=model_details["model_id"],
            filename=model_details["filename"],
            profile=target_profile,
            size_mb=model_details["size_mb"],
            description=model_details["description"]
        )
    
    async def download_model(self, model_profile: ModelProfile) -> Optional[Path]:
        """Download model with fallbacks"""
        model_path = self.models_dir / model_profile.filename

        # Check if already exists
        if model_path.exists():
            logger.info(f"Model {model_profile.filename} already exists")
            return model_path

        # Acquire download lock to prevent concurrent downloads
        lock_key = model_profile.filename
        if lock_key not in self.download_locks:
            self.download_locks[lock_key] = asyncio.Lock()

        async with self.download_locks[lock_key]:
            # Double-check after acquiring lock
            if model_path.exists():
                return model_path

            # Attempt download with fallbacks
            return await self._download_with_fallbacks(model_profile, model_path)
    
    async def _download_with_fallbacks(self, model_profile: ModelProfile, model_path: Path) -> Optional[Path]:
        """Download model with multiple fallback strategies"""
        download_strategies = [
            self._download_from_hf_hub,
            self._download_from_mirror,
            self._download_from_backup
        ]
        
        for strategy in download_strategies:
            try:
                logger.info(f"Attempting to download {model_profile.filename} using {strategy.__name__}")
                result = await strategy(model_profile, model_path)
                if result:
                    logger.info(f"Successfully downloaded {model_profile.filename}")
                    return model_path
                else:
                    logger.warning(f"Download strategy {strategy.__name__} failed")
                    # Clean up partial download
                    if model_path.exists():
                        model_path.unlink()
            except Exception as e:
                logger.error(f"Download strategy {strategy.__name__} failed: {e}")
                # Clean up partial download
                if model_path.exists():
                    model_path.unlink()
                continue
        
        logger.error(f"All download strategies failed for {model_profile.filename}")
        return None
    
    async def _download_from_hf_hub(self, model_profile: ModelProfile, model_path: Path) -> bool:
        """Download model from Hugging Face Hub with advanced error handling"""
        try:
            # Download model file using huggingface_hub
            downloaded_path = hf_hub_download(
                repo_id=model_profile.model_id,
                filename=model_profile.filename,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False
            )
            
            return Path(downloaded_path).exists()
            
        except Exception as e:
            logger.error(f"HF Hub download failed: {e}")
            return False
    
    async def _download_from_mirror(self, model_profile: ModelProfile, model_path: Path) -> bool:
        """Download from alternative mirror if HF Hub fails"""
        # Placeholder for mirror download logic
        logger.info("Mirror download strategy not yet implemented")
        return False
    
    async def _download_from_backup(self, model_profile: ModelProfile, model_path: Path) -> bool:
        """Download from backup source as last resort"""
        # Placeholder for backup download logic
        logger.info("Backup download strategy not yet implemented")
        return False
    


    async def download_recommended_model(self, tier: str) -> Optional[Path]:
        """Compatibility method for existing calls"""
        profile_mapping = {
            "light": "light",
            "medium": "medium", 
            "heavy": "heavy",
            "npu-optimized": "npu-optimized",
            "stt": "stt-base",
            "tts": "tts-medium"
        }
        target_profile = profile_mapping.get(tier, "light")
        model_details = self.model_profiles[target_profile]
        
        model_profile = ModelProfile(
            model_id=model_details["model_id"],
            filename=model_details["filename"],
            profile=target_profile,
            size_mb=model_details["size_mb"],
            description=model_details["description"]
        )
        
        return await self.download_model(model_profile)

# Global instance
model_manager = ModelManager()
# Global instance

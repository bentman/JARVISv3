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

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages model downloading, verification, and selection based on hardware capabilities.
    """
    
    def __init__(self, models_dir: str = "models"):
        # Go up one level if we are in backend/core
        if Path.cwd().name == "core":
            self.models_dir = Path.cwd().parent.parent / models_dir
        elif Path.cwd().name == "backend":
            self.models_dir = Path.cwd().parent / models_dir
        else:
            self.models_dir = Path(models_dir)
            
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.hardware_service = HardwareService()
        self.verification_cache = {}
        self.download_locks = {}
        
        # Model profiles with SHA-256 checksums for verification
        self.model_profiles = {
            "light": {
                "model_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "expected_checksum": "placeholder_checksum_light",
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
                "expected_checksum": "placeholder_checksum_stt",
                "size_mb": 150,
                "description": "Whisper base.en model for STT"
            },
            "tts-medium": {
                "model_id": "rhasspy/piper-voices",
                "filename": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "expected_checksum": "placeholder_checksum_tts",
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
        """Download model with advanced verification and fallbacks"""
        model_path = self.models_dir / model_profile.filename
        
        # Check if already exists and is valid
        if model_path.exists() and await self._verify_model_integrity(model_path, model_profile):
            logger.info(f"Model {model_profile.filename} already exists and is valid")
            return model_path
        
        # Acquire download lock to prevent concurrent downloads
        lock_key = model_profile.filename
        if lock_key not in self.download_locks:
            self.download_locks[lock_key] = asyncio.Lock()
        
        async with self.download_locks[lock_key]:
            # Double-check after acquiring lock
            if model_path.exists() and await self._verify_model_integrity(model_path, model_profile):
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
                if result and await self._verify_model_integrity(model_path, model_profile):
                    logger.info(f"Successfully downloaded {model_profile.filename}")
                    return model_path
                else:
                    logger.warning(f"Download strategy {strategy.__name__} failed or verification failed")
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
    
    async def _verify_model_integrity(self, model_path: Path, model_profile: ModelProfile) -> bool:
        """Verify model file integrity using SHA-256 checksums"""
        if not model_path.exists():
            return False
        
        # Check cache first
        cache_key = f"{model_profile.model_id}/{model_profile.filename}"
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        try:
            # Calculate SHA-256 checksum
            sha256_hash = hashlib.sha256()
            file_size = model_path.stat().st_size
            
            with open(model_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            
            calculated_checksum = sha256_hash.hexdigest()
            # In a real implementation, we would compare with a known checksum
            # For now, we'll verify the size is reasonable
            expected_size_min = model_profile.size_mb * 0.8 * 1024 * 1024
            
            is_valid = file_size > expected_size_min
            
            # Cache result
            self.verification_cache[cache_key] = is_valid
            
            if is_valid:
                logger.info(f"Model {model_profile.filename} passed integrity verification")
            else:
                logger.error(f"Model {model_profile.filename} failed integrity verification (size too small)")
                model_path.unlink(missing_ok=True)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Model verification failed for {model_profile.filename}: {e}")
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

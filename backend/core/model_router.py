"""
Model Router Service for JARVISv3
Routes requests to appropriate models and providers.
"""
import os
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, AsyncIterable, Tuple
from pathlib import Path
from pydantic import BaseModel

from .hardware import HardwareService
from .model_manager import model_manager
from .node_registry import node_registry
from .model_providers.base import ModelProvider, ModelInferenceResult
from .model_providers.llama_cpp import LlamaCppProvider
from .model_providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

class ModelRouter:
    """
    Routes requests to appropriate models based on hardware profile, availability, and task type.
    Supports multiple providers (llama.cpp, Ollama).
    """
    
    def __init__(self):
        self.models_path = Path(MODEL_PATH)
        self.models_path.mkdir(exist_ok=True)
        self.hardware_service = HardwareService()
        
        # Initialize providers
        self.providers: Dict[str, ModelProvider] = {
            "llama_cpp": LlamaCppProvider(self.models_path),
            "ollama": OllamaProvider(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        }
        
    async def get_available_providers(self) -> List[str]:
        """Return list of names of providers that are currently available"""
        available = []
        for name, provider in self.providers.items():
            if await provider.is_available():
                available.append(name)
        return available

    async def select_model_and_provider(self, task_type: str = "chat", required_tier: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
        """
        Select appropriate model and provider based on hardware and availability.
        Can also suggest a remote node.
        Returns (model_name, provider_name, remote_node_id)
        """
        hardware_state = await self.hardware_service.get_hardware_state()
        
        # Determine target profile if not explicitly required
        profile = required_tier
        if not profile:
            profile = "light"
        if "gpu" in hardware_state.available_tiers and hardware_state.gpu_usage < 90:
             if hardware_state.memory_available_gb > 16:
                 profile = "heavy"
             elif hardware_state.memory_available_gb > 8:
                 profile = "medium"
        elif hardware_state.memory_available_gb > 16:
             profile = "medium"
             
        if "npu" in hardware_state.available_tiers:
            profile = "npu-optimized"

        # Get candidate providers that support the required task (regardless of current availability)
        candidate_providers = []
        for name, provider in self.providers.items():
            models = provider.get_supported_models()
            if any(task_type in models.get(p, {}) for p in models):
                candidate_providers.append(name)

        # 1. Prefer Ollama if it has models for this task and is available
        if "ollama" in candidate_providers:
            ollama_provider = self.providers["ollama"]
            if await ollama_provider.is_available():
                ollama_models = ollama_provider.get_supported_models()
                if profile in ollama_models and task_type in ollama_models[profile]:
                    return ollama_models[profile][task_type], "ollama", None

        # 2. Fallback to llama_cpp
        if "llama_cpp" in candidate_providers:
            llama_models = self.providers["llama_cpp"].get_supported_models()
            if profile in llama_models and task_type in llama_models[profile]:
                return llama_models[profile][task_type], "llama_cpp", None

            # Fallback within llama_cpp for different tiers
            for p in ["medium", "light"]:
                if p in llama_models and task_type in llama_models[p]:
                    return llama_models[p][task_type], "llama_cpp", None

        # 3. Check for Remote Nodes if local tiers are insufficient
        best_remote = await node_registry.find_best_node_for_task(profile)
        if best_remote and best_remote.node_id != node_registry.local_node_id:
             # We found a better remote node!
             # For now we'll just return its ID and let the caller handle the proxy
             return "remote_model", "remote_provider", best_remote.node_id

        # 4. Last resort fallback (models may need downloading)
        recommended = model_manager.model_profiles.get(profile, {}).get("filename")
        model_name = recommended if isinstance(recommended, str) else "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        return model_name, "llama_cpp", None

    async def generate_response(self, prompt: str, task_type: str = "chat", **kwargs) -> ModelInferenceResult:
        """Generate a response using the best available model/provider"""
        required_tier = kwargs.get("required_tier")
        model_name, provider_name, remote_node_id = await self.select_model_and_provider(task_type, required_tier)
        
        # Handle remote delegation if suggested by router
        if remote_node_id:
             logger.info(f"Router suggested remote delegation to {remote_node_id}")
             # This is a bit of a circular dependency if we use WorkflowEngine here,
             # so in a real system we'd use a shared RPC client or similar.
             # For now, we'll mark the result as proxied.
             raise Exception(f"Task should be delegated to remote node {remote_node_id}")

        provider = self.providers.get(provider_name)
        
        if not provider:
            raise Exception(f"Provider {provider_name} not found")

        # Special handling for llama_cpp downloads if needed
        if provider_name == "llama_cpp" and isinstance(provider, LlamaCppProvider):
            model_path = Path(provider.get_model_path(model_name))
            if not model_path.exists():
                logger.info(f"Model {model_name} not found for llama_cpp. Attempting download.")
                hardware_profile = self.hardware_service.get_hardware_profile()
                await model_manager.download_recommended_model(hardware_profile)
        
        return await provider.generate_response(prompt, model_name, **kwargs)

    def generate_response_stream(self, prompt: str, task_type: str = "chat", **kwargs) -> AsyncIterable[str]:
        """Generate a streaming response using the best available model/provider"""
        # Since this returns an AsyncIterable, we need to handle the initial selection inside
        # or use an async generator
        return self._stream_wrapper(prompt, task_type, **kwargs)

    async def _stream_wrapper(self, prompt: str, task_type: str, **kwargs) -> AsyncIterable[str]:
        """Internal wrapper to handle async selection for streaming"""
        try:
            required_tier = kwargs.get("required_tier")
            model_name, provider_name, remote_node_id = await self.select_model_and_provider(task_type, required_tier)
            
            if remote_node_id:
                yield f"[Info] Delegating streaming task to remote node {remote_node_id}..."
                # In a real system, we'd open a stream to the remote node here
                return

            provider = self.providers.get(provider_name)
            
            if not provider:
                yield f"[Error] Provider {provider_name} not found"
                return

            if provider_name == "llama_cpp" and isinstance(provider, LlamaCppProvider):
                model_path = Path(provider.get_model_path(model_name))
                if not model_path.exists():
                    hardware_profile = self.hardware_service.get_hardware_profile()
                    await model_manager.download_recommended_model(hardware_profile)
            
            async for chunk in provider.generate_response_stream(prompt, model_name, **kwargs):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming model router: {e}")
            yield f"[Error] {str(e)}"

# Global instance
model_router = ModelRouter()

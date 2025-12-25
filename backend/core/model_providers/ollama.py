"""
Ollama Model Provider for JARVISv3
"""
import asyncio
import time
import logging
import json
from typing import AsyncIterable, Dict, Any, Optional, List
import httpx
from .base import ModelProvider, ModelInferenceResult

logger = logging.getLogger(__name__)

class OllamaProvider(ModelProvider):
    """
    Model provider using Ollama API for model execution
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0))
        
    async def generate_response(self, prompt: str, model_name: str, **kwargs) -> ModelInferenceResult:
        """Generate a response from the model"""
        start_time = time.time()
        
        try:
            response = await self.client.post("/api/generate", json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": kwargs.get("max_tokens", 256),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9)
                }
            })
            
            response.raise_for_status()
            data = response.json()
            
            execution_time = time.time() - start_time
            
            return ModelInferenceResult(
                response=data.get("response", ""),
                tokens_used=data.get("eval_count", 0),
                execution_time=execution_time,
                model_name=model_name,
                provider="ollama"
            )
            
        except Exception as e:
            logger.error(f"Error during Ollama inference: {str(e)}")
            raise

    async def generate_response_stream(self, prompt: str, model_name: str, **kwargs) -> AsyncIterable[str]:
        """Generate a streaming response from the model"""
        try:
            async with self.client.stream("POST", "/api/generate", json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": kwargs.get("max_tokens", 256),
                    "temperature": kwargs.get("temperature", 0.7)
                }
            }) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except Exception as e:
            logger.error(f"Error during Ollama streaming: {str(e)}")
            yield f"[Error] {str(e)}"

    async def is_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def get_supported_models(self) -> Dict[str, Any]:
        """
        Return recommended Ollama models for each tier.
        In a real scenario, this could query Ollama's locally pulled models.
        """
        # Recommended Ollama models
        return {
            "light": {"chat": "llama3.2:1b"},
            "medium": {"chat": "llama3.2:3b"},
            "heavy": {"chat": "llama3.1:8b"},
            "npu-optimized": {"chat": "llama3.2:1b"}
        }

    async def get_installed_models(self) -> List[str]:
        """Get list of models actually installed in Ollama"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

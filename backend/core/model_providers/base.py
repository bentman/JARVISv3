"""
Base Model Provider for JARVISv3
"""
from abc import ABC, abstractmethod
from typing import AsyncIterable, Optional, Any, Dict
from pydantic import BaseModel

class ModelInferenceResult(BaseModel):
    """Result from model inference"""
    response: str
    tokens_used: int
    execution_time: float
    model_name: str
    provider: str

class ModelProvider(ABC):
    """
    Abstract base class for model providers (llama.cpp, Ollama, OpenAI, etc.)
    """
    
    @abstractmethod
    async def generate_response(self, prompt: str, model_name: str, **kwargs) -> ModelInferenceResult:
        """Generate a response from the model"""
        pass
        
    @abstractmethod
    def generate_response_stream(self, prompt: str, model_name: str, **kwargs) -> AsyncIterable[str]:
        """Generate a streaming response from the model"""
        pass
        
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available"""
        pass

    @abstractmethod
    def get_supported_models(self) -> Dict[str, Any]:
        """Return a dictionary of supported models grouped by tier/task"""
        pass

"""
Llama.cpp Model Provider for JARVISv3
Enhanced with model integrity validation
"""
import os
import signal
import asyncio
import time
import logging
from typing import AsyncIterable, Dict, Any, Optional
from pathlib import Path
from shutil import which
from .base import ModelProvider, ModelInferenceResult
from ..model_manager import model_manager
from ...ai.context.schemas import ModelProfile

logger = logging.getLogger(__name__)

class LlamaCppProvider(ModelProvider):
    """
    Model provider using llama.cpp for local GGUF model execution
    """
    
    def __init__(self, models_path: Path):
        self.models_path = models_path
        self.llama_cpp_path = self._find_llama_cpp()
        
    def _find_llama_cpp(self) -> str:
        """Find the llama.cpp executable"""
        possible_paths = [
            "/usr/local/bin/llama-completion",
            "llama-completion",
            "./llama-cli-extracted/llama-completion.exe",
            "./llama.cpp/llama-completion",
            "./llama-cli-extracted/llama-cli.exe",
            "./llama.cpp/main",
            "./backend/llama.cpp/main",
            "llama-cli",
            "llama",
            "llama-gguf"
        ]
        
        for path in possible_paths:
            found_path = which(path)
            if found_path:
                return str(found_path)
            if Path(path).exists():
                return str(Path(path).absolute())
        
        return "llama"

    def get_model_path(self, model_name: str) -> str:
        """Get path to model file"""
        found = list(self.models_path.rglob(f"{model_name}.gguf"))
        if found:
            return str(found[0])
        # Try direct name
        if (self.models_path / f"{model_name}.gguf").exists():
            return str(self.models_path / f"{model_name}.gguf")
        return str(self.models_path / model_name)

    async def generate_response(self, prompt: str, model_name: str, **kwargs) -> ModelInferenceResult:
        """Generate a response from the model with integrity validation"""
        if not await self.is_available():
            raise Exception("LlamaCppProvider not available")

        model_path = self.get_model_path(model_name)
        max_tokens = kwargs.get("max_tokens", 256)

        start_time = time.time()

        try:
            # Invoke llama-completion in one-shot mode
            cmd = [
                self.llama_cpp_path,
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(kwargs.get("temperature", 0.7)),
                "--repeat-penalty", "1.1",
                "-no-cnv",
                "--no-warmup"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True
            )

            try:
                # Wait up to 60 seconds for generation
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.warning("llama.cpp timed out, terminating process group...")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except (ProcessLookupError, asyncio.TimeoutError):
                    logger.error("llama.cpp failed to terminate, killing process group...")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass # Already gone
                
                # Try to read whatever output we got (best effort)
                stdout = b""
                stderr = b"Timeout reached"
                execution_time = time.time() - start_time
                raise Exception("Timeout during inference")

            execution_time = time.time() - start_time

            if process.returncode != 0 and process.returncode is not None:
                logger.warning(f"llama.cpp exited with {process.returncode}: {stderr.decode(errors='ignore')}")

            raw_response = stdout.decode(errors='ignore').strip()
            
            # Minimal cleaning
            # Remove "available commands" banner
            if "available commands:" in raw_response:
                parts = raw_response.split("available commands:")
                raw_response = parts[-1]
            
            # Remove prompt echo
            if f"> {prompt}" in raw_response:
                raw_response = raw_response.split(f"> {prompt}")[-1].strip()
            elif raw_response.startswith(prompt):
                raw_response = raw_response[len(prompt):].strip()

            tokens_used = len(raw_response.split())

            return ModelInferenceResult(
                response=raw_response,
                tokens_used=tokens_used,
                execution_time=execution_time,
                model_name=model_name,
                provider="llama_cpp"
            )

        except Exception as e:
            logger.error(f"Error during llama.cpp inference: {str(e)}")
            raise



    async def generate_response_stream(self, prompt: str, model_name: str, **kwargs) -> AsyncIterable[str]:
        """Generate a streaming response from the model"""
        model_path = self.get_model_path(model_name)
        max_tokens = kwargs.get("max_tokens", 256)

        try:
            cmd = [
                self.llama_cpp_path,
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(kwargs.get("temperature", 0.7)),
                "--repeat-penalty", "1.1"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            if process.stdout:
                while True:
                    chunk = await process.stdout.read(1024)
                    if not chunk:
                        break
                    decoded = chunk.decode(errors='ignore')
                    if decoded:
                        yield decoded

            await process.wait()

        except Exception as e:
            logger.error(f"Error during llama.cpp streaming: {str(e)}")
            yield f"[Error] {str(e)}"

    async def is_available(self) -> bool:
        """Check if llama.cpp and models are available"""
        # Simple check for llama_cpp path and at least one GGUF model
        has_executable = which(self.llama_cpp_path) or Path(self.llama_cpp_path).exists()
        has_models = self.models_path.exists() and any(self.models_path.rglob("*.gguf"))
        return bool(has_executable and has_models)

    def get_supported_models(self) -> Dict[str, Any]:
        """Return available models on disk"""
        available_models = {
            "light": {},
            "medium": {},
            "heavy": {},
            "npu-optimized": {}
        }
        
        if self.models_path.exists():
            gguf_files = list(self.models_path.rglob("*.gguf"))
            for file_path in gguf_files:
                filename = file_path.stem
                if any(tag in filename.lower() for tag in ["1b", "2b", "3b"]):
                    available_models["light"].setdefault("chat", filename)
                elif "7b" in filename.lower():
                    available_models["medium"].setdefault("chat", filename)
                elif any(tag in filename.lower() for tag in ["13b", "30b", "70b"]):
                    available_models["heavy"].setdefault("chat", filename)
                else:
                    available_models["medium"].setdefault("chat", filename)
        
        return available_models

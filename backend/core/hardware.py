"""
Hardware Detection Service for JARVISv3
Enhanced resource-aware execution with dynamic memory management and graceful degradation.
"""
import psutil
import GPUtil
import platform
import logging
import threading
import time
from typing import Dict, Optional, List, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
from ..ai.context.schemas import HardwareState

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Hardware acceleration types"""
    CPU_ONLY = "cpu_only"
    GPU_GENERAL = "gpu_general"  # AMD, Intel Arc
    GPU_CUDA = "gpu_cuda"        # NVIDIA with CUDA
    NPU_APPLE = "npu_apple"      # Apple Silicon M-series
    NPU_QUALCOMM = "npu_qualcomm" # Qualcomm ARM64
    NPU_INTEL = "npu_intel"      # Intel NPU


@dataclass
class MemoryAllocation:
    """Tracks memory allocation for a specific model/provider"""
    model_name: str
    provider: str
    allocated_mb: float
    max_mb: float
    timestamp: float


class ResourceManager:
    """Manages dynamic resource allocation and monitoring"""

    def __init__(self):
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.lock = threading.Lock()
        self.memory_pressure_threshold = 0.85  # 85% memory usage
        self.degradation_callbacks: List[Callable] = []

    def allocate_memory(self, model_name: str, provider: str, requested_mb: float) -> bool:
        """Attempt to allocate GPU memory"""
        with self.lock:
            # Check current memory usage
            try:
                mem = psutil.virtual_memory()
                available_mb = (mem.available / (1024 * 1024))

                if available_mb < requested_mb:
                    logger.warning(f"Insufficient memory for {model_name}: requested {requested_mb}MB, available {available_mb}MB")
                    return False

                # Check if we're approaching memory pressure
                memory_pressure = (mem.total - mem.available) / mem.total
                if memory_pressure > self.memory_pressure_threshold:
                    logger.warning(f"High memory pressure ({memory_pressure:.1%}) detected")
                    self._trigger_degradation()

                # Record allocation
                allocation = MemoryAllocation(
                    model_name=model_name,
                    provider=provider,
                    allocated_mb=requested_mb,
                    max_mb=requested_mb,
                    timestamp=time.time()
                )
                self.allocations[f"{provider}_{model_name}"] = allocation

                logger.info(f"Allocated {requested_mb}MB for {model_name} on {provider}")
                return True

            except Exception as e:
                logger.error(f"Memory allocation failed: {e}")
                return False

    def deallocate_memory(self, model_name: str, provider: str):
        """Deallocate memory for a model"""
        with self.lock:
            key = f"{provider}_{model_name}"
            if key in self.allocations:
                allocation = self.allocations.pop(key)
                logger.info(f"Deallocated {allocation.allocated_mb}MB for {model_name} on {provider}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage by provider"""
        usage = {}
        with self.lock:
            for key, allocation in self.allocations.items():
                provider = allocation.provider
                if provider not in usage:
                    usage[provider] = 0
                usage[provider] += allocation.allocated_mb
        return usage

    def check_resource_exhaustion(self) -> Optional[str]:
        """Check for resource exhaustion and return degradation recommendation"""
        try:
            mem = psutil.virtual_memory()
            memory_pressure = (mem.total - mem.available) / mem.total

            if memory_pressure > 0.95:  # 95% memory usage
                return "critical_memory_exhaustion"

            if memory_pressure > 0.90:  # 90% memory usage
                return "high_memory_pressure"

            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 95:
                return "cpu_exhaustion"

            return None

        except Exception as e:
            logger.error(f"Resource exhaustion check failed: {e}")
            return None

    def _trigger_degradation(self):
        """Trigger degradation callbacks"""
        for callback in self.degradation_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Degradation callback failed: {e}")

    def register_degradation_callback(self, callback: Callable):
        """Register a callback for resource degradation events"""
        self.degradation_callbacks.append(callback)

class HardwareService:
    """
    Enhanced hardware detection and resource management for JARVISv3.
    Provides dynamic memory allocation, graceful degradation, and hardware-aware optimization.
    """

    def __init__(self):
        self._cpu_info = {}
        self._gpu_info = {}
        self._memory_info = {}
        self._accel_providers = []
        self._hardware_type = HardwareType.CPU_ONLY

        # Resource management
        self.resource_manager = ResourceManager()
        self.degradation_active = False

        # Hardware-specific optimizations
        self._cuda_available = False
        self._npu_detected = False
        self._npu_type = None

        # Initial scan
        self.refresh_hardware_info()

        # Set up degradation callbacks
        self.resource_manager.register_degradation_callback(self._handle_resource_degradation)

    def _handle_resource_degradation(self):
        """Handle resource degradation events"""
        if not self.degradation_active:
            logger.warning("Resource degradation triggered - enabling graceful degradation mode")
            self.degradation_active = True
            # Could trigger model unloading, reduced batch sizes, etc.

    def detect_hardware_type(self) -> HardwareType:
        """Detect the specific hardware acceleration type available"""
        # Check for CUDA first (most capable)
        if self._is_cuda_available():
            return HardwareType.GPU_CUDA

        # Check for NPU variants
        npu_type = self._detect_npu_type()
        if npu_type:
            return npu_type

        # Check for general GPU
        if self._gpu_info:
            vendor = self._gpu_info.get("vendor", "unknown")
            if vendor == "nvidia":
                # NVIDIA without CUDA - might be limited drivers
                return HardwareType.GPU_GENERAL
            elif vendor in ["amd", "intel"]:
                return HardwareType.GPU_GENERAL

        # Fallback to CPU
        return HardwareType.CPU_ONLY

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        if self._cuda_available:  # Cached result
            return True

        try:
            providers = self._accel_providers or []
            if any('CUDAExecutionProvider' in p for p in providers):
                self._cuda_available = True
                return True

            # Try to import CUDA-related modules
            try:
                import torch
                if torch.cuda.is_available():
                    self._cuda_available = True
                    return True
            except ImportError:
                pass

        except Exception as e:
            logger.debug(f"CUDA availability check failed: {e}")

        return False

    def _detect_npu_type(self) -> Optional[HardwareType]:
        """Detect specific NPU type"""
        if self._npu_type:  # Cached result
            return self._npu_type

        # Check architecture first
        architecture = self._cpu_info.get("architecture", "").lower()

        # Apple Silicon (ARM64)
        if "arm64" in architecture or platform.system() == "Darwin":
            try:
                providers = self._accel_providers or []
                if any('CoreMLExecutionProvider' in p for p in providers):
                    self._npu_type = HardwareType.NPU_APPLE
                    return self._npu_type
            except Exception:
                pass

        # Qualcomm (ARM64)
        if "arm64" in architecture:
            cpu_name = platform.processor().lower()
            if "qualcomm" in cpu_name or "snapdragon" in cpu_name or "hexagon" in cpu_name:
                self._npu_type = HardwareType.NPU_QUALCOMM
                return self._npu_type

        # Intel NPU (check for OpenVINO NPU)
        try:
            import importlib
            ov = importlib.import_module("openvino.runtime")
            core = ov.Core()
            devices = core.available_devices
            for dev in devices:
                if "NPU" in dev.upper():
                    self._npu_type = HardwareType.NPU_INTEL
                    return self._npu_type
        except Exception:
            pass

        # Generic NPU detection
        if self._has_npu():
            # Default to Intel if we can't determine specific type
            self._npu_type = HardwareType.NPU_INTEL
            return self._npu_type

        return None

    def allocate_model_memory(self, model_name: str, estimated_mb: float) -> bool:
        """Allocate memory for a model based on hardware capabilities"""
        hardware_type = self.detect_hardware_type()

        # Adjust allocation based on hardware type
        if hardware_type == HardwareType.GPU_CUDA:
            # CUDA can handle larger allocations
            provider = "cuda"
        elif hardware_type in [HardwareType.GPU_GENERAL, HardwareType.NPU_APPLE]:
            # General GPU or Apple NPU
            provider = "gpu"
        elif hardware_type in [HardwareType.NPU_QUALCOMM, HardwareType.NPU_INTEL]:
            # Other NPUs
            provider = "npu"
        else:
            # CPU fallback
            provider = "cpu"
            # CPUs can handle smaller allocations, reduce estimate
            estimated_mb = min(estimated_mb, 1024)  # Cap at 1GB for CPU

        return self.resource_manager.allocate_memory(model_name, provider, estimated_mb)

    def deallocate_model_memory(self, model_name: str):
        """Deallocate memory for a model"""
        hardware_type = self.detect_hardware_type()

        if hardware_type == HardwareType.GPU_CUDA:
            provider = "cuda"
        elif hardware_type in [HardwareType.GPU_GENERAL, HardwareType.NPU_APPLE]:
            provider = "gpu"
        elif hardware_type in [HardwareType.NPU_QUALCOMM, HardwareType.NPU_INTEL]:
            provider = "npu"
        else:
            provider = "cpu"

        self.resource_manager.deallocate_memory(model_name, provider)

    def check_resource_health(self) -> Dict[str, Any]:
        """Check overall resource health and provide recommendations"""
        health_status = {
            "hardware_type": self.detect_hardware_type().value,
            "memory_pressure": False,
            "cpu_overload": False,
            "recommendations": []
        }

        # Check memory pressure
        exhaustion = self.resource_manager.check_resource_exhaustion()
        if exhaustion:
            health_status["memory_pressure"] = True
            if exhaustion == "critical_memory_exhaustion":
                health_status["recommendations"].append("Unload unused models immediately")
                health_status["recommendations"].append("Reduce model context sizes")
            elif exhaustion == "high_memory_pressure":
                health_status["recommendations"].append("Consider model quantization")
                health_status["recommendations"].append("Reduce batch sizes")

        # Check CPU usage
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 90:
                health_status["cpu_overload"] = True
                health_status["recommendations"].append("High CPU usage detected - consider GPU acceleration if available")
        except Exception:
            pass

        # Hardware-specific recommendations
        hardware_type = self.detect_hardware_type()
        if hardware_type == HardwareType.CPU_ONLY:
            health_status["recommendations"].append("Consider upgrading to GPU/NPU for better performance")
        elif hardware_type in [HardwareType.NPU_APPLE, HardwareType.NPU_QUALCOMM]:
            health_status["recommendations"].append("NPU detected - ensure models are optimized for NPU inference")

        return health_status

    def get_optimized_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get hardware-optimized configuration for a model type"""
        hardware_type = self.detect_hardware_type()
        config = {
            "batch_size": 1,
            "quantization": "none",
            "precision": "fp32",
            "provider": "cpu"
        }

        if hardware_type == HardwareType.GPU_CUDA:
            config.update({
                "batch_size": 4,
                "quantization": "int8",
                "precision": "fp16",
                "provider": "cuda"
            })
        elif hardware_type in [HardwareType.GPU_GENERAL, HardwareType.NPU_APPLE]:
            config.update({
                "batch_size": 2,
                "quantization": "int8",
                "precision": "fp16",
                "provider": "gpu"
            })
        elif hardware_type in [HardwareType.NPU_QUALCOMM, HardwareType.NPU_INTEL]:
            config.update({
                "batch_size": 1,
                "quantization": "int8",
                "precision": "int8",
                "provider": "npu"
            })

        return config
        
    def refresh_hardware_info(self):
        """Refresh cached hardware info"""
        self._cpu_info = self._get_cpu_info()
        self._gpu_info = self._get_gpu_info()
        self._memory_info = self._get_memory_info()
        self._accel_providers = self._get_runtime_providers()
        
    def _get_cpu_info(self) -> Dict:
        """Get CPU information"""
        try:
            return {
                "cores": psutil.cpu_count(logical=False) or 1,
                "threads": psutil.cpu_count(logical=True) or 1,
                "architecture": platform.machine(),
                "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else 0
            }
        except Exception as e:
            logger.warning(f"Error getting CPU info: {e}")
            return {"cores": 1, "threads": 1, "architecture": "unknown", "frequency": 0}
        
    def _get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU for now
                return {
                    "name": gpu.name,
                    "memory_gb": round(gpu.memoryTotal / 1024, 2),
                    "vendor": self._detect_gpu_vendor(gpu.name),
                    "load": gpu.load * 100
                }
        except Exception as e:
            logger.debug(f"GPUtil failed or no NVIDIA GPU: {e}")
            pass
            
        # Fallback: try vendor from onnxruntime providers
        try:
            providers = self._get_runtime_providers() or []
            if any('CUDAExecutionProvider' in p for p in providers):
                return {"name": "NVIDIA (provider)", "memory_gb": 0, "vendor": "nvidia", "load": 0}
            if any('ROCMExecutionProvider' in p for p in providers):
                return {"name": "AMD ROCm (provider)", "memory_gb": 0, "vendor": "amd", "load": 0}
            if any('DmlExecutionProvider' in p for p in providers):
                return {"name": "DirectML (provider)", "memory_gb": 0, "vendor": "microsoft", "load": 0}
            if any('CoreMLExecutionProvider' in p for p in providers):
                return {"name": "Apple CoreML (provider)", "memory_gb": 0, "vendor": "apple", "load": 0}
        except Exception:
            pass
        return None
        
    def _detect_gpu_vendor(self, gpu_name: str) -> str:
        """Detect GPU vendor from name"""
        gpu_name_lower = gpu_name.lower()
        if "nvidia" in gpu_name_lower:
            return "nvidia"
        elif "amd" in gpu_name_lower or "radeon" in gpu_name_lower:
            return "amd"
        elif "intel" in gpu_name_lower:
            return "intel"
        else:
            return "unknown"
            
    def _get_memory_info(self) -> Dict:
        """Get memory information"""
        try:
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent
            }
        except Exception as e:
             logger.warning(f"Error getting memory info: {e}")
             return {"total_gb": 8, "available_gb": 4, "percent": 50}
        
    def get_hardware_profile(self) -> str:
        """
        Determine hardware profile based on capabilities
        Returns: "light", "medium", "heavy", or "npu-optimized"
        """
        # Check for NPU first
        if self._has_npu():
            return "npu-optimized"
        
        # Prefer known accelerators
        providers = self._accel_providers or []
        accelerators = ["CUDAExecutionProvider","ROCMExecutionProvider","DmlExecutionProvider","CoreMLExecutionProvider"]
        if any(p for p in providers if any(x in p for x in accelerators)):
            if self._memory_info.get("total_gb", 0) >= 16:
                return "heavy"
            return "medium"
        
        # Check for GPU with memory heuristic
        if self._gpu_info and self._gpu_info.get("memory_gb", 0) >= 8:
            if self._memory_info.get("total_gb", 0) >= 16:
                return "heavy"
            else:
                return "medium"
        elif self._gpu_info and self._gpu_info.get("memory_gb", 0) >= 4:
            return "medium"
        else:
            # Check CPU and memory
            cores = self._cpu_info.get("cores", 4)
            mem_total = self._memory_info.get("total_gb", 8)
            
            if cores >= 8 and mem_total >= 16:
                return "medium"
            else:
                return "light"
        
    def _has_npu(self) -> bool:
        """Best-effort NPU detection"""
        # Try OpenVINO
        try:
            import importlib
            ov = importlib.import_module("openvino.runtime")
            core = ov.Core()
            devices = core.available_devices
            for dev in devices:
                if "NPU" in dev.upper() or "GNA" in dev.upper():
                    return True
        except Exception:
            pass
            
        # Try onnxruntime accelerators
        try:
            providers = self._accel_providers or []
            if any('NpuExecutionProvider' in p for p in providers):
                return True
        except Exception:
            pass
            
        # CPU/SoC heuristic
        architecture = self._cpu_info.get("architecture", "").lower()
        cpu_vendor_str = platform.processor().lower()
        hints = ["neural", "npu", "ai engine", "ai boost", "hexagon"]
        if any(h in cpu_vendor_str for h in hints) or "npu" in architecture:
            return True
            
        return False
                
    def _get_runtime_providers(self) -> Optional[List[str]]:
        """Return available onnxruntime execution providers"""
        try:
            import onnxruntime as ort
            return list(getattr(ort, 'get_available_providers', lambda: [])()) or list(getattr(ort, 'get_available_providers', []))
        except Exception:
            return []

    async def get_hardware_state(self) -> HardwareState:
        """
        Get current hardware state as a Pydantic model.
        Async for future compatibility with async checks.
        """
        # Refresh real-time stats (load, usage)
        cpu_usage = psutil.cpu_percent(interval=None)
        
        gpu_usage = 0.0
        if self._gpu_info and "load" in self._gpu_info:
            gpu_usage = self._gpu_info["load"]
        elif self._gpu_info:
             # Try to refresh GPU load if possible
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass

        memory_available = self._memory_info.get("available_gb", 0.0)
        
        # Calculate system load (average over last 1 min / cores)
        # On Windows psutil.getloadavg() might not be available
        try:
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()[0]
                cores = self._cpu_info.get("cores", 1)
                current_load = min(load_avg / cores, 1.0)
            else:
                # Fallback for Windows: use CPU percent as proxy
                current_load = cpu_usage / 100.0
        except:
             current_load = cpu_usage / 100.0

        available_tiers = ["cpu", "cloud"]
        if self._gpu_info:
            available_tiers.append("gpu")
        if self._has_npu():
            available_tiers.append("npu")

        return HardwareState(
            gpu_usage=gpu_usage,
            memory_available_gb=memory_available,
            cpu_usage=cpu_usage,
            available_tiers=available_tiers,
            current_load=current_load
        )

"""
Hardware Detection Service for JARVISv3
Ports JARVISv2 hardware detection to JARVISv3 workflow architecture.
"""
import psutil
import GPUtil
import platform
import logging
from typing import Dict, Optional, List
from ..ai.context.schemas import HardwareState

logger = logging.getLogger(__name__)

class HardwareService:
    """
    Detects hardware capabilities and determines appropriate profile.
    Adapted from JARVISv2 HardwareDetector for JARVISv3.
    """
    
    def __init__(self):
        self._cpu_info = {}
        self._gpu_info = {}
        self._memory_info = {}
        self._accel_providers = []
        
        # Initial scan
        self.refresh_hardware_info()
        
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

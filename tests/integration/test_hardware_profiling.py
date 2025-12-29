"""
Integration tests for Hardware Profiling and Model Routing
"""
import pytest
import asyncio
from backend.core.hardware import HardwareService
from backend.core.model_router import model_router


@pytest.mark.asyncio
async def test_hardware_profiling():
    """Test hardware detection and profiling"""
    # Create hardware service instance
    hardware_service = HardwareService()
    
    # Test hardware detection by accessing the cached info
    assert hasattr(hardware_service, '_cpu_info')
    assert hasattr(hardware_service, '_gpu_info')
    assert hasattr(hardware_service, '_memory_info')
    
    # Test that the info has been populated
    assert hardware_service._cpu_info is not None
    assert hardware_service._memory_info is not None
    
    # Test hardware state retrieval
    hardware_state = await hardware_service.get_hardware_state()
    assert hasattr(hardware_state, 'gpu_usage')
    assert hasattr(hardware_state, 'memory_available_gb')
    assert hasattr(hardware_state, 'cpu_usage')
    assert hasattr(hardware_state, 'current_load')
    
    # Test hardware profile detection
    profile = hardware_service.get_hardware_profile()
    assert profile in ["light", "medium", "heavy", "npu-optimized"]


@pytest.mark.asyncio
async def test_model_routing():
    """Test model routing based on hardware"""
    # Check if providers are available
    providers_available = []
    for provider_name, provider in model_router.providers.items():
        if await provider.is_available():
            providers_available.append(provider_name)
    
    # If providers are available, test routing
    if providers_available:
        prompt = "Reply with exactly one word: 'Validated'."
        result = await model_router.generate_response(prompt)
        
        assert result is not None
        assert hasattr(result, 'response')
        assert result.response is not None
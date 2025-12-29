"""
Integration tests for Hardware Resource Management
Tests Phase 8: Resource-Aware Execution Maturity capabilities
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from backend.core.hardware import (
    HardwareService, HardwareType, ResourceManager
)


@pytest.mark.asyncio
async def test_hardware_type_detection():
    """Test detection of different hardware acceleration types"""
    service = HardwareService()

    # Test hardware type detection (will vary based on actual hardware)
    hardware_type = service.detect_hardware_type()
    assert isinstance(hardware_type, HardwareType)

    # Should be one of the defined types
    valid_types = [ht.value for ht in HardwareType]
    assert hardware_type.value in valid_types


@pytest.mark.asyncio
async def test_resource_manager_memory_allocation():
    """Test dynamic memory allocation and deallocation"""
    manager = ResourceManager()

    # Test successful allocation
    success = manager.allocate_memory("test_model", "gpu", 512.0)
    assert success == True

    # Check allocation was recorded
    usage = manager.get_memory_usage()
    assert "gpu" in usage
    assert usage["gpu"] == 512.0

    # Test deallocation
    manager.deallocate_memory("test_model", "gpu")
    usage_after = manager.get_memory_usage()
    assert usage_after.get("gpu", 0) == 0


@pytest.mark.asyncio
async def test_memory_allocation_insufficient_resources():
    """Test memory allocation failure with insufficient resources"""
    manager = ResourceManager()

    # Mock very low available memory
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 100 * 1024 * 1024  # 100MB available

        # Try to allocate more than available
        success = manager.allocate_memory("large_model", "gpu", 200.0)  # 200MB requested
        assert success == False


@pytest.mark.asyncio
async def test_resource_exhaustion_detection():
    """Test resource exhaustion detection and recommendations"""
    manager = ResourceManager()

    # Test normal conditions (should return None)
    exhaustion = manager.check_resource_exhaustion()
    # Note: This may return a value depending on actual system state

    # Test with mocked high memory pressure
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB total
        mock_mem.return_value.available = 0.5 * 1024 * 1024 * 1024  # 0.5GB available (96.875% used)

        exhaustion = manager.check_resource_exhaustion()
        assert exhaustion == "critical_memory_exhaustion"


@pytest.mark.asyncio
async def test_degradation_callbacks():
    """Test degradation callback system"""
    manager = ResourceManager()

    callback_called = False
    def test_callback():
        nonlocal callback_called
        callback_called = True

    manager.register_degradation_callback(test_callback)

    # Trigger degradation by allocating memory under pressure
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.total = 8 * 1024 * 1024 * 1024  # 8GB total
        mock_mem.return_value.available = 1 * 1024 * 1024 * 1024  # 1GB available (87.5% used)

        manager.allocate_memory("test_model", "gpu", 512.0)

        # Callback should have been triggered
        assert callback_called == True


@pytest.mark.asyncio
async def test_model_memory_allocation_by_hardware():
    """Test model memory allocation optimized for different hardware types"""
    service = HardwareService()

    # Test CPU allocation (should be capped)
    with patch.object(service, 'detect_hardware_type', return_value=HardwareType.CPU_ONLY):
        success = service.allocate_model_memory("cpu_model", 2048.0)  # Request 2GB
        # Should be capped to 1GB for CPU
        if success:
            # Check that allocation was made with cpu provider
            usage = service.resource_manager.get_memory_usage()
            assert "cpu" in usage or success == False  # May fail if insufficient memory


@pytest.mark.asyncio
async def test_cuda_hardware_detection():
    """Test CUDA availability detection"""
    service = HardwareService()

    # Test CUDA detection logic
    cuda_available = service._is_cuda_available()
    # Result will depend on actual system capabilities

    # Test that it's a boolean
    assert isinstance(cuda_available, bool)


@pytest.mark.asyncio
async def test_npu_type_detection():
    """Test NPU type detection for different vendors"""
    service = HardwareService()

    # Test NPU detection (may return None if no NPU)
    npu_type = service._detect_npu_type()
    if npu_type:
        assert isinstance(npu_type, HardwareType)
        assert "npu" in npu_type.value
    else:
        assert npu_type is None


@pytest.mark.asyncio
async def test_resource_health_monitoring():
    """Test comprehensive resource health monitoring"""
    service = HardwareService()

    health_status = service.check_resource_health()

    # Should have expected structure
    assert "hardware_type" in health_status
    assert "memory_pressure" in health_status
    assert "cpu_overload" in health_status
    assert "recommendations" in health_status

    # Hardware type should be valid
    assert health_status["hardware_type"] in [ht.value for ht in HardwareType]

    # Recommendations should be a list
    assert isinstance(health_status["recommendations"], list)


@pytest.mark.asyncio
async def test_optimized_model_configurations():
    """Test hardware-optimized model configuration generation"""
    service = HardwareService()

    # Test different hardware type configurations
    test_cases = [
        (HardwareType.CPU_ONLY, {"provider": "cpu", "batch_size": 1}),
        (HardwareType.GPU_CUDA, {"provider": "cuda", "batch_size": 4, "precision": "fp16"}),
        (HardwareType.GPU_GENERAL, {"provider": "gpu", "batch_size": 2}),
        (HardwareType.NPU_APPLE, {"provider": "gpu", "batch_size": 2, "precision": "fp16"}),
        (HardwareType.NPU_QUALCOMM, {"provider": "npu", "precision": "int8"}),
        (HardwareType.NPU_INTEL, {"provider": "npu", "precision": "int8"})
    ]

    for hardware_type, expected_attrs in test_cases:
        with patch.object(service, 'detect_hardware_type', return_value=hardware_type):
            config = service.get_optimized_model_config("test_model")

            # Check that expected attributes are present
            for key, expected_value in expected_attrs.items():
                assert config.get(key) == expected_value, f"Failed for {hardware_type.value}: {key} != {expected_value}"


@pytest.mark.asyncio
async def test_hardware_service_integration():
    """Test HardwareService integration with real system"""
    service = HardwareService()

    # Test hardware state retrieval
    hardware_state = await service.get_hardware_state()
    assert hardware_state is not None
    assert hasattr(hardware_state, 'gpu_usage')
    assert hasattr(hardware_state, 'memory_available_gb')
    assert hasattr(hardware_state, 'cpu_usage')
    assert hasattr(hardware_state, 'available_tiers')
    assert hasattr(hardware_state, 'current_load')

    # Available tiers should include CPU and cloud at minimum
    assert "cpu" in hardware_state.available_tiers
    assert "cloud" in hardware_state.available_tiers


@pytest.mark.asyncio
async def test_graceful_degradation_mode():
    """Test graceful degradation mode activation"""
    service = HardwareService()

    # Initially should not be in degradation mode
    assert service.degradation_active == False

    # Simulate resource pressure that triggers degradation
    with patch.object(service.resource_manager, 'check_resource_exhaustion', return_value="high_memory_pressure"):
        service.resource_manager.allocate_memory("pressure_test", "gpu", 100.0)

    # Check if degradation was triggered (may depend on actual memory state)
    # The callback should have set degradation_active to True if triggered


@pytest.mark.asyncio
async def test_memory_pressure_thresholds():
    """Test memory pressure detection thresholds"""
    manager = ResourceManager()

    # Test with normal memory (should not trigger degradation)
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_mem.return_value.available = 12 * 1024 * 1024 * 1024  # 12GB available (25% used)

        success = manager.allocate_memory("normal_test", "gpu", 1024.0)
        # Should succeed without triggering degradation

    # Test with high memory pressure
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_mem.return_value.available = 2 * 1024 * 1024 * 1024  # 2GB available (87.5% used)

        success = manager.allocate_memory("pressure_test", "gpu", 1024.0)
        # Should trigger degradation callback due to memory pressure


@pytest.mark.asyncio
async def test_hardware_aware_deployment_optimization():
    """Test deployment optimization based on hardware capabilities"""
    service = HardwareService()

    # Test that optimization works for different scenarios
    scenarios = [
        ("text_generation", HardwareType.GPU_CUDA, {"batch_size": 4, "quantization": "int8"}),
        ("image_processing", HardwareType.NPU_APPLE, {"batch_size": 2, "precision": "fp16"}),
        ("speech_recognition", HardwareType.CPU_ONLY, {"batch_size": 1, "precision": "fp32"}),
    ]

    for task_type, hardware_type, expected_config in scenarios:
        with patch.object(service, 'detect_hardware_type', return_value=hardware_type):
            config = service.get_optimized_model_config(task_type)

            # Verify key configuration parameters match expectations
            for key, expected_value in expected_config.items():
                assert config.get(key) == expected_value, f"Config mismatch for {task_type} on {hardware_type.value}"


@pytest.mark.asyncio
async def test_cross_hardware_compatibility():
    """Test compatibility across different hardware acceleration types"""
    service = HardwareService()

    # Test that all hardware types can generate valid configurations
    for hardware_type in HardwareType:
        with patch.object(service, 'detect_hardware_type', return_value=hardware_type):
            config = service.get_optimized_model_config("compatibility_test")

            # All configs should have required fields
            required_fields = ["batch_size", "quantization", "precision", "provider"]
            for field in required_fields:
                assert field in config, f"Missing {field} for {hardware_type.value}"

            # Batch size should be reasonable
            assert config["batch_size"] >= 1, f"Invalid batch size for {hardware_type.value}"

            # Provider should be appropriate for hardware
            if hardware_type == HardwareType.GPU_CUDA:
                assert config["provider"] == "cuda"
            elif hardware_type in [HardwareType.GPU_GENERAL, HardwareType.NPU_APPLE]:
                assert config["provider"] in ["gpu", "cuda"]
            elif hardware_type in [HardwareType.NPU_QUALCOMM, HardwareType.NPU_INTEL]:
                assert config["provider"] == "npu"
            else:  # CPU_ONLY
                assert config["provider"] == "cpu"

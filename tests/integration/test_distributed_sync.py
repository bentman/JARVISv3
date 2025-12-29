"""
Integration tests for Distributed Sync Service
"""
import pytest
import asyncio
from backend.core.distributed_manager import distributed_manager


@pytest.mark.asyncio
async def test_distributed_manager():
    """Test distributed manager functionality"""
    # Test that distributed manager is available
    assert distributed_manager is not None
    
    # Test manager initialization
    assert hasattr(distributed_manager, 'hardware_service')
    assert hasattr(distributed_manager, '_running')
    
    # Test start/stop functionality
    await distributed_manager.start()
    assert distributed_manager._running is True
    
    # Test that the task was created
    assert distributed_manager._task is not None
    
    # Stop the manager
    await distributed_manager.stop()
    assert distributed_manager._running is False
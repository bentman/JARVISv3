"""
Unit tests for Observability System
"""
import pytest
import asyncio
from backend.core.observability import setup_observability, health_monitor


@pytest.mark.asyncio
async def test_observability_setup():
    """Test observability system setup"""
    # Setup observability
    setup_observability(log_level="INFO")

    # Run health checks
    health_result = await health_monitor.run_health_checks()
    assert isinstance(health_result, dict)

    # Get system metrics
    system_metrics = health_monitor.get_system_metrics()
    assert "uptime_seconds" in system_metrics
    assert "total_requests" in system_metrics
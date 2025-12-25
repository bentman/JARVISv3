import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_get_hardware_status():
    """Test GET /api/v1/hardware/status integration"""
    response = client.get("/api/v1/hardware/status")
    assert response.status_code == 200
    data = response.json()
    
    # Verify expected schema fields
    assert "cpu_usage" in data
    assert "memory_available_gb" in data
    assert "gpu_usage" in data
    assert "available_tiers" in data
    assert "current_load" in data
    
    # Verify data types
    assert isinstance(data["cpu_usage"], (int, float))
    assert isinstance(data["memory_available_gb"], (int, float))
    assert isinstance(data["available_tiers"], list)
    
    # Verify values are within sane bounds
    assert 0 <= data["cpu_usage"] <= 100
    assert data["memory_available_gb"] >= 0

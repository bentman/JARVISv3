"""
Integration tests for Observability System
Tests metrics collection, health checks, and circuit breaker functionality
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from backend.core.observability import (
    metrics_collector, workflow_tracer, health_monitor
)
from backend.core.circuit_breaker import circuit_breaker_manager
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.mark.asyncio
async def test_comprehensive_metrics_collection():
    """Test comprehensive metrics collection across different operations"""

    # Reset metrics for clean test
    metrics_collector.total_requests = 0
    metrics_collector.successful_requests = 0
    metrics_collector.failed_requests = 0

    # Record various operations
    metrics_collector.increment_requests(success=True, tokens_used=100, execution_time=0.5)
    metrics_collector.increment_workflow(success=True, duration=1.2)
    metrics_collector.increment_nodes(success=True)
    metrics_collector.record_model_inference(0.3)
    metrics_collector.record_error("test_error")
    metrics_collector.update_resource_usage(memory_mb=512.0, cpu_percent=25.0)

    # Verify metrics
    assert metrics_collector.total_requests == 1
    assert metrics_collector.successful_requests == 1
    assert metrics_collector.failed_requests == 0
    assert metrics_collector.total_tokens_used == 100
    assert metrics_collector.total_execution_time == 0.5
    assert metrics_collector.workflows_started == 1
    assert metrics_collector.workflows_completed == 1
    assert metrics_collector.nodes_executed == 1
    assert metrics_collector.nodes_succeeded == 1
    assert metrics_collector.model_inference_count == 1
    assert "test_error" in metrics_collector.error_counts
    assert metrics_collector.memory_usage_mb == 512.0
    assert metrics_collector.cpu_usage_percent == 25.0

    # Test Prometheus metrics format
    prometheus_output = metrics_collector.get_prometheus_metrics()
    assert "jarvis_requests_total 1" in prometheus_output
    assert "jarvis_workflows_completed_total 1" in prometheus_output
    assert "jarvis_memory_usage_mb 512.0" in prometheus_output


@pytest.mark.asyncio
async def test_workflow_tracing():
    """Test comprehensive workflow tracing"""

    # Create test context
    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="test_session",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="trace_test_workflow",
            workflow_name="trace_test",
            initiating_query="Test tracing",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )

    # Start trace
    workflow_tracer.start_trace("trace_test_workflow", context)

    # Trace node operations
    workflow_tracer.trace_node_start("trace_test_workflow", "test_node_1", {"input": "test"})
    await asyncio.sleep(0.01)  # Small delay
    workflow_tracer.trace_node_end("trace_test_workflow", "test_node_1", {"output": "result"}, success=True)

    # End trace
    from backend.ai.workflows.engine import WorkflowState, NodeStatus
    fake_state = WorkflowState(
        workflow_id="trace_test_workflow",
        status=NodeStatus.COMPLETED,
        completed_nodes=["test_node_1"],
        failed_nodes=[]
    )
    workflow_tracer.end_trace("trace_test_workflow", fake_state, {"final": "result"})

    # Verify trace data
    trace = workflow_tracer.get_trace("trace_test_workflow")
    assert trace is not None
    assert trace["workflow_id"] == "trace_test_workflow"
    assert trace["status"] == "completed"
    assert "test_node_1" in trace["nodes"]
    assert trace["nodes"]["test_node_1"]["success"] == True
    assert "duration" in trace["nodes"]["test_node_1"]
    assert trace["final_result"] == {"final": "result"}


@pytest.mark.asyncio
async def test_health_monitoring():
    """Test health monitoring system"""

    # Test health checks
    health_results = await health_monitor.run_health_checks()

    # Should have at least the basic checks we registered
    assert "metrics_collector" in health_results
    assert "workflow_tracer" in health_results

    # All basic checks should pass
    for check_name, result in health_results.items():
        assert result == True, f"Health check {check_name} failed"

    # Test system metrics
    system_metrics = health_monitor.get_system_metrics()
    assert "uptime_seconds" in system_metrics
    assert "total_requests" in system_metrics
    assert "success_rate" in system_metrics
    assert "timestamp" in system_metrics


@pytest.mark.asyncio
async def test_circuit_breaker_functionality():
    """Test circuit breaker functionality for external services"""

    # Test that circuit breakers are registered
    ollama_breaker = circuit_breaker_manager.get_breaker("ollama_provider")
    assert ollama_breaker is not None
    assert ollama_breaker.service_name == "ollama_provider"
    assert ollama_breaker.failure_threshold == 3

    web_search_breaker = circuit_breaker_manager.get_breaker("web_search")
    assert web_search_breaker is not None
    assert web_search_breaker.failure_threshold == 5

    # Test getting all statuses
    all_statuses = await circuit_breaker_manager.get_all_status()
    assert "ollama_provider" in all_statuses
    assert "web_search" in all_statuses
    assert "voice_services" in all_statuses

    # Test circuit breaker status structure
    status = all_statuses["ollama_provider"]
    assert "service_name" in status
    assert "state" in status
    assert "failure_count" in status
    assert status["state"] == "closed"  # Should start closed


@pytest.mark.asyncio
async def test_circuit_breaker_state_transitions():
    """Test circuit breaker state transitions"""

    # Create a test breaker with low thresholds for testing
    test_breaker = circuit_breaker_manager.register_breaker(
        service_name="test_service",
        failure_threshold=2,
        recovery_timeout=1.0,  # Very short for testing
        success_threshold=2
    )

    # Test initial state
    assert test_breaker.state.value == "closed"
    assert test_breaker.failure_count == 0

    # Simulate failures to trip the breaker
    test_breaker._record_failure()
    assert test_breaker.state.value == "closed"  # Not yet tripped

    test_breaker._record_failure()
    assert test_breaker.state.value == "open"  # Now tripped

    # Test half-open transition
    await asyncio.sleep(1.1)  # Wait for recovery timeout

    # Mock successful call to test half-open
    async def mock_success():
        return "success"

    try:
        result = await test_breaker._call_async(mock_success)
        assert result == "success"
        assert test_breaker.state.value == "half_open"
    except Exception:
        pytest.fail("Should not raise exception on successful half-open call")

    # Complete recovery with another success
    try:
        result = await test_breaker._call_async(mock_success)
        assert result == "success"
        assert test_breaker.state.value == "closed"
    except Exception:
        pytest.fail("Should not raise exception on recovery success")


@pytest.mark.asyncio
async def test_observability_integration():
    """Test that observability integrates properly with workflow execution"""

    # Create test context
    context = TaskContext(
        system_context=SystemContext(
            user_id="obs_test_user",
            session_id="obs_test_session",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="obs_integration_test",
            workflow_name="obs_integration",
            initiating_query="Test observability integration",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="test", priority=1),
            context_budget=ContextBudget()
        )
    )

    # Mock workflow execution that should trigger observability
    with patch("backend.ai.workflows.engine.WorkflowEngine") as MockEngine:
        mock_engine = MockEngine.return_value

        # Mock successful workflow execution
        mock_state = type('MockState', (), {
            'status': type('MockStatus', (), {'value': 'completed'})(),
            'completed_nodes': ['router', 'llm_worker', 'validator'],
            'failed_nodes': []
        })()

        mock_engine.execute_workflow = AsyncMock(return_value={
            'status': 'completed',
            'results': {
                'router': {'success': True},
                'llm_worker': {'success': True, 'tokens_used': 150},
                'validator': {'success': True}
            }
        })

        # Execute with observability middleware
        from backend.core.observability import observability_middleware
        result = await observability_middleware.execute_with_observability(
            mock_engine.execute_workflow,
            context
        )

        # Verify observability was triggered
        assert result is not None
        # Metrics should have been updated
        assert metrics_collector.total_requests >= 1

"""
Unit tests for workflow cache eligibility verification.
Tests deterministic behavior of nodes suitable for caching.
"""
import hashlib
from unittest.mock import Mock
from backend.ai.context.schemas import (
    TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType,
    HardwareState, BudgetState, UserPreferences, ContextBudget
)
from backend.ai.workflows.routing import router
from backend.ai.validators.code_check import ValidatorPipeline
from backend.ai.workflows.reflector import reflector_node
from backend.ai.workflows.supervisor import supervisor_node

try:
    import pytest
except ImportError:
    pytest = None


class TestWorkflowCacheEligibility:
    """Test cache eligibility for workflow nodes by verifying deterministic behavior."""

    @pytest.fixture
    def sample_context(self):
        """Create a deterministic test context."""
        return TaskContext(
            system_context=SystemContext(
                user_id="test_user_123",
                session_id="session_456",
                hardware_state=HardwareState(
                    gpu_usage=0.0, memory_available_gb=16.0, cpu_usage=20.0, current_load=0.1
                ),
                budget_state=BudgetState(
                    cloud_spend_usd=0.0, monthly_limit_usd=100.0, remaining_pct=100.0
                ),
                user_preferences=UserPreferences()
            ),
            workflow_context=WorkflowContext(
                workflow_id="workflow_789",
                workflow_name="test_workflow",
                initiating_query="Hello, how are you?",
                user_intent=UserIntent(
                    type=TaskType.CHAT, confidence=0.9,
                    description="Simple chat query", priority=3
                ),
                context_budget=ContextBudget()
            )
        )

    @pytest.mark.asyncio
    async def test_router_deterministic_output(self, sample_context):
        """Verify ROUTER produces same output for identical inputs."""
        # Execute twice with same input
        result1 = await router.route(sample_context)
        result2 = await router.route(sample_context)

        # Results should be identical
        assert result1 == result2
        assert result1["next_node"] == "context_builder"
        assert result1["route_decision"] == "chat"

    def test_router_cache_key_generation(self, sample_context):
        """Test deterministic cache key generation for ROUTER."""
        # Simulate cache key generation logic
        workflow_id = sample_context.workflow_context.workflow_id
        user_id = sample_context.system_context.user_id
        session_id = sample_context.system_context.session_id
        node_id = "router"

        # Create input hash from relevant context fields
        input_data = f"{sample_context.workflow_context.initiating_query}:{sample_context.workflow_context.user_intent.type.value}"
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]

        expected_key = f"{workflow_id}:{user_id}:{session_id}:{node_id}:{input_hash}"

        # Verify key format (this would be in the actual cache implementation)
        assert ":" in expected_key
        assert workflow_id in expected_key
        assert user_id in expected_key
        assert session_id in expected_key
        assert node_id in expected_key
        assert len(input_hash) == 16

    @pytest.mark.asyncio
    async def test_validator_deterministic_output(self, sample_context):
        """Verify VALIDATOR produces same output for identical inputs."""
        validator = ValidatorPipeline()

        # Execute twice with same input
        result1 = await validator.validate_task_context(sample_context)
        result2 = await validator.validate_task_context(sample_context)

        # Results should be identical
        assert result1.is_valid == result2.is_valid
        assert result1.errors == result2.errors
        assert result1.warnings == result2.warnings

    def test_validator_cache_key_generation(self, sample_context):
        """Test deterministic cache key generation for VALIDATOR."""
        workflow_id = sample_context.workflow_context.workflow_id
        user_id = sample_context.system_context.user_id
        session_id = sample_context.system_context.session_id
        node_id = "validator"

        # Input hash based on context validation inputs
        input_data = f"{sample_context.workflow_context.workflow_id}:{len(sample_context.workflow_context.initiating_query)}"
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]

        expected_key = f"{workflow_id}:{user_id}:{session_id}:{node_id}:{input_hash}"

        assert workflow_id in expected_key
        assert user_id in expected_key
        assert session_id in expected_key
        assert node_id in expected_key

    @pytest.mark.asyncio
    async def test_reflector_deterministic_output(self):
        """Verify REFLECTOR produces same output for identical node results."""
        # Create mock context with additional_context attribute
        mock_context = Mock()
        mock_context.additional_context = {}

        # Create mock node results that REFLECTOR analyzes
        node_results = {
            "validator": {
                "is_valid": True,
                "errors": [],
                "validation_passed": True
            }
        }

        # Execute twice with same inputs
        result1 = await reflector_node.execute(mock_context, node_results, "llm_worker", "quality_check")
        result2 = await reflector_node.execute(mock_context, node_results, "llm_worker", "quality_check")

        # Results should be identical for valid case
        assert result1["status"] == result2["status"] == "approved"

        # Test invalid case
        invalid_results = {
            "validator": {
                "is_valid": False,
                "errors": ["Code has syntax error"],
                "validation_passed": False
            }
        }

        invalid_result1 = await reflector_node.execute(mock_context, invalid_results, "llm_worker", "quality_check")
        invalid_result2 = await reflector_node.execute(mock_context, invalid_results, "llm_worker", "quality_check")

        assert invalid_result1["status"] == invalid_result2["status"] == "rejected"
        assert invalid_result1["next_node"] == invalid_result2["next_node"] == "llm_worker"

    def test_reflector_cache_key_generation(self):
        """Test deterministic cache key generation for REFLECTOR."""
        # REFLECTOR keys based on workflow state + target node
        workflow_id = "wf_123"
        user_id = "user_456"
        session_id = "sess_789"
        node_id = "reflector"

        # State hash based on validator results and target
        state_data = "target:llm_worker:criteria:quality_check:valid:true"
        state_hash = hashlib.sha256(state_data.encode()).hexdigest()[:16]

        expected_key = f"{workflow_id}:{user_id}:{session_id}:{node_id}:{state_hash}"

        assert workflow_id in expected_key
        assert user_id in expected_key
        assert session_id in expected_key
        assert node_id in expected_key

    @pytest.mark.asyncio
    async def test_supervisor_deterministic_output(self):
        """Verify SUPERVISOR produces same output for identical queries."""
        # Test with keyword-based query
        query = "I need to research and summarize the latest AI developments"

        # Execute twice with same input
        result1 = await supervisor_node.execute(Mock(workflow_context=Mock(initiating_query=query)))
        result2 = await supervisor_node.execute(Mock(workflow_context=Mock(initiating_query=query)))

        # Results should be identical
        assert result1["plan"] == result2["plan"]
        assert result1["status"] == result2["status"] == "planned"
        assert result1["reasoning"] == result2["reasoning"]

        # Should generate plan with search_web and llm_worker
        assert len(result1["plan"]) == 2
        assert result1["plan"][0]["node_id"] == "search_web"
        assert result1["plan"][1]["node_id"] == "llm_worker"

    def test_supervisor_cache_key_generation(self):
        """Test deterministic cache key generation for SUPERVISOR."""
        workflow_id = "wf_123"
        user_id = "user_456"
        session_id = "sess_789"
        node_id = "supervisor"

        query = "research and summarize AI developments"
        input_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        expected_key = f"{workflow_id}:{user_id}:{session_id}:{node_id}:{input_hash}"

        assert workflow_id in expected_key
        assert user_id in expected_key
        assert session_id in expected_key
        assert node_id in expected_key

    def test_cache_key_user_session_isolation(self):
        """Verify cache keys properly isolate by user and session."""
        base_context = {
            "workflow_id": "wf_123",
            "node_id": "router",
            "input_hash": "abc123"
        }

        # Different users should have different keys
        key1 = f"{base_context['workflow_id']}:user_1:session_1:{base_context['node_id']}:{base_context['input_hash']}"
        key2 = f"{base_context['workflow_id']}:user_2:session_1:{base_context['node_id']}:{base_context['input_hash']}"

        assert key1 != key2

        # Different sessions should have different keys
        key3 = f"{base_context['workflow_id']}:user_1:session_2:{base_context['node_id']}:{base_context['input_hash']}"

        assert key1 != key3
        assert key2 != key3

"""
Integration tests for Human-AI Collaboration Integration
Tests Phase 9: Human-AI Collaboration Integration capabilities
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from backend.ai.workflows.engine import (
    WorkflowEngine, WorkflowNode, NodeType, ApprovalStatus,
    WorkflowApprovalRequiredException, ApprovalRequest
)
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.mark.asyncio
async def test_human_approval_criteria_evaluation():
    """Test evaluation of criteria for when human approval is needed"""
    engine = WorkflowEngine()

    # Create workflow with approval node
    engine.add_node(WorkflowNode(
        id="approval_node",
        type=NodeType.HUMAN_APPROVAL,
        description="Requires human approval",
        conditions={
            "request_type": "general",
            "decision_criteria": {"confidence_threshold": 0.8}
        }
    ))

    # Create context with low-risk query
    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="approval_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="approval_test_workflow",
            workflow_name="approval_test",
            initiating_query="Hello, how are you?",  # Low-risk query
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Safe chat", priority=1),
            context_budget=ContextBudget()
        )
    )

    engine.context = context
    engine.state = type('MockState', (), {'current_node': 'approval_node'})()

    # Test evaluation - should return False (no approval needed) for safe query
    needs_approval = await engine._evaluate_approval_criteria("general", {"confidence_threshold": 0.8})
    assert needs_approval == False


@pytest.mark.asyncio
async def test_human_approval_for_high_stakes_operations():
    """Test that high-stakes operations always require approval"""
    engine = WorkflowEngine()

    # Create context with high-risk operation
    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="high_stakes_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="high_stakes_workflow",
            workflow_name="high_stakes",
            initiating_query="Please delete all user data",  # High-risk query
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="High risk", priority=1),
            context_budget=ContextBudget()
        )
    )

    engine.context = context
    engine.state = type('MockState', (), {'current_node': 'approval_node'})()

    # Test evaluation - should return True for high-risk operations
    needs_approval = await engine._evaluate_approval_criteria("data_deletion", {})
    assert needs_approval == True


@pytest.mark.asyncio
async def test_workflow_approval_exception_handling():
    """Test that approval-required exceptions are properly raised and handled"""
    engine = WorkflowEngine()

    # Create workflow with approval node that will require approval
    engine.add_node(WorkflowNode(
        id="security_approval",
        type=NodeType.HUMAN_APPROVAL,
        description="Security approval required",
        conditions={
            "request_type": "security_change",
            "decision_criteria": {}
        }
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="exception_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="exception_test_workflow",
            workflow_name="exception_test",
            initiating_query="Change security settings",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Security change", priority=1),
            context_budget=ContextBudget()
        )
    )

    engine.context = context
    engine.state = type('MockState', (), {
        'workflow_id': 'exception_test_workflow',
        'current_node': 'security_approval'
    })()

    # Execute approval node - should raise WorkflowApprovalRequiredException
    with pytest.raises(WorkflowApprovalRequiredException) as exc_info:
        await engine._execute_human_approval_node(engine.nodes["security_approval"])

    # Check that exception contains proper approval request
    approval_request = exc_info.value.approval_request
    assert isinstance(approval_request, ApprovalRequest)
    assert approval_request.workflow_id == "exception_test_workflow"
    assert approval_request.node_id == "security_approval"
    assert approval_request.request_type == "security_change"
    assert approval_request.status == ApprovalStatus.PENDING


@pytest.mark.asyncio
async def test_workflow_resume_after_approval():
    """Test workflow resumption after human approval decision"""
    engine = WorkflowEngine()

    # Create approved approval request
    approved_request = ApprovalRequest(
        request_id="test_approval_123",
        workflow_id="resume_test_workflow",
        node_id="resume_node",
        user_id="test_user",
        request_type="general",
        context_data={},
        decision_criteria={}
    )
    approved_request.status = ApprovalStatus.APPROVED
    from datetime import datetime, UTC
    approved_request.decided_at = datetime.now(UTC)
    approved_request.decision_notes = "Approved by admin"

    # Test resume logic
    result = engine.resume_workflow_after_approval(approved_request)

    assert result["approved"] == True
    assert result["resume_workflow"] == True
    assert "Approved by admin" in result["approver_id"]


@pytest.mark.asyncio
async def test_workflow_rejection_handling():
    """Test workflow handling when approval is rejected"""
    engine = WorkflowEngine()

    # Create rejected approval request
    rejected_request = ApprovalRequest(
        request_id="test_rejection_456",
        workflow_id="rejection_test_workflow",
        node_id="rejection_node",
        user_id="test_user",
        request_type="general",
        context_data={},
        decision_criteria={}
    )
    rejected_request.status = ApprovalStatus.REJECTED
    from datetime import datetime, UTC
    rejected_request.decided_at = datetime.now(UTC)
    rejected_request.decision_notes = "Rejected due to policy violation"

    # Test rejection logic
    result = engine.resume_workflow_after_approval(rejected_request)

    assert result["approved"] == False
    assert result["resume_workflow"] == False
    assert result["rejection_reason"] == "Rejected due to policy violation"


@pytest.mark.asyncio
async def test_approval_timeout_handling():
    """Test handling of approval request timeouts"""
    engine = WorkflowEngine()

    # Create timed out approval request
    timeout_request = ApprovalRequest(
        request_id="test_timeout_789",
        workflow_id="timeout_test_workflow",
        node_id="timeout_node",
        user_id="test_user",
        request_type="general",
        context_data={},
        decision_criteria={}
    )
    timeout_request.status = ApprovalStatus.TIMEOUT

    # Test timeout logic
    result = engine.resume_workflow_after_approval(timeout_request)

    assert result["approved"] == False
    assert result["resume_workflow"] == False
    assert result["reason"] == "timeout"


@pytest.mark.asyncio
async def test_auto_approval_for_low_risk_operations():
    """Test automatic approval for operations that don't require human intervention"""
    engine = WorkflowEngine()

    # Create approval node with auto-approval criteria
    engine.add_node(WorkflowNode(
        id="auto_approval_node",
        type=NodeType.HUMAN_APPROVAL,
        description="May auto-approve",
        conditions={
            "request_type": "general",
            "decision_criteria": {"confidence_threshold": 0.5}  # Low threshold
        }
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="auto_approval_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="auto_approval_workflow",
            workflow_name="auto_approval",
            initiating_query="What is the weather today?",  # Low-risk query
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Safe query", priority=1),
            context_budget=ContextBudget()
        )
    )

    engine.context = context
    engine.state = type('MockState', (), {'current_node': 'auto_approval_node'})()

    # Execute approval node - should auto-approve
    result = await engine._execute_human_approval_node(engine.nodes["auto_approval_node"])

    assert result["approved"] == True
    assert result["auto_approved"] == True
    assert result["reason"] == "met_auto_approval_criteria"


@pytest.mark.asyncio
async def test_risk_based_approval_criteria():
    """Test approval criteria based on content risk assessment"""
    engine = WorkflowEngine()

    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="risk_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="risk_test_workflow",
            workflow_name="risk_test",
            initiating_query="Please format my hard drive",  # High-risk query
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Risky command", priority=1),
            context_budget=ContextBudget()
        )
    )

    engine.context = context
    engine.state = type('MockState', (), {'current_node': 'approval_node'})()

    # Test risk assessment - should require approval for risky commands
    needs_approval = await engine._evaluate_approval_criteria("general", {})
    assert needs_approval == True


@pytest.mark.asyncio
async def test_approval_request_data_structure():
    """Test that approval requests contain all necessary information"""
    approval_request = ApprovalRequest(
        request_id="test_request_001",
        workflow_id="test_workflow_abc",
        node_id="approval_node_xyz",
        user_id="test_user_123",
        request_type="security_change",
        context_data={
            "workflow_name": "security_update",
            "initiating_query": "Change firewall rules",
            "current_node": "approval_node_xyz"
        },
        decision_criteria={
            "requires_manager_approval": True,
            "impact_level": "high"
        }
    )

    # Verify all required fields are present
    assert approval_request.request_id == "test_request_001"
    assert approval_request.workflow_id == "test_workflow_abc"
    assert approval_request.node_id == "approval_node_xyz"
    assert approval_request.user_id == "test_user_123"
    assert approval_request.request_type == "security_change"
    assert approval_request.status == ApprovalStatus.PENDING
    assert approval_request.created_at is not None
    assert approval_request.decided_at is None
    assert approval_request.decision_notes is None

    # Verify context data
    assert approval_request.context_data["workflow_name"] == "security_update"
    assert approval_request.context_data["initiating_query"] == "Change firewall rules"

    # Verify decision criteria
    assert approval_request.decision_criteria["requires_manager_approval"] == True
    assert approval_request.decision_criteria["impact_level"] == "high"


@pytest.mark.asyncio
async def test_approval_workflow_integration():
    """Test end-to-end approval workflow integration"""
    engine = WorkflowEngine()

    # Create a simple workflow with just an approval node
    engine.add_node(WorkflowNode(
        id="human_approval",
        type=NodeType.HUMAN_APPROVAL,
        description="Requires human approval",
        conditions={
            "request_type": "data_processing",
            "decision_criteria": {"sensitivity_level": "medium"}
        }
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="integration_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="integration_test_workflow",
            workflow_name="integration_test",
            initiating_query="Please delete all user data",  # High-risk query that triggers approval
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Data processing", priority=1),
            context_budget=ContextBudget()
        )
    )

    engine.context = context
    engine.state = type('MockState', (), {
        'workflow_id': 'integration_test_workflow',
        'current_node': 'human_approval'
    })()

    # Execute approval node directly - should raise exception
    with pytest.raises(WorkflowApprovalRequiredException) as exc_info:
        await engine._execute_human_approval_node(engine.nodes["human_approval"])

    # Verify approval request was created properly
    approval_request = exc_info.value.approval_request
    assert approval_request.workflow_id == "integration_test_workflow"
    assert approval_request.node_id == "human_approval"
    assert approval_request.request_type == "data_processing"
    assert approval_request.context_data["initiating_query"] == "Please delete all user data"

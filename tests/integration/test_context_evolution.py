"""
Integration tests for Context Evolution and Intelligent Adaptation
Tests Phase 6: Contextual Intelligence Deepening capabilities
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from backend.ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget


@pytest.mark.asyncio
async def test_active_memory_in_production_workflow():
    """Test Active Memory nodes exercised in a complete production workflow"""
    engine = WorkflowEngine()

    # Build a memory-enabled workflow: Research → Store Findings → Retrieve → Synthesize
    engine.add_node(WorkflowNode(
        id="research_topic",
        type=NodeType.SEARCH_WEB,
        description="Research a topic and store findings",
        conditions={"query": "What are the benefits of renewable energy?"}
    ))

    engine.add_node(WorkflowNode(
        id="store_findings",
        type=NodeType.ACTIVE_MEMORY,
        description="Store research findings in memory",
        dependencies=["research_topic"],
        conditions={"operation": "store", "content": "Renewable energy benefits: clean, sustainable, reduces carbon emissions"}
    ))

    engine.add_node(WorkflowNode(
        id="retrieve_context",
        type=NodeType.ACTIVE_MEMORY,
        description="Retrieve stored findings for synthesis",
        dependencies=["store_findings"],
        conditions={"operation": "retrieve", "query": "renewable energy"}
    ))

    engine.add_node(WorkflowNode(
        id="synthesize_response",
        type=NodeType.LLM_WORKER,
        description="Synthesize final response using retrieved memory",
        dependencies=["retrieve_context"]
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="memory_workflow_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="memory_enabled_workflow",
            workflow_name="research_memory_workflow",
            initiating_query="Research renewable energy benefits",
            user_intent=UserIntent(type=TaskType.RESEARCH, confidence=0.9, description="Research task", priority=1),
            context_budget=ContextBudget()
        )
    )

    # Mock external dependencies
    with patch("backend.ai.workflows.search_node.search_node") as mock_search:
        with patch("backend.ai.workflows.active_memory.memory_service") as mock_memory:
            with patch("backend.core.model_router.model_router") as mock_router:
                # Setup mocks
                mock_search.execute = AsyncMock(return_value={"results": "Research findings on renewable energy"})
                mock_memory.add_message = AsyncMock(return_value="msg_123")
                mock_memory.semantic_search = AsyncMock(return_value=[
                    {"content": "Renewable energy benefits: clean, sustainable, reduces carbon emissions", "score": 0.95}
                ])
                mock_router.generate_response = AsyncMock(return_value=type('Response', (), {
                    'response': 'Based on stored research, renewable energy provides clean, sustainable power.',
                    'tokens_used': 25,
                    'execution_time': 0.5
                })())

                result = await engine.execute_workflow(context)

                # Verify workflow completed successfully
                assert result["status"] == "completed"
                assert "memory_enabled_workflow" == result["workflow_id"]

                # Verify all nodes executed
                assert "research_topic" in result["results"]
                assert "store_findings" in result["results"]
                assert "retrieve_context" in result["results"]
                assert "synthesize_response" in result["results"]

                # Verify Active Memory operations
                store_result = result["results"]["store_findings"]
                assert store_result["status"] == "stored"
                assert store_result["message_id"] == "msg_123"

                retrieve_result = result["results"]["retrieve_context"]
                assert retrieve_result["status"] == "retrieved"
                assert len(retrieve_result["results"]) == 1
                assert "renewable energy" in retrieve_result["results"][0]["content"].lower()

                # Verify context evolution occurred
                assert engine.state is not None
                assert len(engine.state.context_evolution) > 0

                # Check for learned patterns
                assert len(engine.state.learning_patterns) > 0
                assert any("stores_" in key for key in engine.state.learning_patterns.keys())
                assert any("retrieves_" in key for key in engine.state.learning_patterns.keys())


@pytest.mark.asyncio
async def test_context_evolution_during_execution():
    """Test context evolution during multi-step task execution"""
    engine = WorkflowEngine()

    # Create workflow that evolves context through multiple memory operations
    engine.add_node(WorkflowNode(
        id="initial_memory",
        type=NodeType.ACTIVE_MEMORY,
        description="Store initial context",
        conditions={"operation": "store", "content": "Initial task context"}
    ))

    engine.add_node(WorkflowNode(
        id="evolve_context_1",
        type=NodeType.ACTIVE_MEMORY,
        description="Add evolutionary context",
        dependencies=["initial_memory"],
        conditions={"operation": "store", "content": "Additional context from step 1"}
    ))

    engine.add_node(WorkflowNode(
        id="retrieve_evolved",
        type=NodeType.ACTIVE_MEMORY,
        description="Retrieve evolved context",
        dependencies=["evolve_context_1"],
        conditions={"operation": "retrieve", "query": "context"}
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="evolution_test_user",
            session_id="context_evolution_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="context_evolution_workflow",
            workflow_name="context_evolution_workflow",
            initiating_query="Test context evolution",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Evolution test", priority=1),
            context_budget=ContextBudget()
        )
    )

    # Mock memory service
    with patch("backend.ai.workflows.active_memory.memory_service") as mock_memory:
        mock_memory.add_message = AsyncMock(side_effect=["msg_1", "msg_2"])
        mock_memory.semantic_search = AsyncMock(return_value=[
            {"content": "Initial task context", "score": 0.9},
            {"content": "Additional context from step 1", "score": 0.85}
        ])

        result = await engine.execute_workflow(context)

        # Verify workflow completed
        assert result["status"] == "completed"

        # Verify context evolved
        assert engine.state is not None
        assert len(engine.state.context_evolution) >= 2  # At least 2 store operations

        # Check that context was enhanced with retrieved memories
        assert "retrieved_memories" in context.additional_context
        assert len(context.additional_context["retrieved_memories"]) == 2

        # Verify adaptation count increased
        assert engine.state.adaptation_count >= 2


@pytest.mark.asyncio
async def test_intelligent_adaptation_from_patterns():
    """Test intelligent adaptation based on learned execution patterns"""
    engine = WorkflowEngine()

    # Create workflow that demonstrates learning
    engine.add_node(WorkflowNode(
        id="memory_op_1",
        type=NodeType.ACTIVE_MEMORY,
        description="Memory operation 1",
        conditions={"operation": "store", "content": "Pattern learning test 1"}
    ))

    engine.add_node(WorkflowNode(
        id="memory_op_2",
        type=NodeType.ACTIVE_MEMORY,
        description="Memory operation 2",
        dependencies=["memory_op_1"],
        conditions={"operation": "store", "content": "Pattern learning test 2"}
    ))

    engine.add_node(WorkflowNode(
        id="memory_op_3",
        type=NodeType.ACTIVE_MEMORY,
        description="Memory operation 3",
        dependencies=["memory_op_2"],
        conditions={"operation": "retrieve", "query": "pattern"}
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="pattern_test_user",
            session_id="pattern_learning_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="pattern_learning_workflow",
            workflow_name="pattern_learning_workflow",
            initiating_query="Test pattern learning",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Pattern test", priority=1),
            context_budget=ContextBudget()
        )
    )

    with patch("backend.ai.workflows.active_memory.memory_service") as mock_memory:
        mock_memory.add_message = AsyncMock(side_effect=["msg_1", "msg_2"])
        mock_memory.semantic_search = AsyncMock(return_value=[
            {"content": "Pattern learning test 1", "score": 0.9},
            {"content": "Pattern learning test 2", "score": 0.8}
        ])

        result = await engine.execute_workflow(context)

        # Verify workflow completed
        assert result["status"] == "completed"

        # Check that patterns were learned
        assert engine.state is not None
        assert len(engine.state.learning_patterns) > 0

        # Test adaptation logic
        adaptation_result = engine.adapt_workflow_from_patterns()

        # Since we have multiple memory operations, should detect memory-intensive pattern
        assert adaptation_result["adapted"] == True
        assert "memory_intensive_workflow" in adaptation_result["adaptations"]
        assert adaptation_result["pattern_count"] >= 2


@pytest.mark.asyncio
async def test_learning_from_task_execution_patterns():
    """Test learning and adaptation from repeated task execution patterns"""
    engine = WorkflowEngine()

    # Create a workflow that repeats memory operations to build learning patterns
    for i in range(6):  # Create enough operations to trigger pattern detection
        engine.add_node(WorkflowNode(
            id=f"memory_store_{i}",
            type=NodeType.ACTIVE_MEMORY,
            description=f"Store memory item {i}",
            conditions={"operation": "store", "content": f"Memory content {i}"}
        ))

        if i > 0:
            engine.nodes[f"memory_store_{i}"].dependencies = [f"memory_store_{i-1}"]

    context = TaskContext(
        system_context=SystemContext(
            user_id="execution_pattern_user",
            session_id="execution_pattern_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="execution_pattern_workflow",
            workflow_name="execution_pattern_workflow",
            initiating_query="Test execution pattern learning",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Execution pattern test", priority=1),
            context_budget=ContextBudget()
        )
    )

    with patch("backend.ai.workflows.active_memory.memory_service") as mock_memory:
        mock_memory.add_message = AsyncMock(side_effect=[f"msg_{i}" for i in range(6)])

        result = await engine.execute_workflow(context)

        # Verify workflow completed
        assert result["status"] == "completed"

        # Check that repetitive patterns were detected
        assert engine.state is not None
        assert sum(engine.state.learning_patterns.values()) >= 6  # 6 store operations

        # Test adaptation for repetitive operations
        adaptation_result = engine.adapt_workflow_from_patterns()

        # Should detect repetitive operations pattern
        assert adaptation_result["adapted"] == True
        assert "repetitive_operations" in adaptation_result["adaptations"]


@pytest.mark.asyncio
async def test_context_evolution_validation():
    """Test that context evolution is properly validated and tracked"""
    engine = WorkflowEngine()

    # Create workflow with context evolution
    engine.add_node(WorkflowNode(
        id="store_initial",
        type=NodeType.ACTIVE_MEMORY,
        description="Store initial data",
        conditions={"operation": "store", "content": "Initial context data"}
    ))

    engine.add_node(WorkflowNode(
        id="retrieve_and_evolve",
        type=NodeType.ACTIVE_MEMORY,
        description="Retrieve and evolve context",
        dependencies=["store_initial"],
        conditions={"operation": "retrieve", "query": "context"}
    ))

    context = TaskContext(
        system_context=SystemContext(
            user_id="validation_test_user",
            session_id="context_validation_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="context_validation_workflow",
            workflow_name="context_validation_workflow",
            initiating_query="Test context validation",
            user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Validation test", priority=1),
            context_budget=ContextBudget()
        )
    )

    with patch("backend.ai.workflows.active_memory.memory_service") as mock_memory:
        mock_memory.add_message = AsyncMock(return_value="msg_1")
        mock_memory.semantic_search = AsyncMock(return_value=[
            {"content": "Initial context data", "score": 0.95}
        ])

        # Initial context state
        initial_memory_capable = context.additional_context.get("memory_capable", False)
        initial_memories = len(context.additional_context.get("retrieved_memories", []))

        result = await engine.execute_workflow(context)

        # Verify workflow completed
        assert result["status"] == "completed"

        # Verify context evolved
        assert context.additional_context.get("memory_capable") == True  # Added by evolution
        assert len(context.additional_context.get("retrieved_memories", [])) == 1  # Added by retrieval

        # Verify evolution was tracked
        assert engine.state is not None
        assert len(engine.state.context_evolution) >= 2  # Store + retrieve operations
        assert engine.state.adaptation_count >= 2

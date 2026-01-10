"""
Agentic test for Voice Workflow Integration
Tests the VoiceNode within the workflow context.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from backend.ai.context.schemas import TaskContext, SystemContext, WorkflowContext, UserIntent, TaskType, HardwareState, BudgetState, UserPreferences, ContextBudget, VoiceContext
from backend.ai.workflows.voice_node import voice_node

@pytest.mark.asyncio
async def test_voice_node_stt_execution():
    """Test STT node execution with mocked voice service"""
    # Setup Context
    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="voice_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="voice_workflow",
            workflow_name="voice_test",
            initiating_query="", # Empty initially
            user_intent=UserIntent(type=TaskType.VOICE, confidence=1.0, description="Voice test", priority=1),
            context_budget=ContextBudget()
        ),
        voice_context=VoiceContext(
            audio_input=b"fake_audio_bytes"
        )
    )

    # Mock voice service
    with patch("backend.ai.workflows.voice_node.voice_service") as mock_service:
        mock_service.speech_to_text = AsyncMock(return_value=("Hello JARVIS", 0.95))
        
        # Execute Node
        result = await voice_node.execute_stt(context, {})
        
        # Verify Results
        assert result["success"] is True
        assert result["transcription"] == "Hello JARVIS"
        assert context.voice_context.transcription == "Hello JARVIS"
        assert context.workflow_context.initiating_query == "Hello JARVIS" # Should update query
        
        mock_service.speech_to_text.assert_called_once_with(b"fake_audio_bytes")

@pytest.mark.asyncio
async def test_voice_node_tts_execution():
    """Test TTS node execution with mocked voice service"""
    # Setup Context
    context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="voice_test",
            hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
            budget_state=BudgetState(remaining_pct=100.0),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="voice_workflow",
            workflow_name="voice_test",
            initiating_query="Hello",
            user_intent=UserIntent(type=TaskType.VOICE, confidence=1.0, description="Voice test", priority=1),
            context_budget=ContextBudget()
        )
    )
    
    # Mock previous node result (e.g. LLM response)
    node_results = {
        "llm_worker": {
            "response": "I am online."
        }
    }

    # Mock voice service
    with patch("backend.ai.workflows.voice_node.voice_service") as mock_service:
        mock_service.text_to_speech = AsyncMock(return_value=b"fake_audio_output")
        
        # Execute Node
        result = await voice_node.execute_tts(context, node_results)
        
        # Verify Results
        assert result["success"] is True
        assert context.voice_context is not None
        assert context.voice_context.audio_output == b"fake_audio_output"
        
        mock_service.text_to_speech.assert_called_once_with("I am online.")

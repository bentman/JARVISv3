"""
Integration tests for Chat Workflow
"""
import pytest
import asyncio
from backend.ai.workflows.chat_workflow import ChatWorkflow


@pytest.mark.asyncio
async def test_complete_chat_workflow():
    """Test the complete chat workflow end-to-end"""
    workflow = ChatWorkflow()
    
    # Test a simple chat execution
    result = await workflow.execute_chat(
        user_id="test_user_123",
        query="Hello, what can you do?"
    )
    
    # Verify the result structure
    assert "response" in result
    assert "workflow_id" in result
    assert "tokens_used" in result
    assert "validation_passed" in result
    
    # Verify that a response was generated (even if simulated)
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    assert result["validation_passed"] is True
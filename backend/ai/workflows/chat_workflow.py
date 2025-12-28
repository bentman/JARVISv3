"""
Chat Workflow Implementation for JARVISv3
Implements the complete chat workflow using the "Agentic Graph" architecture
"""
import asyncio
import os
from datetime import datetime, UTC
from typing import Dict, Any, Optional, AsyncIterable
from pydantic import BaseModel
import logging

from ..context.schemas import TaskContext, TaskType, NodeContext
from .engine import WorkflowEngine, WorkflowNode, NodeType
from ..generators.context_builder import ContextBuilder
from ..validators.code_check import ValidatorPipeline
from ...core.model_router import model_router
from ...core.memory import memory_service

logger = logging.getLogger(__name__)


class ChatWorkflowState(BaseModel):
    """State for the chat workflow"""
    conversation_id: str
    user_message: str
    agent_response: Optional[str] = None
    context_used: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    workflow_completed: bool = False
    timestamp: datetime = datetime.now(UTC)


class ChatWorkflow:
    """Implements the complete chat workflow"""
    
    def __init__(self):
        self.engine = WorkflowEngine()
        self.context_builder = ContextBuilder()
        self.validator = ValidatorPipeline()
        self.state = None
        
        # Initialize the workflow
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Set up the chat workflow nodes"""
        # Router node - determines the type of response needed
        self.engine.add_node(WorkflowNode(
            id="router",
            type=NodeType.ROUTER,
            description="Classify user intent and route to appropriate handler",
            execute_func=self._execute_router
        ))
        
        # Context builder node - assembles all relevant context
        self.engine.add_node(WorkflowNode(
            id="context_builder",
            type=NodeType.CONTEXT_BUILDER,
            description="Build context from user input, history, and available tools",
            dependencies=["router"],
            execute_func=self._execute_context_builder
        ))
        
        # LLM worker node - processes the request with LLM
        self.engine.add_node(WorkflowNode(
            id="llm_worker",
            type=NodeType.LLM_WORKER,
            description="Process request with LLM and generate response",
            dependencies=["context_builder"],
            execute_func=self._execute_llm_worker
        ))
        
        # Validator node - validates the response
        self.engine.add_node(WorkflowNode(
            id="validator",
            type=NodeType.VALIDATOR,
            description="Validate response quality and safety",
            dependencies=["llm_worker"],
            execute_func=self._execute_validator
        ))
        
        # Response formatter node - formats the final response
        self.engine.add_node(WorkflowNode(
            id="response_formatter",
            type=NodeType.LLM_WORKER,  # Reusing for formatting
            description="Format response for user consumption",
            dependencies=["validator"],
            execute_func=self._execute_response_formatter
        ))
    
    async def _execute_router(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the router node"""
        logger.info("Executing router node")
        
        intent = context.workflow_context.user_intent
        route_decision = {
            "next_node": "context_builder",
            "intent_type": intent.type,
            "confidence": intent.confidence,
            "priority": intent.priority
        }
        
        # Update context with routing decision
        if context.node_context:
            context.node_context.output_context = route_decision
        
        return route_decision
    
    async def _execute_context_builder(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the context builder node"""
        logger.info("Executing context builder node")
        from ...core.lifecycle import context_lifecycle_manager
        
        # Fetch conversation history from MemoryService
        session_id = context.system_context.session_id
        history = await memory_service.get_messages(session_id)
        
        additional_context = {
            "conversation_history": history,
            "current_time": datetime.now(UTC).isoformat(),
            "user_preferences": context.system_context.user_preferences.model_dump()
        }
        
        # Update the task context with additional information
        context.additional_context.update(additional_context)
        
        # Apply lifecycle management (summarization/pruning)
        await context_lifecycle_manager.manage_context_lifecycle(context)
        
        context_info = {
            "context_size": await self.context_builder.get_context_size(context),
            "system_info": {
                "user_id": context.system_context.user_id,
                "hardware_state": context.system_context.hardware_state.dict(),
                "budget_state": context.system_context.budget_state.dict()
            },
            "workflow_info": {
                "workflow_id": context.workflow_context.workflow_id,
                "initiating_query": context.workflow_context.initiating_query
            }
        }
        
        return context_info
    
    async def _execute_llm_worker(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM worker node"""
        logger.info("Executing LLM worker node")
        
        # Get the actual LLM provider and model to use
        from ...core.budget import cloud_escalation_manager
        from ...core.lifecycle import context_lifecycle_manager
        
        # Determine if we should use local or cloud LLM based on architecture decisions
        hardware_available = context.system_context.hardware_state.dict()
        task_requirements = {
            "priority": context.workflow_context.user_intent.priority,
            "requires_gpu": True if context.workflow_context.user_intent.type == "coding" else False,
            "memory_gb": 4 if context.workflow_context.user_intent.type == "coding" else 2
        }
        
        # Check if we should escalate to cloud based on local capabilities and budget
        should_escalate = await cloud_escalation_manager.should_escalate_to_cloud(
            context.system_context.user_id,
            hardware_available,
            task_requirements
        )
        
        if should_escalate:
            # Use cloud LLM (OpenAI, Anthropic, etc.)
            response = await self._call_cloud_llm(context)
        else:
            # Use local LLM if available
            response = await self._call_local_llm(context)
        
        # Calculate token usage
        tokens_used = len(response.split())
        context.update_tokens_consumed(tokens_used)
        
        llm_result = {
            "response": response,
            "tokens_used": tokens_used,
            "model_used": "cloud_model" if should_escalate else "local_model",
            "processing_time": 0.1  # This would be measured in real implementation
        }
        
        return llm_result
    
    async def _call_cloud_llm(self, context: TaskContext) -> str:
        """Call cloud-based LLM with proper context and safety measures"""
        import openai
        from ...core.security import security_validator
        
        # Validate the context before sending to cloud
        context_str = str(context.model_dump())
        security_check = await security_validator.validate_input(context_str)
        
        if security_check['has_critical']:
            raise Exception("Security validation failed for cloud LLM call")
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("Cloud LLM API key not configured")
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Prepare the context for the LLM call
        system_message = f"""
        You are JARVISv3, an advanced agentic system with workflow architecture and code-driven context.
        The user has provided the following context: {context.workflow_context.initiating_query}
        """
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4-turbo",  # This would be configurable
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": context.workflow_context.initiating_query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content if content is not None else "I received an empty response from the cloud model."
            
        except Exception as e:
            # Log the error and potentially fall back to local processing
            logger.error(f"Cloud LLM call failed: {str(e)}")
            # Fall back to local processing
            return await self._call_local_llm(context)
    
    async def _call_local_llm(self, context: TaskContext) -> str:
        """Call local LLM using ModelRouter"""
        try:
            prompt = context.workflow_context.initiating_query
            # TODO: Format full prompt with history from context.additional_context["conversation_history"]
            
            result = await model_router.generate_response(prompt)
            return result.response or "I received an empty response from the local model."
        except Exception as e:
            logger.error(f"Local LLM call failed: {str(e)}")
            return "I encountered an error while processing your request locally."
    
    async def _execute_validator(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the validator node"""
        logger.info("Executing validator node")
        
        # Get the LLM response to validate
        llm_result = node_results.get("llm_worker", {})
        response = llm_result.get("response", "")
        
        # Validate the response
        validation_result = await self.validator.validate_llm_output(
            {"response": response}, 
            context
        )
        
        # Update context with validation results
        validation_info = {
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "validation_passed": len(validation_result.errors) == 0
        }
        
        return validation_info
    
    async def _execute_response_formatter(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the response formatter node"""
        logger.info("Executing response formatter node")
        
        # Get the validated response
        llm_result = node_results.get("llm_worker", {})
        validator_result = node_results.get("validator", {})
        
        response = llm_result.get("response", "I'm sorry, I couldn't process your request.")
        is_valid = validator_result.get("is_valid", True)
        
        # Add workflow metadata to the response
        formatted_response = {
            "response": response,
            "workflow_id": context.workflow_context.workflow_id,
            "tokens_used": llm_result.get("tokens_used", 0),
            "validation_passed": is_valid,
            "timestamp": datetime.now(UTC).isoformat(),
            "workflow_info": {
                "budget_consumed": context.workflow_context.context_budget.consumed_tokens,
                "budget_remaining": context.workflow_context.context_budget.remaining_tokens,
                "context_size": await self.context_builder.get_context_size(context)
            }
        }
        
        return formatted_response
    
    async def execute_chat(self, user_id: str, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the complete chat workflow"""
        logger.info(f"Starting chat workflow for user {user_id} with query: {query}")
        
        session_id = conversation_id or f"session_{datetime.now(UTC).timestamp()}"

        # Build the initial context
        task_context = await self.context_builder.build_task_context(
            user_id=user_id,
            session_id=session_id,
            workflow_id=f"chat_{datetime.now(UTC).timestamp()}",
            workflow_name="chat_workflow",
            initiating_query=query,
            task_type=TaskType.CHAT
        )
        
        # Execute the workflow
        workflow_result = await self.engine.execute_workflow(task_context)
        
        # Extract the final response
        final_response = workflow_result.get("results", {}).get("response_formatter", {})
        
        # Create workflow state
        self.state = ChatWorkflowState(
            conversation_id=task_context.workflow_context.workflow_id,
            user_message=query,
            agent_response=final_response.get("response"),
            context_used=task_context.dict(),
            tokens_used=final_response.get("tokens_used", 0),
            workflow_completed=workflow_result.get("status") == "completed"
        )
        
        # Persist conversation to memory
        session_id = task_context.system_context.session_id
        # Ensure conversation exists (idempotent-ish via DB handling)
        await memory_service.store_conversation(title=f"Chat {session_id}", conversation_id=session_id)
        
        await memory_service.add_message(session_id, "user", query)
        if final_response.get("response"):
            await memory_service.add_message(session_id, "assistant", final_response.get("response"), tokens=final_response.get("tokens_used", 0))
        
        logger.info(f"Chat workflow completed with status: {workflow_result.get('status')}")
        
        return {
            "response": final_response.get("response", "I'm sorry, I couldn't process your request."),
            "workflow_id": task_context.workflow_context.workflow_id,
            "tokens_used": final_response.get("tokens_used", 0),
            "validation_passed": final_response.get("validation_passed", False),
            "execution_time": workflow_result.get("execution_time", 0),
            "workflow_status": workflow_result.get("status"),
            "workflow_details": workflow_result
        }

    async def execute_chat_stream(self, user_id: str, query: str) -> AsyncIterable[Dict[str, Any]]:
        """Execute the complete chat workflow in streaming mode"""
        logger.info(f"Starting streaming chat workflow for user {user_id}")

        # Build initial context
        task_context = await self.context_builder.build_task_context(
            user_id=user_id,
            session_id=f"session_{datetime.now(UTC).timestamp()}",
            workflow_id=f"chat_{datetime.now(UTC).timestamp()}",
            workflow_name="chat_workflow",
            initiating_query=query,
            task_type=TaskType.CHAT
        )

        full_response = ""
        workflow_id = task_context.workflow_context.workflow_id

        async for event in self.engine.execute_workflow_stream(task_context):
            if event["type"] == "stream_chunk":
                full_response += event["chunk"]
            yield event

        # Persist conversation to memory
        session_id = task_context.system_context.session_id
        await memory_service.store_conversation(title=f"Chat {session_id}", conversation_id=session_id)
        await memory_service.add_message(session_id, "user", query)
        if full_response:
            await memory_service.add_message(session_id, "assistant", full_response, tokens=len(full_response.split()))


# Example usage and test functions
async def test_chat_workflow():
    """Test function to demonstrate the chat workflow"""
    workflow = ChatWorkflow()
    
    # Test a simple chat
    result = await workflow.execute_chat(
        user_id="test_user_123",
        query="Hello, what can you do?"
    )
    
    print(f"Chat workflow result: {result}")
    
    # Test with a more complex query
    result2 = await workflow.execute_chat(
        user_id="test_user_123",
        query="What is the weather like today?"
    )
    
    print(f"Chat workflow result 2: {result2}")
    
    return result, result2


if __name__ == "__main__":
    asyncio.run(test_chat_workflow())

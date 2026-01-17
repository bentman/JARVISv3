"""
Main FastAPI application for JARVISv3
Implements the "Unified Golden Stack" architecture with Workflow Architecture
and Code-Driven Context as first-class concerns

Follows the architecture decisions outlined in Project.md:
- Local-first with explicit cloud escalation
- Python-first workflow engine with declarative extensions
- Defense-in-depth security and privacy controls
- Built-in observability from the start
- Multi-layer validation strategy
- Managed context lifecycle
- Resilient error handling
"""
import asyncio
import logging
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uvicorn

from .ai.context.schemas import TaskContext, TaskType
from .ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from .ai.generators.context_builder import ContextBuilder
from .ai.validators.code_check import ValidatorPipeline
from .ai.workflows.chat_workflow import ChatWorkflow
from .ai.workflows.research_workflow import research_workflow
from .ai.workflows.dev_workflow import init_dev_workflow
from .ai.workflows.agent_registry import agent_registry
from .ai.context.schemas import RemoteNode
from .ai.workflows.templates import workflow_composer, WorkflowTemplate
from .core.node_registry import node_registry
from .core.observability import setup_observability, observability_middleware, health_monitor
from .core.security import security_validator
from .core.budget import budget_manager, cloud_escalation_manager
from .core.lifecycle import context_lifecycle_manager, context_archiver
from .core.auth import auth_manager, User
from .core.database import database_manager
from .core.voice import voice_service
from .core.cache_service import cache_service
from .core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JARVISv3 API",
    description="Advanced agentic system with workflow architecture and code-driven context",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
workflow_engine = WorkflowEngine()
context_builder = ContextBuilder()
validator_pipeline = ValidatorPipeline()
chat_workflow = ChatWorkflow()
dev_workflow = init_dev_workflow(workflow_engine)

# Setup observability
setup_observability(log_level="INFO")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    user_id: str
    query: str
    session_id: Optional[str] = None
    workflow_type: str = "chat"
    user_preferences: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    workflow_id: str
    tokens_used: int
    validation_passed: bool
    execution_time: float


class VoiceSessionRequest(BaseModel):
    """Request model for voice session endpoint"""
    audio_data: str # Base64 encoded audio
    conversation_id: Optional[str] = None
    mode: str = "chat"
    include_web: bool = False
    escalate_llm: bool = False


class VoiceSessionResponse(BaseModel):
    """Response model for voice session endpoint"""
    conversation_id: str
    text_response: str
    audio_data: Optional[str] # Base64 encoded audio response
    detected: bool = True
    workflow_id: str


class WorkflowDefinition(BaseModel):
    """Model for defining workflows via API"""
    workflow_id: str
    nodes: list
    edges: list


class MessageSchema(BaseModel):
    """Schema for a single message"""
    message_id: str
    conversation_id: str
    role: str
    content: str
    timestamp: datetime
    tokens: int = 0
    mode: str = "chat"


class ConversationSchema(BaseModel):
    """Schema for a conversation"""
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = []
    messages: Optional[List[MessageSchema]] = None


# Initialize default workflows
async def initialize_workflows():
    """Initialize default workflows for the system"""
    logger.info("Initializing default workflows...")
    
    # Chat workflow
    workflow_engine.add_node(WorkflowNode(
        id="router",
        type=NodeType.ROUTER,
        description="Route to appropriate workflow based on intent"
    ))
    
    workflow_engine.add_node(WorkflowNode(
        id="context_builder",
        type=NodeType.CONTEXT_BUILDER,
        description="Build context from various sources",
        dependencies=["router"]
    ))
    
    workflow_engine.add_node(WorkflowNode(
        id="llm_worker",
        type=NodeType.LLM_WORKER,
        description="Process with LLM",
        dependencies=["context_builder"]
    ))
    
    workflow_engine.add_node(WorkflowNode(
        id="validator",
        type=NodeType.VALIDATOR,
        description="Validate output",
        dependencies=["llm_worker"]
    ))
    
    logger.info("Default workflows initialized")


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    await initialize_workflows()
    if settings.ENABLE_CACHE:
        await cache_service.initialize()
    logger.info("JARVISv3 API started successfully")


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    from .core.model_manager import model_manager
    available_models = await model_manager.get_available_models()
    cache_healthy = await cache_service.healthy()

    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.1.0",
        "modules": {
            "workflow_engine": "loaded",
            "context_builder": "loaded",
            "validator_pipeline": "loaded",
            "model_manager": "loaded",
            "cache_service": "connected" if cache_healthy else "disconnected"
        },
        "models": {
            "available": available_models,
            "count": len(available_models)
        }
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with system metrics"""
    from .core.observability import health_monitor

    # Run all registered health checks
    health_results = await health_monitor.run_health_checks()

    # Get system metrics
    system_metrics = health_monitor.get_system_metrics()

    # Determine overall health status
    all_healthy = all(health_results.values())
    status = "healthy" if all_healthy else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.1.0",
        "health_checks": health_results,
        "system_metrics": system_metrics,
        "uptime_seconds": system_metrics["uptime_seconds"]
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    from .core.observability import metrics_collector

    # Update resource usage (in a real system, this would be done periodically)
    import psutil
    memory_mb = psutil.virtual_memory().used / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)

    metrics_collector.update_resource_usage(memory_mb=memory_mb, cpu_percent=cpu_percent)

    # Return Prometheus-formatted metrics
    return Response(
        content=metrics_collector.get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/health/circuit-breakers")
async def circuit_breaker_status():
    """Check status of circuit breakers for external dependencies"""
    from .core.circuit_breaker import circuit_breaker_manager

    if not hasattr(circuit_breaker_manager, 'get_all_status'):
        return {"status": "circuit_breaker_not_configured"}

    return await circuit_breaker_manager.get_all_status()

@app.get("/api/v1/hardware/status")
async def get_hardware_status():
    """Get current hardware telemetry"""
    from .core.hardware import HardwareService
    service = HardwareService()
    return await service.get_hardware_state()

@app.get("/api/v1/budget/status")
async def get_budget_status(user_id: str = "admin_123"): # Default for now
    """Get current budget status"""
    from .core.budget import budget_service
    return await budget_service.get_budget_state(user_id)

@app.get("/api/v1/agents")
async def list_agents():
    """List all registered specialized agents"""
    return agent_registry.list_agents()

@app.get("/api/v1/conversations", response_model=List[ConversationSchema])
async def list_conversations():
    """List all conversations"""
    from .core.memory import memory_service
    return await memory_service.get_conversations()

@app.get("/api/v1/conversation/{conversation_id}", response_model=ConversationSchema)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with messages"""
    from .core.memory import memory_service
    conv = await memory_service.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get messages
    messages = await memory_service.get_messages(conversation_id)
    conv['messages'] = messages
    return conv

@app.delete("/api/v1/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages"""
    from .core.memory import memory_service
    success = await memory_service.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
    return {"status": "success", "conversation_id": conversation_id}

@app.post("/api/v1/distributed/register")
async def register_remote_node(node: RemoteNode):
    """Register a remote JARVISv3 node"""
    await node_registry.register_node(node)
    return {"status": "success", "local_node_id": node_registry.local_node_id}

@app.post("/api/v1/distributed/heartbeat")
async def node_heartbeat(node_id: str, load: float):
    """Update heartbeat and load for a registered node"""
    async with node_registry._lock:
        if node_id in node_registry.nodes:
            node_registry.nodes[node_id].last_heartbeat = datetime.now(UTC)
            node_registry.nodes[node_id].current_load = load
            return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Node not registered")

@app.get("/api/v1/distributed/nodes")
async def list_distributed_nodes():
    """List all active nodes in the distributed network"""
    return await node_registry.get_active_nodes()

@app.post("/api/v1/distributed/execute")
async def execute_remote_task(context: TaskContext, node_id: str):
    """Execute a specific workflow node received from a remote instance"""
    # Initialize engine with the received context
    workflow_engine.context = context
    try:
        result = await workflow_engine.execute_node(node_id)
        return {"status": "success", "result": result, "updated_context": workflow_engine.context}
    except Exception as e:
        logger.error(f"Remote execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that uses the workflow architecture"""
    try:
        start_time = datetime.now(UTC)
        
        # Validate input for security issues
        validation_result = await security_validator.validate_input(request.query)
        if validation_result['has_critical']:
            raise HTTPException(status_code=400, detail="Input contains security violations")
        
        # Authenticate user if needed
        user = await database_manager.get_user(request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check user permissions
        user_obj = User(user['user_id'], user['username'], user['email'], user['role'], user['permissions'])
        await auth_manager.require_permission(user_obj, "execute")
        # Route to appropriate workflow implementation
        if request.workflow_type == "research":
            result = await research_workflow.execute(
                user_id=request.user_id,
                query=request.query
            )
            # Adapt result to ChatResponse format
            result["validation_passed"] = result.get("details", {}).get("results", {}).get("validator", {}).get("is_valid", True)
            result["tokens_used"] = result.get("details", {}).get("results", {}).get("llm_worker", {}).get("tokens_used", 0)
        elif request.workflow_type == "dev":
            # Build context for dev workflow
            task_context = await context_builder.build_task_context(
                user_id=request.user_id,
                session_id=request.session_id or f"session_{datetime.now(UTC).timestamp()}",
                workflow_id=f"dev_{datetime.now(UTC).timestamp()}",
                workflow_name="dev",
                initiating_query=request.query,
                task_type=TaskType.CODING
            )
            
            # Execute dev workflow
            result_raw = await dev_workflow.execute(task_context)
            
            # Extract final response (e.g. from last agent or validator)
            final_node = result_raw.get("results", {}).get("final_validator", {})
            response_text = str(final_node) # Simplification
            
            result = {
                "response": response_text,
                "workflow_id": result_raw.get("workflow_id", "dev_unknown"),
                "tokens_used": sum([r.get("tokens_used", 0) for r in result_raw.get("results", {}).values() if isinstance(r, dict)]),
                "validation_passed": final_node.get("is_valid", True) if isinstance(final_node, dict) else True
            }
        else:
            result = await chat_workflow.execute_chat(
                user_id=request.user_id,
                query=request.query
            )
        
        # Calculate execution time
        execution_time = (datetime.now(UTC) - start_time).total_seconds()
        
        # Log the workflow execution
        log_data = {
            'log_id': f"log_{request.user_id}_{datetime.now(UTC).timestamp()}",
            'workflow_id': result["workflow_id"],
            'log_level': 'info',
            'message': f"Chat completed for user {request.user_id}",
            'context_snapshot': {
                'query': request.query,
                'response_length': len(result["response"]),
                'execution_time': execution_time
            }
        }
        await database_manager.log_observability_event(log_data)
        
        return ChatResponse(
            response=result["response"],
            workflow_id=result["workflow_id"],
            tokens_used=result["tokens_used"],
            validation_passed=result["validation_passed"],
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/v1/chat/stream")
async def chat_endpoint_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SE)"""
    try:
        # Validate input for security issues
        validation_result = await security_validator.validate_input(request.query)
        if validation_result['has_critical']:
            raise HTTPException(status_code=400, detail="Input contains security violations")
        
        # Simple permission check
        user = await database_manager.get_user(request.user_id)
        if not user:
             raise HTTPException(status_code=404, detail="User not found")

        async def event_generator():
            try:
                async for event in chat_workflow.execute_chat_stream(request.user_id, request.query):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Chat stream endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")


@app.post("/api/v1/workflow/define")
async def define_workflow(definition: WorkflowDefinition):
    """Endpoint to define new workflows programmatically"""
    try:
        # Clear existing nodes for this workflow
        workflow_engine.nodes = {}
        
        # Add nodes from the definition
        for node_data in definition.nodes:
            node = WorkflowNode(**node_data)
            workflow_engine.add_node(node)
        
        # Add edges
        for edge in definition.edges:
            workflow_engine.add_edge(edge["from"], edge["to"])
        
        return {
            "status": "success",
            "workflow_id": definition.workflow_id,
            "nodes_added": len(definition.nodes)
        }
    except Exception as e:
        logger.error(f"Workflow definition error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow definition failed: {str(e)}")


@app.get("/api/v1/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get the status of a specific workflow execution"""
    # In a real implementation, this would track running workflows
    # For now, we'll return a simulated response
    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "progress": 100,
        "completed_nodes": ["router", "context_builder", "llm_worker", "validator"],
        "failed_nodes": [],
        "execution_time": 0.5
    }


@app.post("/api/v1/context/build")
async def build_context(request: ChatRequest):
    """Build context without executing a workflow"""
    try:
        # Validate input for security issues
        validation_result = await security_validator.validate_input(request.query)
        if validation_result['has_critical']:
            raise HTTPException(status_code=400, detail="Input contains security violations")
            
        task_context = await context_builder.build_task_context(
            user_id=request.user_id,
            session_id=request.session_id or f"session_{datetime.now(UTC).timestamp()}",
            workflow_id=f"context_build_{datetime.now(UTC).timestamp()}",
            workflow_name=request.workflow_type,
            initiating_query=request.query,
            task_type=TaskType.CHAT,
            additional_context={"user_preferences": request.user_preferences or {}}
        )
        
        validation_result = await validator_pipeline.validate_task_context(task_context)
        
        return {
            "context_built": True,
            "context_size": await context_builder.get_context_size(task_context),
            "validation_result": validation_result.model_dump(),
            "context_summary": {
                "user_id": task_context.system_context.user_id,
                "workflow_id": task_context.workflow_context.workflow_id,
                "initiating_query": task_context.workflow_context.initiating_query[:100] + "..."
            }
        }
    except Exception as e:
        logger.error(f"Context building error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context building failed: {str(e)}")


@app.get("/api/v1/context/schema")
async def get_context_schema():
    """Get the schema for context objects"""
    return {
        "system_context": "System-level context including hardware state, budget, and user preferences",
        "workflow_context": "Workflow-specific context including user intent and budget tracking",
        "node_context": "Node-specific execution context",
        "tool_context": "Tool availability and permission context",
        "task_context": "The 'Golden Context' packet combining all context types"
    }


@app.get("/api/v1/templates")
async def list_workflow_templates(category: Optional[str] = None):
    """List available workflow templates"""
    templates = workflow_composer.list_templates(category)
    return {
        "templates": [
            {
                "template_id": t.template_id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "complexity": t.complexity,
                "estimated_duration": t.estimated_duration,
                "required_capabilities": t.required_capabilities,
                "input_schema": t.input_schema,
                "output_schema": t.output_schema
            }
            for t in templates
        ],
        "total_count": len(templates),
        "categories": list(set(t.category for t in templates))
    }


@app.get("/api/v1/templates/{template_id}")
async def get_workflow_template(template_id: str):
    """Get details of a specific workflow template"""
    template = workflow_composer.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return {
        "template_id": template.template_id,
        "name": template.name,
        "description": template.description,
        "version": template.version,
        "category": template.category,
        "complexity": template.complexity,
        "estimated_duration": template.estimated_duration,
        "nodes": template.nodes,
        "edges": template.edges,
        "input_schema": template.input_schema,
        "output_schema": template.output_schema,
        "required_capabilities": template.required_capabilities,
        "validation_rules": template.validation_rules,
        "author": template.author,
        "created_at": template.created_at.isoformat() if template.created_at else None,
        "last_validated": template.last_validated.isoformat() if template.last_validated else None
    }


@app.post("/api/v1/templates/compose")
async def compose_workflow_from_templates(composition_spec: Dict[str, Any]):
    """Compose a workflow from templates"""
    try:
        # Validate user permissions (would be more sophisticated in production)
        user_id = composition_spec.get("user_id", "system")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        # Create task context for composition
        from .ai.context.schemas import (
            SystemContext, WorkflowContext, UserIntent, TaskType,
            HardwareState, BudgetState, UserPreferences, ContextBudget
        )

        context = TaskContext(
            system_context=SystemContext(
                user_id=user_id,
                session_id=f"compose_{datetime.now(UTC).timestamp()}",
                hardware_state=HardwareState(gpu_usage=0, memory_available_gb=16, cpu_usage=10, current_load=0),
                budget_state=BudgetState(remaining_pct=100.0),
                user_preferences=UserPreferences()
            ),
            workflow_context=WorkflowContext(
                workflow_id=f"composed_{datetime.now(UTC).timestamp()}",
                workflow_name=composition_spec.get("name", "composed_workflow"),
                initiating_query="Workflow composition request",
                user_intent=UserIntent(type=TaskType.CHAT, confidence=0.9, description="Compose workflow", priority=1),
                context_budget=ContextBudget()
            )
        )

        # Compose the workflow
        composed_engine = await workflow_composer.compose_workflow(composition_spec, context)

        if not composed_engine:
            raise HTTPException(status_code=400, detail="Workflow composition failed")

        # Return composition result
        return {
            "status": "success",
            "workflow_id": context.workflow_context.workflow_id,
            "composed_engine": {
                "node_count": len(composed_engine.nodes),
                "nodes": list(composed_engine.nodes.keys()),
                "template_instances": len(composition_spec.get("templates", []))
            },
            "composition_spec": composition_spec
        }

    except Exception as e:
        logger.error(f"Workflow composition error: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow composition failed: {str(e)}")


@app.post("/api/v1/templates/execute/{workflow_id}")
async def execute_composed_workflow(workflow_id: str, execution_request: Dict[str, Any]):
    """Execute a previously composed workflow"""
    try:
        user_id = execution_request.get("user_id", "system")
        parameters = execution_request.get("parameters", {})

        # In a production system, we'd store composed workflows and retrieve them
        # For now, this is a placeholder that would need the composition to be passed

        # This endpoint would execute composed workflows
        # Implementation would depend on how composed workflows are stored/retrieved

        return {
            "status": "not_implemented",
            "message": "Composed workflow execution not yet implemented",
            "workflow_id": workflow_id
        }

    except Exception as e:
        logger.error(f"Composed workflow execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.post("/api/v1/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file"""
    try:
        content = await file.read()
        text, confidence = await voice_service.speech_to_text(content)
        # Validate the transcribed text for security issues
        validation_result = await security_validator.validate_input(text)
        if validation_result['has_critical']:
            raise HTTPException(status_code=400, detail="Transcribed text contains security violations")
        return {"text": text, "confidence": confidence}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/speak")
async def text_to_speech(request: Dict[str, str]):
    """Convert text to speech"""
    text = request.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text required")
    # Validate the text for security issues
    validation_result = await security_validator.validate_input(text)
    if validation_result['has_critical']:
        raise HTTPException(status_code=400, detail="Text contains security violations")
    try:
        audio = await voice_service.text_to_speech(text)
        return Response(content=audio, media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/voice/session", response_model=VoiceSessionResponse)
async def voice_session(request: VoiceSessionRequest):
    """
    Unified voice session endpoint:
    1. Transcribe audio (STT)
    2. Execute Chat Workflow
    3. Synthesize response (TTS)
    """
    import base64
    
    try:
        # 1. Decode Audio
        try:
            audio_bytes = base64.b64decode(request.audio_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")

        # 2. STT & Wake Word
        # Note: In a real "loop", client might handle wake word, but we check here if needed
        # For this endpoint, we assume client detected wake word or push-to-talk was used
        # unless we want to implement server-side wake word check.
        # Let's perform STT first.
        transcription, confidence = await voice_service.speech_to_text(audio_bytes)
        
        if not transcription or len(transcription.strip()) < 2:
            return VoiceSessionResponse(
                conversation_id=request.conversation_id or "new",
                text_response="",
                audio_data=None,
                detected=False,
                workflow_id="none"
            )

        # 3. Chat Workflow
        # Use a default user for headless voice if not authenticated
        user_id = "voice_user" 
        
        # Ensure user exists (optional, or use admin)
        
        result = await chat_workflow.execute_chat(
            user_id=user_id,
            query=transcription,
            conversation_id=request.conversation_id
        )
        
        response_text = result["response"]
        
        # 4. TTS
        audio_response_b64 = None
        if response_text:
            try:
                audio_response = await voice_service.text_to_speech(response_text)
                if audio_response:
                    audio_response_b64 = base64.b64encode(audio_response).decode("utf-8")
            except Exception as e:
                logger.error(f"Voice session TTS failed: {e}")
                # We return text even if audio fails
        
        return VoiceSessionResponse(
            conversation_id=result.get("conversation_id", request.conversation_id or "new"),
            text_response=response_text,
            audio_data=audio_response_b64,
            detected=True,
            workflow_id=result["workflow_id"]
        )

    except Exception as e:
        logger.error(f"Voice session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return {"error": "Internal server error", "detail": str(exc)}


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # In production, set to False
        log_level="info"
    )

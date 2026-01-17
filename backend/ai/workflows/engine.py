"""
Workflow Engine for JARVISv3 - implementing the "Agentic Graph" architecture
"""
import asyncio
import httpx
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, AsyncIterable
from pydantic import BaseModel, Field
from datetime import datetime, UTC
from dataclasses import dataclass
import logging
from ..context.schemas import TaskContext, NodeContext, SystemContext, WorkflowContext, TaskType, UserIntent, ContextBudget

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Status of a workflow node"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(str, Enum):
    """Types of workflow nodes"""
    ROUTER = "router"
    CONTEXT_BUILDER = "context_builder"
    LLM_WORKER = "llm_worker"
    AGENT_WORKER = "agent_worker"
    VALIDATOR = "validator"
    TOOL_CALL = "tool_call"
    HUMAN_APPROVAL = "human_approval"
    SEARCH_WEB = "search_web"
    REFLECTOR = "reflector"
    SUPERVISOR = "supervisor"
    ACTIVE_MEMORY = "active_memory"
    END = "end"


class ApprovalStatus(str, Enum):
    """Status of human approval requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """Represents a human approval request"""
    request_id: str
    workflow_id: str
    node_id: str
    user_id: str
    request_type: str
    context_data: Dict[str, Any]
    decision_criteria: Dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    decision_notes: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


class WorkflowApprovalRequiredException(Exception):
    """Exception raised when workflow execution requires human approval"""

    def __init__(self, approval_request: ApprovalRequest):
        self.approval_request = approval_request
        super().__init__(f"Workflow {approval_request.workflow_id} requires human approval for node {approval_request.node_id}")


class WorkflowNode(BaseModel):
    """Definition of a single workflow node"""
    id: str
    type: NodeType
    description: str
    dependencies: List[str] = []
    conditions: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 300  # 5 minutes default
    retry_count: int = 3
    max_retries: int = 3
    parallel_execution: bool = False
    
    # Function to execute when node runs
    execute_func: Optional[Callable] = None


class WorkflowState(BaseModel):
    """Current state of a workflow execution"""
    workflow_id: str
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    current_node: Optional[str] = None
    completed_nodes: List[str] = Field(default=[])
    failed_nodes: List[str] = Field(default=[])
    execution_history: List[Dict[str, Any]] = Field(default=[])
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 50  # Circuit breaker for cyclic loops
    plan_queue: List[Dict[str, Any]] = Field(default_factory=list)  # For dynamic planning

    # Context evolution tracking
    context_evolution: List[Dict[str, Any]] = Field(default_factory=list)  # Track context changes
    learning_patterns: Dict[str, Any] = Field(default_factory=dict)  # Learned patterns
    adaptation_count: int = 0  # How many times workflow adapted


class WorkflowEngine:
    """Main workflow engine implementing the agentic graph architecture"""
    
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.state: Optional[WorkflowState] = None
        self.context: Optional[TaskContext] = None
        self.node_results: Dict[str, Any] = {}
        
    def add_node(self, node: WorkflowNode):
        """Add a node to the workflow"""
        self.nodes[node.id] = node
        logger.info(f"Added node {node.id} to workflow")
    
    def add_edge(self, from_node: str, to_node: str, condition: Optional[str] = None):
        """Add an edge between nodes (simplified for now)"""
        if to_node not in self.nodes[from_node].dependencies:
            self.nodes[from_node].dependencies.append(to_node)
    
    async def execute_node(self, node_id: str) -> Dict[str, Any]:
        """Execute a single workflow node with retries and timeout"""
        from ...core.observability import workflow_tracer
        from ...core.node_registry import node_registry
        
        node = self.nodes[node_id]
        logger.info(f"Executing node {node_id} of type {node.type}")
        
        # Check if node should be executed remotely
        remote_node_id = node.conditions.get("remote_node_id") if node.conditions else None
        if remote_node_id:
             return await self._execute_remote_node(node_id, remote_node_id)
        
        # Update state
        if self.state:
            self.state.current_node = node_id
            self.state.status = NodeStatus.RUNNING
            workflow_tracer.trace_node_start(self.state.workflow_id, node_id, {"type": node.type})
        
        retries = 0
        last_error = None

        while retries <= node.max_retries:
            try:
                # Execute the node function with timeout
                if node.execute_func:
                    result = await asyncio.wait_for(
                        node.execute_func(self.context, self.node_results),
                        timeout=node.timeout_seconds
                    )
                else:
                    # Default execution based on node type
                    result = await self._execute_node_by_type(node)
                
                # Success path
                if self.state:
                    self.state.execution_history.append({
                        "node_id": node_id,
                        "status": NodeStatus.COMPLETED,
                        "result": result,
                        "timestamp": datetime.now(UTC).isoformat()
                    })
                    self.state.completed_nodes.append(node_id)
                    workflow_tracer.trace_node_end(self.state.workflow_id, node_id, result, success=True)
                
                self.node_results[node_id] = result
                logger.info(f"Node {node_id} completed successfully")
                
                # Create checkpoint
                await self.checkpoint(node_id)
                
                return result

            except Exception as e:
                last_error = e
                retries += 1
                if retries <= node.max_retries:
                    wait_time = (2 ** retries) * 0.5
                    logger.warning(f"Node {node_id} failed: {e}. Retrying in {wait_time}s... ({retries}/{node.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Node {node_id} failed after {node.max_retries} retries: {e}")

        # If we reached here, it failed
        error_msg = str(last_error) or type(last_error).__name__
        if self.state:
            workflow_tracer.trace_node_end(self.state.workflow_id, node_id, {"error": error_msg}, success=False)
        await self._handle_node_failure(node_id, error_msg)
        raise last_error if last_error else Exception(f"Node {node_id} failed")

    async def checkpoint(self, node_id: str):
        """Save current workflow state and context to database"""
        from ...core.database import database_manager
        if not self.state or not self.context:
            return

        checkpoint_data = {
            "checkpoint_id": f"cp_{self.state.workflow_id}_{node_id}_{datetime.now(UTC).timestamp()}",
            "workflow_id": self.state.workflow_id,
            "node_id": node_id,
            "state_data": self.state.model_dump(mode='json'),
            "context_data": self.context.model_dump(mode='json'),
            "results_data": self.node_results
        }
        await database_manager.save_workflow_checkpoint(checkpoint_data)
        logger.info(f"Checkpoint saved for node {node_id}")
    
    async def _execute_node_by_type(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a node based on its type"""
        if node.type == NodeType.ROUTER:
            return await self._execute_router_node(node)
        elif node.type == NodeType.CONTEXT_BUILDER:
            return await self._execute_context_builder_node(node)
        elif node.type == NodeType.LLM_WORKER:
            return await self._execute_llm_worker_node(node)
        elif node.type == NodeType.AGENT_WORKER:
            return await self._execute_agent_worker_node(node)
        elif node.type == NodeType.VALIDATOR:
            return await self._execute_validator_node(node)
        elif node.type == NodeType.TOOL_CALL:
            return await self._execute_tool_call_node(node)
        elif node.type == NodeType.HUMAN_APPROVAL:
            return await self._execute_human_approval_node(node)
        elif node.type == NodeType.SEARCH_WEB:
            return await self._execute_search_web_node(node)
        elif node.type == NodeType.REFLECTOR:
            return await self._execute_reflector_node(node)
        elif node.type == NodeType.SUPERVISOR:
            return await self._execute_supervisor_node(node)
        elif node.type == NodeType.ACTIVE_MEMORY:
            return await self._execute_active_memory_node(node)
        else:
            raise ValueError(f"Unknown node type: {node.type}")
    
    async def _execute_agent_worker_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute an agent worker node"""
        from .collaboration import agent_collaborator
        
        agent_id = str(node.conditions.get("agent_id", "default")) if node.conditions else "default"
        input_data = dict(node.conditions.get("agent_input", {})) if node.conditions else {}
        
        if self.context:
            return await agent_collaborator.execute_agent_step(agent_id, self.context, input_data)
            
        return {"error": "No context available"}
    
    async def _execute_router_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a router node that determines workflow path"""
        from .routing import router
        
        if self.context:
            return await router.route(self.context)
            
        return {"next_node": "context_builder", "route_decision": "chat", "confidence": 0.8}
    
    async def _execute_context_builder_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a context builder node"""
        # This would build context from various sources
        if self.context:
            # Update context with additional information
            self.context.additional_context["context_built"] = True
            return {
                "context_updated": True,
                "context_size": self.context.get_context_size(),
                "artifacts_created": ["context_packet"]
            }
        return {"context_updated": False, "artifacts_created": []}
    
    async def _execute_llm_worker_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute an LLM worker node"""
        from ...core.model_router import model_router
        
        # This would call the LLM with the context
        if self.context:
            prompt = self.context.workflow_context.initiating_query
            # In a real implementation, we would format the full context into a prompt
            # For now, just use the query
            
            try:
                result = await model_router.generate_response(prompt)
                
                self.context.update_tokens_consumed(result.tokens_used)
                
                return {
                    "response": result.response,
                    "tokens_used": result.tokens_used,
                    "model_used": "llama-local", # TODO: Get actual model name from router
                    "execution_time": result.execution_time
                }
            except Exception as e:
                logger.error(f"LLM execution failed: {e}")
                # Fallback or re-raise
                return {"error": str(e), "status": "failed"}
                
        return {"response": "No context available", "tokens_used": 0}

    async def execute_workflow_stream(self, initial_context: TaskContext) -> AsyncIterable[Dict[str, Any]]:
        """Execute the workflow and yield streaming events"""
        from ...core.observability import workflow_tracer
        from ...core.model_router import model_router
        
        logger.info(f"Starting streaming workflow execution")
        
        # Start trace
        workflow_tracer.start_trace(initial_context.workflow_context.workflow_id, initial_context)
        
        # Initialize state
        self.context = initial_context
        self.state = WorkflowState(
            workflow_id=initial_context.workflow_context.workflow_id,
            start_time=datetime.now(UTC)
        )
        
        try:
            # For now, we'll implement a simplified linear streaming for the chat workflow
            # router -> context_builder -> llm_worker (stream) -> validator
            
            # Node 1: Router
            yield {"type": "node_start", "node_id": "router"}
            route_res = await self._execute_router_node(self.nodes.get("router", WorkflowNode(id="router", type=NodeType.ROUTER, description="")))
            self.node_results["router"] = route_res
            yield {"type": "node_end", "node_id": "router", "result": route_res}
            
            # Node 2: Context Builder
            yield {"type": "node_start", "node_id": "context_builder"}
            ctx_res = await self._execute_context_builder_node(self.nodes.get("context_builder", WorkflowNode(id="context_builder", type=NodeType.CONTEXT_BUILDER, description="")))
            self.node_results["context_builder"] = ctx_res
            yield {"type": "node_end", "node_id": "context_builder", "result": ctx_res}
            
            # Node 3: LLM Worker (Streaming)
            yield {"type": "node_start", "node_id": "llm_worker"}
            full_response = ""
            async for chunk in model_router.generate_response_stream(self.context.workflow_context.initiating_query):
                full_response += chunk
                yield {"type": "stream_chunk", "node_id": "llm_worker", "chunk": chunk}
            
            llm_res = {
                "response": full_response,
                "tokens_used": len(full_response.split()), # Estimate
                "status": "completed"
            }
            self.node_results["llm_worker"] = llm_res
            yield {"type": "node_end", "node_id": "llm_worker", "result": llm_res}
            
            # Node 4: Validator
            yield {"type": "node_start", "node_id": "validator"}
            val_res = await self._execute_validator_node(self.nodes.get("validator", WorkflowNode(id="validator", type=NodeType.VALIDATOR, description="")))
            self.node_results["validator"] = val_res
            yield {"type": "node_end", "node_id": "validator", "result": val_res}
            
            # Final result
            self.state.status = NodeStatus.COMPLETED
            self.state.end_time = datetime.now(UTC)
            yield {"type": "workflow_completed", "workflow_id": self.state.workflow_id, "final_response": full_response}
            
        except Exception as e:
            logger.error(f"Streaming workflow failed: {e}")
            yield {"type": "error", "message": str(e)}
    
    async def _execute_validator_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a validation node"""
        # Validate the results from previous nodes
        validation_errors = []
        
        if self.context:
            validation_errors = self.context.validate_context()
        
        is_valid = len(validation_errors) == 0
        return {
            "is_valid": is_valid,
            "errors": validation_errors,
            "validation_passed": is_valid
        }
    
    async def _execute_tool_call_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a tool call node using MCP servers"""
        from ...mcp_servers.base_server import mcp_dispatcher
        
        tool_name = str(node.conditions.get("tool_name", "")) if node.conditions else ""
        tool_input = dict(node.conditions.get("tool_input", {})) if node.conditions else {}
        
        if not tool_name:
             return {"success": False, "error": "No tool_name specified in node conditions"}

        try:
            result = await mcp_dispatcher.call_tool(tool_name, tool_input)
            
            if self.context and self.context.tool_context:
                self.context.tool_context.add_tool_output(tool_name, result)
                
            return {
                "tool_results": [result],
                "tools_called": [tool_name],
                "success": True
            }
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_human_approval_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a human approval node with workflow pausing"""
        if not self.context or not self.state:
            return {"error": "No context or state available"}

        # Extract approval criteria from node conditions
        request_type = node.conditions.get("request_type", "general") if node.conditions else "general"
        decision_criteria = node.conditions.get("decision_criteria", {}) if node.conditions else {}
        timeout_seconds = node.conditions.get("timeout_seconds", 300) if node.conditions else 300  # 5 minutes default

        # Determine if human approval is needed based on criteria
        needs_approval = await self._evaluate_approval_criteria(request_type, decision_criteria)

        if not needs_approval:
            # Auto-approve based on criteria
            return {
                "approved": True,
                "auto_approved": True,
                "reason": "met_auto_approval_criteria",
                "approval_time": datetime.now(UTC).isoformat(),
                "approver_id": "system"
            }

        # Create approval request
        approval_request = ApprovalRequest(
            request_id=f"apr_{self.state.workflow_id}_{node.id}_{datetime.now(UTC).timestamp()}",
            workflow_id=self.state.workflow_id,
            node_id=node.id,
            user_id=self.context.system_context.user_id,
            request_type=request_type,
            context_data={
                "workflow_name": self.context.workflow_context.workflow_name,
                "initiating_query": self.context.workflow_context.initiating_query,
                "node_results": self.node_results,
                "current_node": node.id
            },
            decision_criteria=decision_criteria
        )

        # Store approval request (in a real system, this would be in a database)
        # For now, we'll simulate by pausing execution
        logger.info(f"Created approval request {approval_request.request_id} for workflow {self.state.workflow_id}")

        # Pause workflow execution by raising a special exception
        raise WorkflowApprovalRequiredException(approval_request)

    async def _evaluate_approval_criteria(self, request_type: str, criteria: Dict[str, Any]) -> bool:
        """Evaluate if human approval is needed based on criteria"""
        if not self.context or not self.state:
            return True  # Default to requiring approval if no context

        # High-stakes operations always require approval
        if request_type in ["code_deployment", "security_change", "data_deletion", "financial_transaction"]:
            return True

        # Check confidence thresholds
        confidence_threshold = criteria.get("confidence_threshold", 0.8)
        if "llm_worker" in self.node_results:
            llm_result = self.node_results["llm_worker"]
            # In a real system, we'd have confidence scores
            # For now, assume approval needed for uncertain operations
            if llm_result.get("tokens_used", 0) > 1000:  # Large responses might need review
                return True

        # Check for security violations
        if "validator" in self.node_results:
            validator_result = self.node_results["validator"]
            if not validator_result.get("validation_passed", True):
                return True

        # Check for high-risk content patterns
        query = self.context.workflow_context.initiating_query.lower()
        risk_patterns = ["delete", "drop", "remove", "uninstall", "format", "reset"]
        if any(pattern in query for pattern in risk_patterns):
            return True

        # Default: auto-approve for low-risk operations
        return False

    def resume_workflow_after_approval(self, approval_request: ApprovalRequest) -> Dict[str, Any]:
        """Resume workflow execution after approval decision"""
        if approval_request.status == ApprovalStatus.APPROVED:
            logger.info(f"Workflow {approval_request.workflow_id} approved by {approval_request.decision_notes}")
            return {
                "approved": True,
                "approval_time": approval_request.decided_at.isoformat() if approval_request.decided_at else None,
                "approver_id": approval_request.decision_notes or "unknown",
                "resume_workflow": True
            }
        elif approval_request.status == ApprovalStatus.REJECTED:
            logger.info(f"Workflow {approval_request.workflow_id} rejected by {approval_request.decision_notes}")
            return {
                "approved": False,
                "rejection_reason": approval_request.decision_notes,
                "approval_time": approval_request.decided_at.isoformat() if approval_request.decided_at else None,
                "resume_workflow": False
            }
        elif approval_request.status == ApprovalStatus.TIMEOUT:
            logger.warning(f"Approval request {approval_request.request_id} timed out")
            return {
                "approved": False,
                "reason": "timeout",
                "resume_workflow": False
            }
        else:
            return {"error": f"Unknown approval status: {approval_request.status}"}

    async def _execute_search_web_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a web search node"""
        from .search_node import search_node
        if self.context:
            return await search_node.execute(self.context, self.node_results)
        return {"success": False, "error": "No context available"}

    async def _execute_reflector_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a reflector node"""
        from .reflector import reflector_node
        
        target_node_id = str(node.conditions.get("target_node_id", "")) if node.conditions else ""
        criteria = str(node.conditions.get("criteria", "")) if node.conditions else ""
        
        if not target_node_id:
             return {"next_node": None, "error": "No target_node_id specified"}

        if self.context:
            return await reflector_node.execute(self.context, self.node_results, target_node_id, criteria)
            
        return {"next_node": None, "error": "No context available"}

    async def _execute_supervisor_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a supervisor node"""
        from .supervisor import supervisor_node
        
        if self.context:
            # Execute supervisor logic
            result = await supervisor_node.execute(self.context)
            
            # Update plan queue in state
            if self.state and "plan" in result:
                self.state.plan_queue.extend(result["plan"])
                logger.info(f"Supervisor updated plan queue with {len(result['plan'])} steps")
                
                # If plan has items, return the first one as next_node to kickstart
                if self.state.plan_queue:
                    next_step = self.state.plan_queue.pop(0)
                    if isinstance(next_step, str):
                        return {"next_node": next_step}
                    elif isinstance(next_step, dict):
                        return {"next_node": next_step.get("node_id")}
            
            return result
            
        return {"error": "No context available"}

    async def _execute_active_memory_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute an active memory node with context evolution"""
        from .active_memory import active_memory_node

        operation = str(node.conditions.get("operation", "")) if node.conditions else ""
        content = node.conditions.get("content") if node.conditions else None
        query = str(node.conditions.get("query", "")) if node.conditions else ""

        if not self.context:
            return {"error": "No context available"}

        # Execute memory operation
        result = await active_memory_node.execute(self.context, operation, content, query)

        # Context evolution: Learn from this memory operation
        if self.state:
            await self._evolve_context_from_memory(operation, result)

        return result

    async def _evolve_context_from_memory(self, operation: str, result: Dict[str, Any]):
        """Evolve context based on memory operations"""
        if not self.context or not self.state:
            return

        evolution_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "node_id": self.state.current_node,
            "changes": []
        }

        # Learn patterns based on memory operations
        if operation == "store" and result.get("status") == "stored":
            # Learned: This workflow stores information for future use
            pattern_key = f"stores_{self.context.workflow_context.workflow_name}"
            if pattern_key not in self.state.learning_patterns:
                self.state.learning_patterns[pattern_key] = 0
            self.state.learning_patterns[pattern_key] += 1

            # Evolve context: Add memory capability awareness
            if "memory_capable" not in self.context.additional_context:
                self.context.additional_context["memory_capable"] = True
                evolution_entry["changes"].append("added_memory_capability")

        elif operation == "retrieve" and result.get("status") == "retrieved":
            # Learned: This workflow uses historical context
            pattern_key = f"retrieves_{self.context.workflow_context.workflow_name}"
            if pattern_key not in self.state.learning_patterns:
                self.state.learning_patterns[pattern_key] = 0
            self.state.learning_patterns[pattern_key] += 1

            # Evolve context: Enhance with retrieved information
            retrieved_data = result.get("results", [])
            if retrieved_data:
                # Add retrieved context to workflow context
                if "retrieved_memories" not in self.context.additional_context:
                    self.context.additional_context["retrieved_memories"] = []
                self.context.additional_context["retrieved_memories"].extend(retrieved_data)
                evolution_entry["changes"].append(f"added_{len(retrieved_data)}_memories")

        # Record evolution if any changes occurred
        if evolution_entry["changes"]:
            self.state.context_evolution.append(evolution_entry)
            self.state.adaptation_count += 1
            logger.info(f"Context evolved: {evolution_entry['changes']} (adaptations: {self.state.adaptation_count})")

    def adapt_workflow_from_patterns(self) -> Dict[str, Any]:
        """Adapt workflow behavior based on learned patterns"""
        if not self.state:
            return {"adapted": False, "reason": "no_state"}

        adaptations = []

        # Check for memory-intensive workflows
        memory_patterns = [k for k in self.state.learning_patterns.keys() if "stores_" in k or "retrieves_" in k]
        if len(memory_patterns) > 1:  # Lower threshold for pattern detection
            adaptations.append("memory_intensive_workflow")
            # Could add memory optimization nodes here

        # Check for repetitive patterns
        repetitive_ops = sum(self.state.learning_patterns.values())
        if repetitive_ops > 5:
            adaptations.append("repetitive_operations")
            # Could suggest workflow optimization

        if adaptations:
            logger.info(f"Workflow adapted based on patterns: {adaptations}")
            return {
                "adapted": True,
                "adaptations": adaptations,
                "pattern_count": len(self.state.learning_patterns)
            }

        return {"adapted": False, "reason": "insufficient_patterns"}

    async def _execute_remote_node(self, node_id: str, remote_node_id: str) -> Dict[str, Any]:
        """Proxy node execution to a remote JARVISv3 instance"""
        from ...core.node_registry import node_registry
        
        async with node_registry._lock:
            remote_node = node_registry.nodes.get(remote_node_id)
            
        if not remote_node:
            raise ValueError(f"Remote node {remote_node_id} not found")
            
        logger.info(f"Proxying node {node_id} to remote node {remote_node.name} ({remote_node_id})")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{remote_node.base_url}/api/v1/distributed/execute",
                    json={
                        "context": self.context.model_dump(mode='json') if self.context else {},
                        "node_id": node_id
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "success":
                    # Update local context with the one returned from remote
                    if self.context and data.get("updated_context"):
                        from ..context.schemas import TaskContext
                        self.context = TaskContext(**data["updated_context"])
                    return data.get("result", {})
                else:
                    raise Exception(f"Remote execution failed: {data.get('detail')}")
                    
        except Exception as e:
            logger.error(f"Error during remote node proxying: {e}")
            raise
    
    async def _handle_node_failure(self, node_id: str, error_msg: str):
        """Handle node execution failure"""
        if self.state:
            self.state.failed_nodes.append(node_id)
            self.state.status = NodeStatus.FAILED
            self.state.error_message = error_msg
            self.state.execution_history.append({
                "node_id": node_id,
                "status": NodeStatus.FAILED,
                "error": error_msg,
                "timestamp": datetime.now(UTC).isoformat()
            })
    
    async def execute_workflow(self, initial_context: TaskContext) -> Dict[str, Any]:
        """Execute the complete workflow"""
        from ...core.observability import workflow_tracer
        logger.info(f"Starting workflow execution with context")
        
        # Start trace
        workflow_tracer.start_trace(initial_context.workflow_context.workflow_id, initial_context)
        
        # Initialize state
        self.context = initial_context
        self.state = WorkflowState(
            workflow_id=initial_context.workflow_context.workflow_id,
            start_time=datetime.now(UTC)
        )
        
        try:
            # Find the starting node (nodes with no dependencies)
            start_nodes = [node_id for node_id, node in self.nodes.items() 
                          if not node.dependencies]
            
            if not start_nodes:
                raise ValueError("No starting nodes found in workflow")
            
            # Smart Start: Prioritize Supervisor for dynamic workflows
            supervisor_nodes = [nid for nid in start_nodes if self.nodes[nid].type == NodeType.SUPERVISOR]
            
            if supervisor_nodes:
                # If a Supervisor exists, it drives the process as a State Machine
                await self._execute_cyclic_state_machine(supervisor_nodes[0])
            elif len(start_nodes) == 1:
                # Single entry point, safe to use State Machine
                await self._execute_cyclic_state_machine(start_nodes[0])
            else:
                # Fallback to legacy recursive DAG execution for multi-start graphs
                for node_id in start_nodes:
                    await self._execute_node_recursive(node_id)
            
            # Complete workflow
            final_res = {
                "status": "completed",
                "workflow_id": self.state.workflow_id if self.state else None,
                "execution_time": 0.0,
                "results": self.node_results
            }
            if self.state:
                self.state.status = NodeStatus.COMPLETED
                self.state.end_time = datetime.now(UTC)
                final_res["execution_time"] = (self.state.end_time - self.state.start_time).total_seconds()
                workflow_tracer.end_trace(self.state.workflow_id, self.state, final_res)
            
            logger.info("Workflow completed successfully")
            return final_res
            
        except Exception as e:
            error_msg = str(e) or type(e).__name__
            logger.error(f"Workflow failed: {error_msg}")
            if self.state:
                self.state.status = NodeStatus.FAILED
                self.state.error_message = error_msg
                self.state.end_time = datetime.now(UTC)
            
            return {
                "status": "failed",
                "error": error_msg,
                "workflow_id": self.state.workflow_id if self.state else None,
                "execution_time": (self.state.end_time - self.state.start_time).total_seconds() if self.state and self.state.end_time else 0
            }
    
    async def _execute_node_recursive(self, node_id: str):
        """Recursively execute nodes based on dependencies (Legacy DAG mode)"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in workflow")
        
        node = self.nodes[node_id]
        
        # Execute current node
        result = await self.execute_node(node_id)
        
        # Find next nodes that depend on this one
        next_nodes = [nid for nid, n in self.nodes.items() 
                     if node_id in n.dependencies]
        
        # Execute next nodes
        for next_node_id in next_nodes:
            await self._execute_node_recursive(next_node_id)

    async def _execute_cyclic_state_machine(self, start_node_id: str):
        """Execute workflow as a Cyclic State Machine"""
        if not self.state:
            raise ValueError("Workflow state not initialized")
            
        current_node_id = start_node_id
        
        while current_node_id:
            # Check circuit breaker
            if self.state.iteration_count >= self.state.max_iterations:
                raise Exception(f"Workflow exceeded maximum iterations ({self.state.max_iterations}). Possible infinite loop.")
            
            self.state.iteration_count += 1
            
            # Execute the node
            result = await self.execute_node(current_node_id)
            
            # Determine next node
            
            # 1. Dynamic 'next_node' from result (Router pattern)
            next_via_router = None
            if isinstance(result, dict) and "next_node" in result:
                next_via_router = result["next_node"]
                if next_via_router and next_via_router.lower() == "end":
                    next_via_router = None # Stop or fallback
                elif next_via_router and next_via_router not in self.nodes:
                     raise ValueError(f"Router sent to unknown node: {next_via_router}")
            
            if next_via_router:
                current_node_id = next_via_router
                continue

            # 2. Static dependency (DAG pattern)
            # Find nodes that depend on the CURRENT node
            next_static_nodes = [nid for nid, n in self.nodes.items() 
                         if current_node_id in n.dependencies]
            
            if next_static_nodes:
                if len(next_static_nodes) > 1:
                    logger.warning(f"Ambiguous path from {current_node_id}: {next_static_nodes}. Taking {next_static_nodes[0]}.")
                current_node_id = next_static_nodes[0]
                continue

            # 3. Plan Queue (Dynamic pattern) - Only if no other path
            logger.info(f"Checking plan queue {id(self.state.plan_queue)}. Size: {len(self.state.plan_queue)}. Content: {self.state.plan_queue}")
            if self.state.plan_queue:
                next_step = self.state.plan_queue.pop(0)
                logger.info(f"Popped from plan queue: {next_step}")
                if isinstance(next_step, str) and next_step in self.nodes:
                    current_node_id = next_step
                    continue
                elif isinstance(next_step, dict) and "node_id" in next_step:
                     if next_step["node_id"] in self.nodes:
                         current_node_id = next_step["node_id"]
                         continue
            
            # If we get here, no next node found
            current_node_id = None
    


# Example usage and test functions
async def test_workflow_engine():
    """Test function to demonstrate the workflow engine"""
    engine = WorkflowEngine()
    
    # Add nodes to the workflow
    engine.add_node(WorkflowNode(
        id="router",
        type=NodeType.ROUTER,
        description="Route to appropriate workflow based on intent"
    ))
    
    engine.add_node(WorkflowNode(
        id="context_builder",
        type=NodeType.CONTEXT_BUILDER,
        description="Build context from various sources",
        dependencies=["router"]
    ))
    
    engine.add_node(WorkflowNode(
        id="llm_worker",
        type=NodeType.LLM_WORKER,
        description="Process with LLM",
        dependencies=["context_builder"]
    ))
    
    engine.add_node(WorkflowNode(
        id="validator",
        type=NodeType.VALIDATOR,
        description="Validate output",
        dependencies=["llm_worker"]
    ))
    
    # Create a test context
    from ..context.schemas import (
        SystemContext, WorkflowContext, UserIntent, TaskType,
        HardwareState, BudgetState, UserPreferences, ContextBudget
    )
    
    test_context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="test_session",
            hardware_state=HardwareState(
                gpu_usage=0.0,
                memory_available_gb=16.0,
                cpu_usage=20.0,
                current_load=0.1
            ),
            budget_state=BudgetState(
                cloud_spend_usd=0.0,
                monthly_limit_usd=100.0,
                remaining_pct=100.0
            ),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="test_workflow",
            workflow_name="test_workflow",
            initiating_query="Hello, how are you?",
            user_intent=UserIntent(
                type=TaskType.CHAT,
                confidence=0.9,
                description="Simple chat query",
                priority=3
            ),
            context_budget=ContextBudget()
        )
    )
    
    # Execute the workflow
    result = await engine.execute_workflow(test_context)
    print(f"Workflow result: {result}")
    return result


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_workflow_engine())

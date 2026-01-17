"""
Full-Stack Development Workflow for JARVISv3
Demonstrates multi-agent collaboration for complex engineering tasks.
"""
from .engine import WorkflowEngine, WorkflowNode, NodeType
from .agent_registry import agent_registry
from ..context.schemas import TaskContext, TaskType
import logging

logger = logging.getLogger(__name__)

class DevWorkflow:
    """
    Coordinates multiple specialized agents to perform full-stack development tasks.
    """
    
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
        self._setup_workflow()
        
    def _setup_workflow(self):
        """Define the multi-agent DAG for development"""
        
        # 1. Requirements Extraction Node
        self.engine.add_node(WorkflowNode(
            id="extract_requirements",
            type=NodeType.AGENT_WORKER,
            description="Extracts and clarifies user requirements",
            conditions={"agent_id": "requirements_analyst"}
        ))
        
        # 2. Architectural Design Node
        self.engine.add_node(WorkflowNode(
            id="design_architecture",
            type=NodeType.AGENT_WORKER,
            description="Designs system architecture based on requirements",
            dependencies=["extract_requirements"],
            conditions={"agent_id": "software_architect"}
        ))
        
        # 3. Implementation Node
        self.engine.add_node(WorkflowNode(
            id="implement_code",
            type=NodeType.AGENT_WORKER,
            description="Implements the design in code",
            dependencies=["design_architecture"],
            conditions={"agent_id": "senior_coder"}
        ))
        
        # 4. Security Audit Node
        self.engine.add_node(WorkflowNode(
            id="security_audit",
            type=NodeType.AGENT_WORKER,
            description="Audits the implementation for security issues",
            dependencies=["implement_code"],
            conditions={"agent_id": "security_auditor"}
        ))
        
        # 5. Validator Node (Check Security + Implementation)
        self.engine.add_node(WorkflowNode(
            id="validator",
            type=NodeType.VALIDATOR,
            description="Validates code functionality and security findings",
            dependencies=["security_audit"]
        ))
        
        # 6. Reflector Node (Cyclic Logic)
        self.engine.add_node(WorkflowNode(
            id="reflector",
            type=NodeType.REFLECTOR,
            description="Checks validation and triggers self-correction loop",
            dependencies=["validator"],
            conditions={
                "target_node_id": "implement_code", 
                "criteria": "Code must pass security audit and linting"
            }
        ))
        
    async def execute(self, context: TaskContext) -> dict:
        """Execute the full development workflow"""
        logger.info(f"Starting Full-Stack Dev workflow for query: {context.workflow_context.initiating_query}")
        
        # Ensure context task type is correct
        context.workflow_context.user_intent.type = TaskType.CODING
        
        return await self.engine.execute_workflow(context)

# Global instance initialization helper
def init_dev_workflow(engine: WorkflowEngine) -> DevWorkflow:
    return DevWorkflow(engine)

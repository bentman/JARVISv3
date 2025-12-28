"""
Research Workflow Implementation for JARVISv3
Implements a workflow that includes web search.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, UTC

from ..context.schemas import TaskContext, TaskType
from .engine import WorkflowEngine, WorkflowNode, NodeType
from ..generators.context_builder import ContextBuilder
from ..validators.code_check import ValidatorPipeline

logger = logging.getLogger(__name__)

class ResearchWorkflow:
    """Implements a research workflow with web search"""
    
    def __init__(self):
        self.engine = WorkflowEngine()
        self.context_builder = ContextBuilder()
        self.validator = ValidatorPipeline()
        
        self._setup_workflow()
        
    def _setup_workflow(self):
        """Set up the research workflow nodes"""
        # 1. Router (Intent already checked, but needed for entry)
        self.engine.add_node(WorkflowNode(
            id="router",
            type=NodeType.ROUTER,
            description="Entry point"
        ))
        
        # 2. Search Node
        self.engine.add_node(WorkflowNode(
            id="search_node",
            type=NodeType.SEARCH_WEB,
            description="Search the web for information",
            dependencies=["router"]
        ))
        
        # 3. Context Builder (Incorporate search results)
        self.engine.add_node(WorkflowNode(
            id="context_builder",
            type=NodeType.CONTEXT_BUILDER,
            description="Build context including search results",
            dependencies=["search_node"]
        ))
        
        # 4. LLM Worker (Synthesize answer)
        self.engine.add_node(WorkflowNode(
            id="llm_worker",
            type=NodeType.LLM_WORKER,
            description="Synthesize search results into an answer",
            dependencies=["context_builder"]
        ))
        
        # 5. Validator
        self.engine.add_node(WorkflowNode(
            id="validator",
            type=NodeType.VALIDATOR,
            description="Validate synthesized answer",
            dependencies=["llm_worker"]
        ))

    async def execute(self, user_id: str, query: str) -> Dict[str, Any]:
        """Execute the complete research workflow"""
        # Build initial task context
        task_context = await self.context_builder.build_task_context(
            user_id=user_id,
            session_id=f"session_{datetime.now(UTC).timestamp()}",
            workflow_id=f"research_{datetime.now(UTC).timestamp()}",
            workflow_name="research_workflow",
            initiating_query=query,
            task_type=TaskType.RESEARCH
        )
        
        # Execute
        result = await self.engine.execute_workflow(task_context)
        
        # Extract final answer
        llm_result = result.get("results", {}).get("llm_worker", {})
        
        return {
            "response": llm_result.get("response", "Research failed."),
            "workflow_id": task_context.workflow_context.workflow_id,
            "status": result.get("status"),
            "details": result
        }

research_workflow = ResearchWorkflow()

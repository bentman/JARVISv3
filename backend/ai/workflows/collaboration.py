"""
Multi-Agent Collaboration for JARVISv3
Orchestrates specialized agents within a workflow.
"""
import logging
from typing import Dict, Any, List, Optional
from ..context.schemas import TaskContext, NodeContext, AgentPersona
from .agent_registry import agent_registry
from ...core.model_router import model_router

logger = logging.getLogger(__name__)

class AgentCollaborator:
    """
    Manages the interaction and handoff between specialized agents.
    """
    
    async def execute_agent_step(self, agent_id: str, context: TaskContext, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single step for a specific agent.
        """
        agent_persona = agent_registry.get_agent(agent_id)
        if not agent_persona:
            raise ValueError(f"Agent {agent_id} not found in registry")
            
        logger.info(f"Executing agent step: {agent_persona.name} ({agent_id})")
        
        # 1. Update node context with agent persona
        if context.node_context:
            context.node_context.agent_persona = agent_persona
            context.node_context.agent_id = agent_id
            
        # 2. Prepare prompt with agent's system prompt
        # We combine system prompt, context artifacts, and input data
        artifacts = "\n".join([str(a) for a in context.workflow_context.accumulated_artifacts])
        
        prompt = f"""
        {agent_persona.system_prompt}
        
        Current Workflow History:
        {artifacts}
        
        Current Task Data:
        {input_data}
        
        Initiating Query:
        {context.workflow_context.initiating_query}
        
        Please provide your contribution based on your specialized role:
        """
        
        # 3. Select model and generate response
        # Using agent's preferred tier for model selection
        result = await model_router.generate_response(
            prompt=prompt,
            task_type="analysis", # Default for agent collaboration
            max_tokens=1000
        )
        
        # 4. Update context with results
        context.update_tokens_consumed(result.tokens_used)
        
        # Add contribution as artifact
        artifact_id = f"contribution_{agent_id}_{context.node_context.node_id if context.node_context else 'step'}"
        context.workflow_context.add_artifact(f"[{agent_persona.name}]: {result.response[:200]}...")
        
        return {
            "agent_id": agent_id,
            "agent_name": agent_persona.name,
            "response": result.response,
            "tokens_used": result.tokens_used,
            "execution_time": result.execution_time
        }

    async def orchestrate_handoff(self, from_agent: str, to_agent: str, context: TaskContext, handoff_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explicitly handoff control from one agent to another.
        """
        logger.info(f"Handoff from {from_agent} to {to_agent}")
        
        # Register handoff event in context
        context.workflow_context.metadata[f"handoff_{from_agent}_{to_agent}"] = {
            "timestamp": context.system_context.timestamp.isoformat(),
            "handoff_summary": str(handoff_data)[:100]
        }
        
        # Execute next agent step
        return await self.execute_agent_step(to_agent, context, handoff_data)

# Global instance
agent_collaborator = AgentCollaborator()

"""
Memory Context Generator for JARVISv3
"""
import logging
from typing import Dict, Any, Optional
from .base import ContextGenerator
from ..context.schemas import TaskContext
from ...core.memory import memory_service

logger = logging.getLogger(__name__)

class MemoryGenerator(ContextGenerator):
    """Contributes semantic memory and conversation history to the context"""
    
    @property
    def name(self) -> str:
        return "memory_generator"
        
    async def generate(self, context: TaskContext, **kwargs) -> TaskContext:
        try:
            query = context.workflow_context.initiating_query
            conversation_id = context.system_context.session_id # Assuming session_id is conversation_id
            
            memory_data = await memory_service.search_and_retrieve_context(
                query=query,
                conversation_id=conversation_id,
                max_messages=5
            )
            
            # Store in additional_context
            context.additional_context["memory"] = memory_data
            
            # Add to artifacts if needed
            if memory_data.get("semantic_matches"):
                context.workflow_context.add_artifact(f"memory_hits_{len(memory_data['semantic_matches'])}")
                
            logger.debug("Memory context added successfully")
        except Exception as e:
            logger.error(f"Error generating memory context: {e}")
            
        return context

"""
Context Builder for JARVISv3 - dynamically assembles context just-in-time
using pluggable generators and "Code-Driven Context" principles.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..context.schemas import (
    TaskContext, SystemContext, WorkflowContext, NodeContext, 
    HardwareState, BudgetState, UserPreferences, UserIntent, TaskType,
    ContextBudget, ToolContext
)
from .base import ContextGenerator
from .system_generators import HardwareGenerator, BudgetGenerator
from .memory_generator import MemoryGenerator
from .extraction_generator import StructuredExtractionGenerator

logger = logging.getLogger(__name__)

class ContextBuilder:
    """Builds context packets dynamically using pluggable generators"""
    
    def __init__(self):
        self.generators: List[ContextGenerator] = [
            HardwareGenerator(),
            BudgetGenerator(),
            MemoryGenerator(),
            StructuredExtractionGenerator()
        ]
        
    def register_generator(self, generator: ContextGenerator):
        """Register a new context generator"""
        self.generators.append(generator)
        logger.info(f"Registered context generator: {generator.name}")

    async def build_task_context(
        self,
        user_id: str,
        session_id: str,
        workflow_id: str,
        workflow_name: str,
        initiating_query: str,
        task_type: TaskType,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> TaskContext:
        """Build the complete task context packet using registered generators"""
        
        # 1. Initialize base context components
        user_intent = UserIntent(
            type=task_type,
            confidence=0.85,
            description=initiating_query[:100],
            priority=3
        )
        
        system_context = SystemContext(
            user_id=user_id,
            session_id=session_id,
            hardware_state=HardwareState(
                gpu_usage=0, memory_available_gb=0, cpu_usage=0, available_tiers=[], current_load=0
            ),
            budget_state=BudgetState(cloud_spend_usd=0, monthly_limit_usd=0, remaining_pct=100),
            user_preferences=UserPreferences(
                preferred_model="local",
                privacy_level="medium"
            )
        )
        
        workflow_context = WorkflowContext(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            initiating_query=initiating_query,
            user_intent=user_intent,
            context_budget=ContextBudget()
        )
        
        tool_context = ToolContext(
            tools_available=["search", "code_execution", "file_access"],
            permissions={}
        )
        
        task_context = TaskContext(
            system_context=system_context,
            workflow_context=workflow_context,
            tool_context=tool_context,
            additional_context=additional_context or {}
        )
        
        # 2. Run all registered generators
        for generator in self.generators:
            try:
                task_context = await generator.generate(task_context)
                logger.debug(f"Generator {generator.name} processed context")
            except Exception as e:
                logger.error(f"Error in generator {generator.name}: {e}")
        
        # 3. Final validation
        validation_errors = task_context.validate_context()
        if validation_errors:
            logger.warning(f"Context validation warnings: {validation_errors}")
            
        return task_context

    async def get_context_size(self, task_context: TaskContext) -> int:
        """Get the size of the context in bytes"""
        import json
        try:
            context_dict = task_context.model_dump(mode='json')
            return len(json.dumps(context_dict, default=str).encode('utf-8'))
        except Exception:
            return 0

    # Backward compatibility helpers
    async def build_context_from_template(
        self,
        template_name: str,
        template_params: Dict[str, Any]
    ) -> TaskContext:
        """Build context from a predefined template"""
        task_type_map = {
            "chat": TaskType.CHAT,
            "coding": TaskType.CODING,
            "research": TaskType.RESEARCH
        }
        
        return await self.build_task_context(
            user_id=template_params.get("user_id", "default_user"),
            session_id=template_params.get("session_id", "default_session"),
            workflow_id=template_params.get("workflow_id", f"{template_name}_workflow"),
            workflow_name=template_name,
            initiating_query=template_params.get("query", ""),
            task_type=task_type_map.get(template_name, TaskType.CHAT)
        )

"""
System Context Generators for JARVISv3
"""
import logging
from typing import Dict, Any
from .base import ContextGenerator
from ..context.schemas import TaskContext
from ...core.hardware import HardwareService
from ...core.budget import budget_service

logger = logging.getLogger(__name__)

class HardwareGenerator(ContextGenerator):
    """Contributes hardware state to the context"""
    
    def __init__(self):
        self.hardware_service = HardwareService()
        
    @property
    def name(self) -> str:
        return "hardware_generator"
        
    async def generate(self, context: TaskContext, **kwargs) -> TaskContext:
        try:
            hardware_state = await self.hardware_service.get_hardware_state()
            context.system_context.hardware_state = hardware_state
            logger.debug("Hardware state added to context")
        except Exception as e:
            logger.error(f"Error generating hardware context: {e}")
        return context

class BudgetGenerator(ContextGenerator):
    """Contributes budget state to the context"""
    
    @property
    def name(self) -> str:
        return "budget_generator"
        
    async def generate(self, context: TaskContext, **kwargs) -> TaskContext:
        try:
            user_id = context.system_context.user_id
            budget_state = await budget_service.get_budget_state(user_id)
            context.system_context.budget_state = budget_state
            logger.debug("Budget state added to context")
        except Exception as e:
            logger.error(f"Error generating budget context: {e}")
        return context

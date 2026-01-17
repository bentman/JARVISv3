"""
Base Context Generator for JARVISv3
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..context.schemas import TaskContext

class ContextGenerator(ABC):
    """
    Abstract base class for context generators.
    Generators contribute specific data to the TaskContext.
    """
    
    @abstractmethod
    async def generate(self, context: TaskContext, **kwargs) -> TaskContext:
        """
        Contribute data to the provided context.
        Returns the updated context.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the generator"""
        pass

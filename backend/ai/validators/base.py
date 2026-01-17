"""
Base Validator for JARVISv3
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from ..context.schemas import TaskContext

class ValidationResult(BaseModel):
    """Result of a validation operation"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

class BaseValidator(ABC):
    """
    Abstract base class for validators.
    Validators ensure the quality, security, and correctness of context and outputs.
    """
    
    @abstractmethod
    async def validate(self, target: Any, context: Optional[TaskContext] = None, **kwargs) -> ValidationResult:
        """
        Perform validation on the target object.
        Returns a ValidationResult.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the validator"""
        pass

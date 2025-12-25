"""
Budget Validator for JARVISv3
"""
import logging
from typing import Any, Optional
from .base import BaseValidator, ValidationResult
from ..context.schemas import TaskContext
from ...core.budget import budget_service

logger = logging.getLogger(__name__)

class BudgetValidator(BaseValidator):
    """Validates budget and resource constraints"""
    
    @property
    def name(self) -> str:
        return "budget_validator"
        
    async def validate(self, target: Any, context: Optional[TaskContext] = None, **kwargs) -> ValidationResult:
        errors = []
        warnings = []
        
        if not context:
            return ValidationResult(is_valid=True)
            
        try:
            # 1. Check user's global budget via BudgetService
            user_id = context.system_context.user_id
            budget_state = await budget_service.get_budget_state(user_id)
            
            if budget_state.monthly_limit_usd > 0 and budget_state.cloud_spend_usd >= budget_state.monthly_limit_usd:
                errors.append(f"Monthly budget limit reached: ${budget_state.cloud_spend_usd:.2f}")
            elif budget_state.remaining_pct < 10:
                warnings.append(f"Monthly budget is low: {budget_state.remaining_pct:.1f}% remaining")
            
            # 2. Check workflow-specific token budget
            workflow_budget = context.workflow_context.context_budget
            if workflow_budget.consumed_tokens > workflow_budget.max_tokens:
                errors.append(f"Workflow token budget exceeded: {workflow_budget.consumed_tokens}/{workflow_budget.max_tokens}")
            elif workflow_budget.consumed_tokens > workflow_budget.max_tokens * 0.9:
                warnings.append("Workflow token budget is almost exhausted")
                
            # 3. Check context size
            context_size = context.get_context_size()
            if context_size > workflow_budget.max_size_bytes:
                 errors.append(f"Context size limit exceeded: {context_size} bytes")
            elif context_size > workflow_budget.max_size_bytes * 0.9:
                 warnings.append("Context size is approaching limit")

        except Exception as e:
            logger.error(f"Error in budget validator: {e}")
            warnings.append(f"Budget validation partially failed: {e}")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

"""
Budget Service for JARVISv3
Ports JARVISv2 budget enforcement to JARVISv3 workflow architecture.
Uses DatabaseManager for persistence.
"""
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from ..core.database import database_manager
from ..ai.context.schemas import BudgetState

logger = logging.getLogger(__name__)

class BudgetService:
    """
    Manages budget tracking and enforcement for workflows.
    """
    
    def __init__(self):
        self.db = database_manager
        
    async def get_budget_state(self, user_id: str) -> BudgetState:
        """Get current budget state for a user"""
        try:
            # Ensure DB is initialized (idempotent)
            await self.db.initialize()
            
            record = await self.db.get_budget_record(user_id)
            
            if record:
                remaining = max(0, record['monthly_limit_usd'] - record['monthly_spent_usd'])
                remaining_pct = (remaining / record['monthly_limit_usd']) * 100 if record['monthly_limit_usd'] > 0 else 0
                
                return BudgetState(
                    cloud_spend_usd=record['monthly_spent_usd'],
                    monthly_limit_usd=record['monthly_limit_usd'],
                    remaining_pct=remaining_pct,
                    daily_spending=record['daily_spent_usd']
                )
            else:
                # Return default state if no record exists
                return BudgetState(
                    cloud_spend_usd=0.0,
                    monthly_limit_usd=100.0,
                    remaining_pct=100.0,
                    daily_spending=0.0
                )
        except Exception as e:
            logger.error(f"Error getting budget state: {e}")
            # Fallback to safe default
            return BudgetState(
                cloud_spend_usd=0.0,
                monthly_limit_usd=100.0,
                remaining_pct=100.0,
                daily_spending=0.0
            )

    async def check_budget(self, user_id: str, estimated_cost: float) -> bool:
        """Check if user has enough budget for an operation"""
        state = await self.get_budget_state(user_id)
        if state.cloud_spend_usd + estimated_cost > state.monthly_limit_usd:
            return False
        return True

    async def log_usage(self, user_id: str, workflow_id: str, tokens: int, cost: float):
        """Log usage and update budget"""
        try:
            await self.db.initialize()
            await self.db.update_budget_usage(user_id, workflow_id, cost, tokens)
        except Exception as e:
            logger.error(f"Error logging usage: {e}")

class CloudEscalationManager:
    """Manages decisions about when to escalate to cloud resources"""
    
    def __init__(self):
        self.budget_service = BudgetService()
        
    async def should_escalate_to_cloud(self, user_id: str, hardware_state: Dict, task_requirements: Dict) -> bool:
        """
        Determine if a task should be processed in the cloud based on:
        1. Hardware limitations
        2. Task priority/requirements
        3. Budget availability
        """
        # 1. Hardware check
        local_capable = True
        if task_requirements.get("requires_gpu") and hardware_state.get("gpu_usage", 100) > 90:
            local_capable = False
        if hardware_state.get("memory_available_gb", 0) < task_requirements.get("memory_gb", 0):
            local_capable = False
            
        if local_capable:
            return False # Prefer local if capable
            
        # 2. Budget check for cloud
        # Estimate cost (placeholder)
        estimated_cost = 0.01 
        has_budget = await self.budget_service.check_budget(user_id, estimated_cost)
        
        return has_budget

# Global instances
budget_service = BudgetService()
budget_manager = budget_service # Alias for compatibility
cloud_escalation_manager = CloudEscalationManager()

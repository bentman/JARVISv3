"""
Reflector Node for JARVISv3
Implements self-correction logic by analyzing results and routing accordingly.
"""
from typing import Dict, Any, Optional
import logging
from ..context.schemas import TaskContext

logger = logging.getLogger(__name__)

class ReflectorNode:
    """
    Node that critiques a result and determines the next step in a cyclic workflow.
    """
    
    async def execute(self, context: TaskContext, node_results: Dict[str, Any], target_node_id: str, criteria: str) -> Dict[str, Any]:
        """
        Execute reflection logic.
        
        Args:
            context: The task context
            node_results: Results from previous nodes
            target_node_id: The node to loop back to if validation fails
            criteria: Description of success criteria
            
        Returns:
            Dict with 'next_node' key indicating where to go next.
        """
        # Look for the most recent validation result
        # It could be in 'validator' or 'final_validator' or just the last result
        validator_result = node_results.get("validator", {})
        if not validator_result:
             validator_result = node_results.get("final_validator", {})
             
        is_valid = validator_result.get("is_valid", False)
        
        # Get success node from context or default to None (End)
        # We might want to pass this in conditions too, but for now let's assume if valid we stop or go to next linear step?
        # In `_execute_cyclic_state_machine`, if we return None, it stops.
        # If we want to continue, we must return a node ID.
        # Let's check if there's a 'success_node_id' in node conditions passed via context? 
        # No, 'execute' signature doesn't take node config directly, but `engine.py` calls it with args from conditions.
        # I should have added `success_node_id` to the `execute` method signature in `engine.py`.
        
        # For now, let's assume if valid, we return None (finish workflow) or we rely on the engine's fallback?
        # Engine fallback: "Priority 2: Static dependency".
        # So if we return {} (no next_node), the engine looks at static dependencies.
        # PERFECT.
        
        if is_valid:
            logger.info("Validation passed. Proceeding via static dependencies.")
            return {"status": "approved"} 
        
        else:
            logger.info(f"Validation failed. Looping back to {target_node_id}.")
            
            # Add feedback to context so the generator knows what to fix
            errors = validator_result.get("errors", [])
            feedback = f"Previous attempt failed validation: {'; '.join(errors)}."
            
            # Update additional_context with feedback
            if "feedback_history" not in context.additional_context:
                context.additional_context["feedback_history"] = []
            context.additional_context["feedback_history"].append(feedback)
            context.additional_context["last_feedback"] = feedback
            
            return {"next_node": target_node_id, "status": "rejected", "feedback": feedback}

reflector_node = ReflectorNode()

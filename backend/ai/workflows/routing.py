"""
Routing logic for workflow selection.
"""
from typing import Dict, Any
from ..context.schemas import TaskContext, TaskType

class Router:
    @staticmethod
    async def route(context: TaskContext) -> Dict[str, Any]:
        """
        Determine the next step based on user intent and system state.
        """
        intent = context.workflow_context.user_intent
        query = context.workflow_context.initiating_query.lower()
        
        # Enhanced routing logic
        if "read file" in query or "list files" in query:
             return {
                "next_node": "tool_call",
                "route_decision": "mcp_tool",
                "confidence": 1.0
            }
            
        if intent.type == TaskType.CODING:
            return {
                "next_node": "llm_worker", # For now, direct to worker, maybe specialized later
                "route_decision": "coding",
                "confidence": intent.confidence
            }
        elif intent.type == TaskType.RESEARCH:
            return {
                "next_node": "search_node", # Route to the new SearchNode
                "route_decision": "research",
                "confidence": intent.confidence
            }
        else:
            # Default to chat
            return {
                "next_node": "context_builder", # Proceed to standard chat flow
                "route_decision": "chat",
                "confidence": intent.confidence
            }

router = Router()

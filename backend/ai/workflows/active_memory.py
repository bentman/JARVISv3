"""
Active Memory Node for JARVISv3
Allows workflows to explicitly read/write to long-term memory.
"""
import logging
from typing import Dict, Any, List
from ..context.schemas import TaskContext
from ...core.memory import memory_service

logger = logging.getLogger(__name__)

class ActiveMemoryNode:
    """
    Node for active memory management during workflow execution.
    """
    
    async def execute(self, context: TaskContext, operation: str, content: Any = None, query: str = None) -> Dict[str, Any]:
        """
        Execute memory operation.
        
        Args:
            context: Task context
            operation: 'store', 'retrieve', 'pin'
            content: Content to store (for store/pin)
            query: Query string (for retrieve)
        """
        session_id = context.system_context.session_id
        
        if operation == "store":
            if not content:
                return {"error": "No content provided for storage"}
            
            msg_id = await memory_service.add_message(
                conversation_id=session_id,
                role="system", # Stored as system note
                content=str(content),
                mode="active_memory"
            )
            return {"status": "stored", "message_id": msg_id}
            
        elif operation == "pin":
            # Store and tag as pinned
            if not content:
                return {"error": "No content provided for pinning"}
                
            msg_id = await memory_service.add_message(
                conversation_id=session_id,
                role="system",
                content=str(content),
                mode="active_memory"
            )
            await memory_service.set_message_tags(msg_id, ["pinned"])
            return {"status": "pinned", "message_id": msg_id}
            
        elif operation == "retrieve":
            if not query:
                return {"error": "No query provided for retrieval"}
                
            results = await memory_service.semantic_search(query, k=5)
            return {"status": "retrieved", "results": results}
            
        else:
            return {"error": f"Unknown operation: {operation}"}

active_memory_node = ActiveMemoryNode()

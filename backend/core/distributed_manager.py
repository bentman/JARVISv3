"""
Distributed Manager for JARVISv3
Orchestrates heartbeats, node discovery, and load balancing across the distributed network.
"""
import asyncio
import logging
import httpx
from datetime import datetime
from .node_registry import node_registry
from .hardware import HardwareService

logger = logging.getLogger(__name__)

class DistributedManager:
    """
    Background service for managing distributed node state.
    """
    
    def __init__(self):
        self.hardware_service = HardwareService()
        self._running = False
        self._task = None
        
    async def start(self):
        """Start the background management task"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run_loop())
            logger.info("DistributedManager started")
            
    async def stop(self):
        """Stop the background management task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("DistributedManager stopped")

    async def _run_loop(self):
        """Main loop for heartbeats and discovery"""
        while self._running:
            try:
                # 1. Update self in registry (heartbeat simulation for local)
                local_node = await node_registry.get_local_node()
                
                # 2. Send heartbeat to all known remote nodes
                active_nodes = await node_registry.get_active_nodes()
                for node in active_nodes:
                    if node.node_id != node_registry.local_node_id:
                        await self._send_heartbeat(node, local_node)
                        
                # 3. Discovery (Optional: scan for new nodes if configured)
                # ...
                
            except Exception as e:
                logger.error(f"Error in distributed manager loop: {e}")
                
            await asyncio.sleep(60) # Heartbeat every minute

    async def _send_heartbeat(self, remote_node, local_node):
        """Send local heartbeat and load to a remote node"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{remote_node.base_url}/api/v1/distributed/heartbeat",
                    params={
                        "node_id": local_node.node_id,
                        "load": local_node.current_load
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to send heartbeat to {remote_node.name}: {e}")

# Global instance
distributed_manager = DistributedManager()

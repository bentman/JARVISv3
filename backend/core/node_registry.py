"""
Node Registry Service for distributed JARVISv3 instances.
Handles registration, discovery, and health monitoring of nodes.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime, UTC
from ..ai.context.schemas import RemoteNode, NodeCapability
from .hardware import HardwareService

logger = logging.getLogger(__name__)

class NodeRegistry:
    """
    Manages the distributed network of JARVISv3 nodes.
    """
    
    def __init__(self):
        self.nodes: Dict[str, RemoteNode] = {}
        self.local_node_id = str(uuid.uuid4())
        self.hardware_service = HardwareService()
        self._lock = asyncio.Lock()
        
    async def get_local_node(self) -> RemoteNode:
        """Construct and return the representation of the local node"""
        hardware_state = await self.hardware_service.get_hardware_state()
        
        # Determine tier
        tier = "light"
        if "gpu" in hardware_state.available_tiers:
            tier = "heavy" if hardware_state.memory_available_gb > 16 else "medium"
        elif "npu" in hardware_state.available_tiers:
            tier = "npu-optimized"
            
        capability = NodeCapability(
            hardware_tier=tier,
            specialized_models=[], # Would be populated from model_manager
            supports_voice=True
        )
        
        return RemoteNode(
            node_id=self.local_node_id,
            name="Local Node",
            base_url="http://localhost:8000",
            status="online",
            capabilities=capability,
            current_load=hardware_state.current_load
        )

    async def register_node(self, node: RemoteNode):
        """Register or update a remote node"""
        async with self._lock:
            self.nodes[node.node_id] = node
            logger.info(f"Registered node: {node.name} ({node.node_id}) at {node.base_url}")

    async def unregister_node(self, node_id: str):
        """Remove a node from the registry"""
        async with self._lock:
            if node_id in self.nodes:
                name = self.nodes[node_id].name
                del self.nodes[node_id]
                logger.info(f"Unregistered node: {name} ({node_id})")

    async def get_active_nodes(self) -> List[RemoteNode]:
        """Return list of online nodes that have sent a heartbeat recently"""
        active = []
        now = datetime.now(UTC)
        async with self._lock:
            for node in self.nodes.values():
                # Node is active if online and heartbeat < 5 mins old
                if node.status == "online" and (now - node.last_heartbeat).total_seconds() < 300:
                    active.append(node)
        return active

    async def find_best_node_for_task(self, required_tier: str) -> Optional[RemoteNode]:
        """
        Find the most suitable node based on hardware tier and current load.
        """
        active_nodes = await self.get_active_nodes()
        
        # Include local node in consideration
        local_node = await self.get_local_node()
        active_nodes.append(local_node)
        
        # Filter by tier
        suitable_nodes = [
            n for n in active_nodes 
            if self._is_tier_sufficient(n.capabilities.hardware_tier, required_tier)
        ]
        
        if not suitable_nodes:
            return None
            
        # Sort by load (ascending)
        suitable_nodes.sort(key=lambda x: x.current_load)
        
        return suitable_nodes[0]

    def _is_tier_sufficient(self, node_tier: str, required_tier: str) -> bool:
        """Check if node tier meets or exceeds required tier"""
        tiers = ["light", "medium", "heavy"]
        if node_tier == "npu-optimized" and required_tier == "light":
            return True
        if node_tier not in tiers or required_tier not in tiers:
            return node_tier == required_tier
        return tiers.index(node_tier) >= tiers.index(required_tier)

# Global instance
node_registry = NodeRegistry()

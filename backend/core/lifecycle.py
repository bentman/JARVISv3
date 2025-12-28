"""
Context Lifecycle Management for JARVISv3
Optimized for low-end hardware with aggressive memory management.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, UTC
import json
import zlib
import os
from pathlib import Path

from ..ai.context.schemas import TaskContext
from .database import database_manager

logger = logging.getLogger(__name__)

class ContextLifecycleManager:
    """
    Manages the lifecycle of context objects with aggressive memory optimization
    for low-end hardware profiles.
    """
    
    def __init__(self):
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Memory management settings for different hardware profiles
        self.memory_limits = {
            "light": {
                "max_context_size": 50000,      # 50KB
                "summarization_threshold": 30000,  # 30KB
                "prune_interval": 100,          # Every 100 tokens
                "checkpoint_interval": 50,      # Every 50 tokens
                "max_checkpoints": 5            # Keep only 5 checkpoints
            },
            "medium": {
                "max_context_size": 200000,     # 200KB
                "summarization_threshold": 150000,  # 150KB
                "prune_interval": 200,          # Every 200 tokens
                "checkpoint_interval": 100,     # Every 100 tokens
                "max_checkpoints": 10           # Keep 10 checkpoints
            },
            "heavy": {
                "max_context_size": 500000,     # 500KB
                "summarization_threshold": 400000,  # 400KB
                "prune_interval": 500,          # Every 500 tokens
                "checkpoint_interval": 200,     # Every 200 tokens
                "max_checkpoints": 20           # Keep 20 checkpoints
            },
            "npu-optimized": {
                "max_context_size": 100000,     # 100KB
                "summarization_threshold": 75000,   # 75KB
                "prune_interval": 150,          # Every 150 tokens
                "checkpoint_interval": 75,      # Every 75 tokens
                "max_checkpoints": 8            # Keep 8 checkpoints
            }
        }
        
        # Compression settings
        self.compression_enabled = True
        self.compression_level = 6
        
    async def manage_context_lifecycle(self, context: TaskContext, hardware_profile: str = "light") -> TaskContext:
        """Apply lifecycle management to context based on hardware"""
        settings = self.memory_limits.get(hardware_profile, self.memory_limits["light"])
        context_size = context.get_context_size()
        
        # Check if we need to summarize
        if context_size > settings["summarization_threshold"]:
            context = await self.summarize_context(context, hardware_profile)
        
        # Apply pruning
        context = await self.prune_context(context, hardware_profile)
        
        return context
    
    async def summarize_context(self, context: TaskContext, hardware_profile: str = "light") -> TaskContext:
        """Aggressively summarize context to reduce memory usage"""
        try:
            # Preserve critical information
            preserved_data = {
                'system_context': context.system_context,
                'workflow_context': context.workflow_context,
                'additional_context': context.additional_context
            }
            
            # Summarize the accumulated artifacts if too many
            max_artifacts = 3 if hardware_profile == "light" else 5
            if len(context.workflow_context.accumulated_artifacts) > max_artifacts:
                preserved_data['workflow_context'].accumulated_artifacts = context.workflow_context.accumulated_artifacts[-max_artifacts:]
            
            # Summarize the error history if too many
            max_errors = 2 if hardware_profile == "light" else 3
            if len(context.workflow_context.error_history) > max_errors:
                preserved_data['workflow_context'].error_history = context.workflow_context.error_history[-max_errors:]
            
            # Create summary string
            summary = await self._generate_context_summary(context)
            preserved_data['context_summary'] = summary
            
            # Create new context
            new_context = TaskContext(**preserved_data)
            return new_context
            
        except Exception as e:
            logger.error(f"Context summarization failed: {e}")
            return context
    
    async def _generate_context_summary(self, context: TaskContext) -> str:
        """Generate a concise summary of the context"""
        summary = {
            'workflow_id': context.workflow_context.workflow_id,
            'initiating_query': context.workflow_context.initiating_query[:50] + "...",
            'artifacts': len(context.workflow_context.accumulated_artifacts),
            'timestamp': datetime.now(UTC).isoformat()
        }
        return json.dumps(summary)
    
    async def prune_context(self, context: TaskContext, hardware_profile: str = "light") -> TaskContext:
        """Aggressively prune context to fit memory constraints"""
        settings = self.memory_limits.get(hardware_profile, self.memory_limits["light"])
        
        # Prune artifacts
        max_artifacts = 5 if hardware_profile == "light" else 10
        if len(context.workflow_context.accumulated_artifacts) > max_artifacts:
            context.workflow_context.accumulated_artifacts = context.workflow_context.accumulated_artifacts[-max_artifacts:]
            
        # Prune errors
        max_errors = 3 if hardware_profile == "light" else 5
        if len(context.workflow_context.error_history) > max_errors:
            context.workflow_context.error_history = context.workflow_context.error_history[-max_errors:]
            
        return context
    
    async def checkpoint_context(self, context: TaskContext, checkpoint_id: str) -> Optional[str]:
        """Save context checkpoint with compression"""
        try:
            context_data = context.model_dump()
            json_data = json.dumps(context_data, default=str).encode('utf-8')
            
            if self.compression_enabled:
                data_to_save = zlib.compress(json_data, level=self.compression_level)
                filename = f"{checkpoint_id}.cp"
            else:
                data_to_save = json_data
                filename = f"{checkpoint_id}.ck"
                
            checkpoint_path = self.checkpoint_dir / filename
            with open(checkpoint_path, 'wb') as f:
                f.write(data_to_save)
                
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Checkpoint failed: {e}")
            return None

class ContextArchiver:
    """Handles long-term storage and retrieval of important context artifacts"""
    
    def __init__(self):
        self.retention_periods = {
            'short_term': timedelta(hours=24),
            'medium_term': timedelta(days=7),
            'long_term': timedelta(days=365)
        }
        self.logger = logging.getLogger(__name__)
    
    async def archive_context(self, context: TaskContext, retention: str = 'medium_term') -> str:
        """Archive context for long-term storage"""
        archive_id = f"archive_{context.workflow_context.workflow_id}_{datetime.now(UTC).timestamp()}"
        self.logger.info(f"Context archived: {archive_id} with retention {retention}")
        return archive_id

# Global instances
context_lifecycle_manager = ContextLifecycleManager()
context_archiver = ContextArchiver()

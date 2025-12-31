"""
Cache Service for JARVISv3
Implements caching using Redis for ephemeral data storage and performance optimization.
"""
import json
import time
import hashlib
from typing import Optional, Any, List
import logging
import asyncio

import redis.asyncio as redis

from .config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Async cache service using Redis for ephemeral data storage
    """
    
    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.REDIS_URL
        self.client: Optional[redis.Redis] = None
        self._connected = False
        self._enabled = settings.ENABLE_CACHE

    async def initialize(self):
        """Initialize the Redis connection"""
        if not self._enabled:
            logger.info("Cache service disabled via configuration")
            return

        if not self._connected:
            try:
                self.client = redis.from_url(self.url, decode_responses=True)
                # Ping to ensure connection is alive
                await self.client.ping()
                self._connected = True
                logger.info("Cache service connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False

    async def healthy(self) -> bool:
        """Check if cache service is healthy"""
        if not self._enabled:
            return False  # Consider disabled cache as not healthy
        if not self._connected or not self.client:
            return False
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    def key_for_chat(self, mode: str, message: str) -> str:
        """Generate a cache key for chat responses"""
        h = hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
        return f"chat:{mode}:{h}"

    def key_for_workflow(self, workflow_id: str, node_id: str) -> str:
        """Generate a cache key for workflow nodes"""
        return f"workflow:{workflow_id}:{node_id}"

    def key_for_model(self, model_name: str, input_hash: str) -> str:
        """Generate a cache key for model responses"""
        return f"model:{model_name}:{input_hash}"

    async def set_json(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        """Set a JSON value in cache with TTL"""
        if not self._connected or not self.client:
            return False
        try:
            payload = json.dumps(value)
            await self.client.setex(name=key, time=ttl_seconds, value=payload)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    async def get_json(self, key: str) -> Optional[Any]:
        """Get a JSON value from cache"""
        if not self._connected or not self.client:
            return None
        try:
            data = await self.client.get(key)
            if not data:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        if not self._connected or not self.client:
            return False
        try:
            result = await self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        if not self._connected or not self.client:
            return False
        try:
            result = await self.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check existence of cache key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern, return count of deleted keys"""
        if not self._connected or not self.client:
            return 0
        try:
            keys = await self.client.keys(pattern)
            if keys:
                await self.client.delete(*keys)
                return len(keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache pattern {pattern}: {e}")
            return 0


# Global instance
cache_service = CacheService()


async def initialize_cache():
    """Initialize the cache service"""
    await cache_service.initialize()


# Example usage and test function
async def test_cache_service():
    """Test function to demonstrate cache service features"""
    await initialize_cache()
    
    # Test basic operations
    test_key = "test:key:123"
    test_value = {"message": "Hello, Cached World!", "timestamp": time.time()}
    
    # Set a value
    success = await cache_service.set_json(test_key, test_value, ttl_seconds=60)
    print(f"Set cache result: {success}")
    
    # Get the value back
    retrieved = await cache_service.get_json(test_key)
    print(f"Retrieved from cache: {retrieved}")
    
    # Test health check
    is_healthy = await cache_service.healthy()
    print(f"Cache health: {is_healthy}")
    
    # Test key generation
    chat_key = cache_service.key_for_chat("text", "Hello world")
    print(f"Generated chat key: {chat_key}")
    
    workflow_key = cache_service.key_for_workflow("wf_123", "node_abc")
    print(f"Generated workflow key: {workflow_key}")
    
    # Clean up
    deleted = await cache_service.delete(test_key)
    print(f"Deleted key result: {deleted}")
    
    return {
        "set_result": success,
        "get_result": retrieved,
        "health": is_healthy,
        "generated_keys": {
            "chat": chat_key,
            "workflow": workflow_key
        }
    }


if __name__ == "__main__":
    asyncio.run(test_cache_service())

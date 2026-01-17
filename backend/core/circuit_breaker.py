"""
Circuit Breaker implementation for JARVISv3
Provides failure recovery and resilience for external dependencies
"""
import asyncio
import time
from datetime import datetime, UTC
from enum import Enum
from typing import Dict, Any, Optional, Callable, Awaitable, Type
import logging

logger = logging.getLogger("JARVISv3.circuit_breaker")


class CircuitBreakerState(Enum):
    """States of the circuit breaker"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for a single external service"""

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

        self.logger = logging.getLogger(f"JARVISv3.circuit_breaker.{service_name}")

    def _can_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    async def _call_async(self, func: Callable[..., Awaitable], *args, **kwargs):
        """Execute an async function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._can_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker for {self.service_name} entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenException(self.service_name)

        try:
            result = await func(*args, **kwargs)

            # Success handling
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._reset()
                else:
                    self.logger.debug(f"Circuit breaker for {self.service_name}: {self.success_count}/{self.success_threshold} successes")
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = 0
                    self.logger.info(f"Circuit breaker for {self.service_name} recovered")

            return result

        except self.expected_exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self._trip()
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self._trip()

    def _trip(self):
        """Trip the circuit breaker to OPEN state"""
        self.state = CircuitBreakerState.OPEN
        self.logger.warning(f"Circuit breaker for {self.service_name} tripped to OPEN state (failures: {self.failure_count})")

    def _reset(self):
        """Reset the circuit breaker to CLOSED state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger.info(f"Circuit breaker for {self.service_name} reset to CLOSED state")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the circuit breaker"""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "can_attempt_reset": self._can_attempt_reset() if self.state == CircuitBreakerState.OPEN else None,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(f"Circuit breaker is OPEN for service: {service_name}")


class CircuitBreakerManager:
    """Manages multiple circuit breakers"""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger("JARVISv3.circuit_breaker.manager")

    def register_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 3
    ) -> CircuitBreaker:
        """Register a new circuit breaker"""
        breaker = CircuitBreaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold
        )
        self.breakers[service_name] = breaker
        self.logger.info(f"Registered circuit breaker for service: {service_name}")
        return breaker

    def get_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by service name"""
        return self.breakers.get(service_name)

    async def call_with_breaker(
        self,
        service_name: str,
        func: Callable[..., Awaitable],
        *args,
        **kwargs
    ):
        """Execute a function with circuit breaker protection"""
        breaker = self.get_breaker(service_name)
        if not breaker:
            # If no breaker registered, just call the function
            return await func(*args, **kwargs)

        return await breaker._call_async(func, *args, **kwargs)

    async def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {name: breaker.get_status() for name, breaker in self.breakers.items()}


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


def setup_circuit_breakers():
    """Set up default circuit breakers for common external services"""
    # Ollama model provider
    circuit_breaker_manager.register_breaker(
        service_name="ollama_provider",
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2
    )

    # Web search providers (DuckDuckGo)
    circuit_breaker_manager.register_breaker(
        service_name="web_search",
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=3
    )

    # Voice services (if external)
    circuit_breaker_manager.register_breaker(
        service_name="voice_services",
        failure_threshold=3,
        recovery_timeout=45.0,
        success_threshold=2
    )

    logger.info("Circuit breakers initialized for external services")


# Initialize circuit breakers on import
setup_circuit_breakers()

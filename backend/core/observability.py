"""
Observability and monitoring for JARVISv3
Implements logging, metrics, and tracing for the agentic system
"""
import logging
import time
from datetime import datetime, UTC
from typing import Dict, Any, Optional, Callable, Awaitable
from functools import wraps
import asyncio
from pydantic import BaseModel

from ..ai.context.schemas import TaskContext
from ..ai.workflows.engine import WorkflowState


class MetricsCollector(BaseModel):
    """Collects and stores metrics for the system (Prometheus-compatible)"""
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance metrics
    total_tokens_used: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0

    # Workflow metrics
    workflows_started: int = 0
    workflows_completed: int = 0
    workflows_failed: int = 0
    average_workflow_duration: float = 0.0

    # Node metrics
    nodes_executed: int = 0
    nodes_succeeded: int = 0
    nodes_failed: int = 0

    # Model metrics
    model_inference_count: int = 0
    model_inference_total_time: float = 0.0
    model_average_inference_time: float = 0.0

    # Error metrics
    error_counts: Dict[str, int] = {}

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # System metrics
    start_time: datetime = datetime.now(UTC)
    last_updated: datetime = datetime.now(UTC)

    def increment_requests(self, success: bool = True, tokens_used: int = 0, execution_time: float = 0.0):
        """Increment request counters"""
        self.total_requests += 1
        self.last_updated = datetime.now(UTC)

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_tokens_used += tokens_used
        self.total_execution_time += execution_time

        if self.total_requests > 0:
            self.average_execution_time = self.total_execution_time / self.total_requests

    def increment_workflow(self, success: bool = True, duration: float = 0.0):
        """Increment workflow counters"""
        self.workflows_started += 1
        self.last_updated = datetime.now(UTC)

        if success:
            self.workflows_completed += 1
        else:
            self.workflows_failed += 1

        if self.workflows_completed > 0:
            # This is a simplified calculation - in reality we'd track each workflow duration
            self.average_workflow_duration = (self.average_workflow_duration + duration) / 2

    def increment_nodes(self, success: bool = True):
        """Increment node execution counters"""
        self.nodes_executed += 1
        self.last_updated = datetime.now(UTC)

        if success:
            self.nodes_succeeded += 1
        else:
            self.nodes_failed += 1

    def record_model_inference(self, duration: float):
        """Record model inference metrics"""
        self.model_inference_count += 1
        self.model_inference_total_time += duration
        self.model_average_inference_time = self.model_inference_total_time / self.model_inference_count
        self.last_updated = datetime.now(UTC)

    def record_error(self, error_type: str):
        """Record error occurrence"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        self.last_updated = datetime.now(UTC)

    def update_resource_usage(self, memory_mb: Optional[float] = None, cpu_percent: Optional[float] = None):
        """Update resource usage metrics"""
        if memory_mb is not None:
            self.memory_usage_mb = memory_mb
        if cpu_percent is not None:
            self.cpu_usage_percent = cpu_percent
        self.last_updated = datetime.now(UTC)

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output"""
        lines = [
            "# HELP jarvis_requests_total Total number of requests processed",
            "# TYPE jarvis_requests_total counter",
            f"jarvis_requests_total {self.total_requests}",
            "",
            "# HELP jarvis_requests_success_total Number of successful requests",
            "# TYPE jarvis_requests_success_total counter",
            f"jarvis_requests_success_total {self.successful_requests}",
            "",
            "# HELP jarvis_requests_failed_total Number of failed requests",
            "# TYPE jarvis_requests_failed_total counter",
            f"jarvis_requests_failed_total {self.failed_requests}",
            "",
            "# HELP jarvis_tokens_used_total Total tokens used",
            "# TYPE jarvis_tokens_used_total counter",
            f"jarvis_tokens_used_total {self.total_tokens_used}",
            "",
            "# HELP jarvis_execution_time_total Total execution time in seconds",
            "# TYPE jarvis_execution_time_total counter",
            f"jarvis_execution_time_total {self.total_execution_time}",
            "",
            "# HELP jarvis_execution_time_average Average execution time in seconds",
            "# TYPE jarvis_execution_time_average gauge",
            f"jarvis_execution_time_average {self.average_execution_time}",
            "",
            "# HELP jarvis_workflows_started_total Total workflows started",
            "# TYPE jarvis_workflows_started_total counter",
            f"jarvis_workflows_started_total {self.workflows_started}",
            "",
            "# HELP jarvis_workflows_completed_total Total workflows completed",
            "# TYPE jarvis_workflows_completed_total counter",
            f"jarvis_workflows_completed_total {self.workflows_completed}",
            "",
            "# HELP jarvis_nodes_executed_total Total nodes executed",
            "# TYPE jarvis_nodes_executed_total counter",
            f"jarvis_nodes_executed_total {self.nodes_executed}",
            "",
            "# HELP jarvis_model_inference_total Total model inferences",
            "# TYPE jarvis_model_inference_total counter",
            f"jarvis_model_inference_total {self.model_inference_count}",
            "",
            "# HELP jarvis_memory_usage_mb Current memory usage in MB",
            "# TYPE jarvis_memory_usage_mb gauge",
            f"jarvis_memory_usage_mb {self.memory_usage_mb}",
            "",
            "# HELP jarvis_cpu_usage_percent Current CPU usage percentage",
            "# TYPE jarvis_cpu_usage_percent gauge",
            f"jarvis_cpu_usage_percent {self.cpu_usage_percent}",
        ]

        # Add error metrics
        for error_type, count in self.error_counts.items():
            lines.extend([
                "",
                f"# HELP jarvis_errors_total_{error_type} Total {error_type} errors",
                f"# TYPE jarvis_errors_total_{error_type} counter",
                f"jarvis_errors_total_{error_type} {count}"
            ])

        return "\n".join(lines)


class LoggerConfig:
    """Configuration for logging in JARVISv3"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
        """Set up logging configuration"""
        # Create custom logger
        logger = logging.getLogger("JARVISv3")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatters and add to handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:  # Avoid adding multiple handlers
            logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class WorkflowTracer:
    """Traces workflow execution for observability"""
    
    def __init__(self):
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("JARVISv3.observability")
    
    def start_trace(self, workflow_id: str, context: TaskContext):
        """Start tracing a workflow"""
        self.traces[workflow_id] = {
            "workflow_id": workflow_id,
            "start_time": datetime.now(UTC),
            "context_summary": {
                "user_id": context.system_context.user_id,
                "workflow_name": context.workflow_context.workflow_name,
                "initiating_query": context.workflow_context.initiating_query[:100] + "..."
            },
            "nodes": {},
            "status": "running"
        }
        self.logger.info(f"Started trace for workflow {workflow_id}")
    
    def trace_node_start(self, workflow_id: str, node_id: str, input_data: Dict[str, Any]):
        """Trace the start of a workflow node"""
        if workflow_id in self.traces:
            self.traces[workflow_id]["nodes"][node_id] = {
                "node_id": node_id,
                "start_time": datetime.now(UTC),
                "input_data": input_data,
                "status": "running"
            }
            self.logger.debug(f"Started node {node_id} in workflow {workflow_id}")
    
    def trace_node_end(self, workflow_id: str, node_id: str, output_data: Dict[str, Any], success: bool = True):
        """Trace the end of a workflow node"""
        if workflow_id in self.traces:
            if node_id in self.traces[workflow_id]["nodes"]:
                node_trace = self.traces[workflow_id]["nodes"][node_id]
                node_trace["end_time"] = datetime.now(UTC)
                node_trace["output_data"] = output_data
                node_trace["success"] = success
                node_trace["duration"] = (node_trace["end_time"] - node_trace["start_time"]).total_seconds()
                node_trace["status"] = "completed" if success else "failed"
                
                self.logger.debug(f"Completed node {node_id} in workflow {workflow_id} (success: {success})")
    
    def end_trace(self, workflow_id: str, state: WorkflowState, final_result: Dict[str, Any]):
        """End tracing a workflow"""
        if workflow_id in self.traces:
            trace = self.traces[workflow_id]
            trace["end_time"] = datetime.now(UTC)
            trace["duration"] = (trace["end_time"] - trace["start_time"]).total_seconds()
            trace["final_result"] = final_result
            trace["status"] = state.status.value
            trace["completed_nodes"] = state.completed_nodes
            trace["failed_nodes"] = state.failed_nodes
            
            self.logger.info(f"Completed trace for workflow {workflow_id} (duration: {trace['duration']:.2f}s)")
    
    def get_trace(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get trace data for a workflow"""
        return self.traces.get(workflow_id)
    
    def get_all_traces(self) -> Dict[str, Dict[str, Any]]:
        """Get all trace data"""
        return self.traces


class ObservabilityMiddleware:
    """Middleware for observability in FastAPI"""
    
    def __init__(self, metrics_collector: MetricsCollector, workflow_tracer: WorkflowTracer):
        self.metrics_collector = metrics_collector
        self.workflow_tracer = workflow_tracer
        self.logger = logging.getLogger("JARVISv3.middleware")
    
    def log_request(self, endpoint: str, method: str, user_id: Optional[str] = None):
        """Log an incoming request"""
        self.logger.info(f"Request: {method} {endpoint} (user: {user_id or 'unknown'})")
    
    def log_response(self, endpoint: str, method: str, status_code: int, execution_time: float, user_id: Optional[str] = None):
        """Log a response"""
        self.logger.info(f"Response: {method} {endpoint} - {status_code} (time: {execution_time:.3f}s, user: {user_id or 'unknown'})")
    
    async def execute_with_observability(
        self, 
        func: Callable, 
        context: TaskContext,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a function with observability tracking"""
        start_time = time.time()
        workflow_id = context.workflow_context.workflow_id
        
        try:
            # Start tracing
            self.workflow_tracer.start_trace(workflow_id, context)
            
            # Execute the function
            result = await func(context, *args, **kwargs) if asyncio.iscoroutinefunction(func) else func(context, *args, **kwargs)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            tokens_used = context.workflow_context.context_budget.consumed_tokens
            
            # Update metrics
            self.metrics_collector.increment_requests(
                success=True, 
                tokens_used=tokens_used, 
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            # Update metrics for failure
            execution_time = time.time() - start_time
            self.metrics_collector.increment_requests(success=False, execution_time=execution_time)
            
            # Log the error
            self.logger.error(f"Error in workflow {workflow_id}: {str(e)}", exc_info=True)
            
            # Re-raise the exception
            raise


class SystemHealthMonitor:
    """Monitors system health and performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger("JARVISv3.health")
        self.checks: Dict[str, Callable] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], Awaitable[bool]]):
        """Register a health check function"""
        self.checks[name] = check_func
    
    async def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = result
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {str(e)}")
                results[name] = False
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        uptime = (datetime.now(UTC) - self.metrics_collector.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.metrics_collector.total_requests,
            "successful_requests": self.metrics_collector.successful_requests,
            "failed_requests": self.metrics_collector.failed_requests,
            "success_rate": self.metrics_collector.successful_requests / max(self.metrics_collector.total_requests, 1),
            "total_tokens_used": self.metrics_collector.total_tokens_used,
            "total_execution_time": self.metrics_collector.total_execution_time,
            "average_execution_time": self.metrics_collector.average_execution_time,
            "timestamp": datetime.now(UTC).isoformat()
        }


# Global instances
logger_config = LoggerConfig()
metrics_collector = MetricsCollector()
workflow_tracer = WorkflowTracer()
observability_middleware = ObservabilityMiddleware(metrics_collector, workflow_tracer)
health_monitor = SystemHealthMonitor(metrics_collector)


def setup_observability(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up observability for the entire system"""
    logger = logger_config.setup_logging(log_level, log_file)
    
    # Register basic health checks
    async def check_metrics_collector():
        return metrics_collector is not None
    
    async def check_workflow_tracer():
        return workflow_tracer is not None
    
    health_monitor.register_health_check("metrics_collector", check_metrics_collector)
    health_monitor.register_health_check("workflow_tracer", check_workflow_tracer)
    
    logger.info("Observability system initialized")
    
    return logger


# Example usage and test functions
async def test_observability():
    """Test function to demonstrate observability features"""
    logger = setup_observability(log_level="INFO")
    
    # Simulate a workflow execution
    from ai.context.schemas import (
        SystemContext, WorkflowContext, UserIntent, TaskType,
        HardwareState, BudgetState, UserPreferences, ContextBudget, TaskContext
    )
    
    # Create a test context
    test_context = TaskContext(
        system_context=SystemContext(
            user_id="test_user",
            session_id="test_session",
            hardware_state=HardwareState(
                gpu_usage=0.0,
                memory_available_gb=16.0,
                cpu_usage=20.0,
                current_load=0.2
            ),
            budget_state=BudgetState(
                cloud_spend_usd=0.0,
                monthly_limit_usd=100.0,
                remaining_pct=100.0
            ),
            user_preferences=UserPreferences()
        ),
        workflow_context=WorkflowContext(
            workflow_id="test_workflow_obs",
            workflow_name="test_workflow",
            initiating_query="Test observability",
            user_intent=UserIntent(
                type=TaskType.CHAT,
                confidence=0.9,
                description="Test query",
                priority=3
            ),
            context_budget=ContextBudget()
        )
    )
    
    # Start a trace
    workflow_tracer.start_trace("test_workflow_obs", test_context)
    
    # Simulate node execution
    workflow_tracer.trace_node_start("test_workflow_obs", "test_node", {"input": "test"})
    await asyncio.sleep(0.1)  # Simulate work
    workflow_tracer.trace_node_end("test_workflow_obs", "test_node", {"output": "test"}, success=True)
    
    # End trace
    from ai.workflows.engine import WorkflowState, NodeStatus
    fake_state = WorkflowState(
        workflow_id="test_workflow_obs",
        status=NodeStatus.COMPLETED,
        completed_nodes=["test_node"],
        failed_nodes=[]
    )
    workflow_tracer.end_trace("test_workflow_obs", fake_state, {"result": "success"})
    
    # Update metrics
    metrics_collector.increment_requests(success=True, tokens_used=100, execution_time=0.5)
    
    # Get system metrics
    system_metrics = health_monitor.get_system_metrics()
    print(f"System metrics: {system_metrics}")
    
    # Run health checks
    health_results = await health_monitor.run_health_checks()
    print(f"Health check results: {health_results}")
    
    # Get trace
    trace = workflow_tracer.get_trace("test_workflow_obs")
    print(f"Workflow trace: {trace}")
    
    return {
        "system_metrics": system_metrics,
        "health_results": health_results,
        "trace": trace
    }


if __name__ == "__main__":
    asyncio.run(test_observability())

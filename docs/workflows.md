# JARVISv3 Workflow Documentation

This document provides comprehensive guidance on creating, defining, and managing workflows in JARVISv3 using both Python and declarative YAML approaches.

## Overview

JARVISv3 implements a **Workflow Architecture** where tasks are modeled as directed acyclic graphs (DAGs) with explicit nodes for routing, context building, LLM processing, validation, and formatting. This architecture enables:

- **Explicit orchestration** of complex multi-step tasks
- **Reusability** of workflow components
- **Auditability** of task execution
- **Extensibility** for new capabilities

## Workflow Architecture

### Core Components

1. **Workflow Engine**: The graph runner that executes workflows
2. **Nodes**: Individual steps in a workflow (router, context builder, LLM worker, validator, etc.)
3. **Context**: Typed, validated data objects that flow between nodes
4. **Validation Gates**: Multi-layer validation at every workflow step

### Node Types

| Node Type | Purpose | Description |
|-----------|---------|-------------|
| `router` | Intent Classification | Determines which workflow path to take based on user input |
| `context_builder` | Context Assembly | Builds context from various sources (memory, tools, user preferences) |
| `llm_worker` | LLM Processing | Processes input with LLM using assembled context |
| `validator` | Output Validation | Validates LLM output for security, format, and correctness |
| `tool_call` | External Tools | Calls external tools via MCP (Model Context Protocol) |
| `human_approval` | Human Intervention | Requests human approval for high-risk operations |
| `search_web` | Web Search | Performs web searches and aggregates results |
| `end` | Workflow Completion | Marks the end of a workflow |

## Creating Workflows

### Implementing Custom Node Logic (Developer Guide)

For developers wanting to extend the framework with custom logic, there are two primary ways to implement node behavior:

#### 1. Function-Based Nodes (Recommended for simple logic)

You can pass an `async` function to the `execute_func` parameter of a `WorkflowNode`. This function must accept `context: TaskContext` and `results: Dict[str, Any]` and return a dictionary.

```python
async def my_custom_logic(context: TaskContext, results: Dict[str, Any]) -> Dict[str, Any]:
    # results contains output from previous nodes
    query = context.workflow_context.initiating_query
    return {"processed_query": query.upper(), "status": "ok"}

node = WorkflowNode(
    id="custom_node",
    type=NodeType.LLM_WORKER,
    description="Transforms query to uppercase",
    execute_func=my_custom_logic
)
```

#### 2. Specialized Workers (Recommended for complex services)

For complex logic, register a specialized worker in `backend/ai/workflows/collaboration.py` or create a new node type handler in `WorkflowEngine._execute_node_by_type`.

### Method 1: Python API (Recommended for Complex Workflows)

```python
from ai.workflows.engine import WorkflowEngine, WorkflowNode, NodeType
from ai.context.schemas import TaskContext

# Initialize workflow engine
engine = WorkflowEngine()

# Define nodes
engine.add_node(WorkflowNode(
    id="router",
    type=NodeType.ROUTER,
    description="Route to appropriate workflow based on intent",
    dependencies=[]
))

engine.add_node(WorkflowNode(
    id="context_builder",
    type=NodeType.CONTEXT_BUILDER,
    description="Build context from various sources",
    dependencies=["router"]
))

engine.add_node(WorkflowNode(
    id="llm_worker",
    type=NodeType.LLM_WORKER,
    description="Process with LLM",
    dependencies=["context_builder"]
))

engine.add_node(WorkflowNode(
    id="validator",
    type=NodeType.VALIDATOR,
    description="Validate output",
    dependencies=["llm_worker"]
))

# Execute workflow
result = await engine.execute_workflow(task_context)
```

### Method 2: Declarative YAML (Recommended for Simple Workflows)

Create a YAML file defining your workflow:

```yaml
# workflows/simple_chat.yaml
workflow_id: "simple_chat"
nodes:
  - id: "router"
    type: "router"
    description: "Route to appropriate workflow based on intent"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build context from various sources"
    dependencies: ["router"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Process with LLM"
    dependencies: ["context_builder"]
    
  - id: "validator"
    type: "validator"
    description: "Validate output"
    dependencies: ["llm_worker"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
```

Load and execute the workflow:

```python
from ai.workflows.engine import WorkflowEngine
import yaml

# Load workflow definition
with open("workflows/simple_chat.yaml", "r") as f:
    workflow_def = yaml.safe_load(f)

# Create and configure engine
engine = WorkflowEngine()

# Add nodes from definition
for node_data in workflow_def["nodes"]:
    node = WorkflowNode(**node_data)
    engine.add_node(node)

# Add edges
for edge in workflow_def["edges"]:
    engine.add_edge(edge["from"], edge["to"])

# Execute workflow
result = await engine.execute_workflow(task_context)
```

## Built-in Workflows

### 1. Chat Workflow

The default chat workflow handles conversational interactions:

```yaml
# workflows/chat.yaml
workflow_id: "chat"
nodes:
  - id: "router"
    type: "router"
    description: "Classify user intent"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build conversation context"
    dependencies: ["router"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Generate response"
    dependencies: ["context_builder"]
    
  - id: "validator"
    type: "validator"
    description: "Validate response"
    dependencies: ["llm_worker"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
```

### 2. Research Workflow

Handles complex research tasks with web search and tool integration:

```yaml
# workflows/research.yaml
workflow_id: "research"
nodes:
  - id: "router"
    type: "router"
    description: "Identify research intent"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build research context"
    dependencies: ["router"]
    
  - id: "search_web"
    type: "search_web"
    description: "Search the web for information"
    dependencies: ["context_builder"]
    
  - id: "tool_call"
    type: "tool_call"
    description: "Call external tools if needed"
    dependencies: ["search_web"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Synthesize information"
    dependencies: ["tool_call"]
    
  - id: "validator"
    type: "validator"
    description: "Validate research output"
    dependencies: ["llm_worker"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "search_web"
  - from: "search_web"
    to: "tool_call"
  - from: "tool_call"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
```

### 3. Code Generation Workflow

Handles code generation with validation and testing:

```yaml
# workflows/code_generation.yaml
workflow_id: "code_generation"
nodes:
  - id: "router"
    type: "router"
    description: "Identify coding task"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build coding context"
    dependencies: ["router"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Generate code"
    dependencies: ["context_builder"]
    
  - id: "validator"
    type: "validator"
    description: "Validate code quality and security"
    dependencies: ["llm_worker"]
    
  - id: "tool_call"
    type: "tool_call"
    description: "Execute and test code"
    dependencies: ["validator"]
    
  - id: "human_approval"
    type: "human_approval"
    description: "Request approval for deployment"
    dependencies: ["tool_call"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
  - from: "validator"
    to: "tool_call"
  - from: "tool_call"
    to: "human_approval"

### 4. Voice Session Flow (Unified Orchestration)

The voice session provides a high-performance path for headless interaction:

1.  **Audio Ingestion**: Accepts Base64 encoded WAV audio via `/api/v1/voice/session`.
2.  **STT**: Transcribes audio to text using local Whisper.
3.  **Workflow Execution**: Pipes transcription into the Chat Workflow.
4.  **TTS**: Synthesizes the response to audio using local Piper (or espeak fallback).
5.  **Output**: Returns JSON with text, audio data, and conversation ID.

## Advanced Workflow Features

### Conditional Execution

Use node conditions to enable conditional execution:

```yaml
nodes:
  - id: "router"
    type: "router"
    description: "Classify user intent"
    dependencies: []
    
  - id: "web_search"
    type: "search_web"
    description: "Search the web"
    dependencies: ["router"]
    conditions:
      required_intent: ["research", "information"]
      
  - id: "local_processing"
    type: "llm_worker"
    description: "Process locally"
    dependencies: ["router"]
    conditions:
      required_intent: ["chat", "coding"]
```

### Parallel Execution

Enable parallel execution for independent nodes:

```yaml
nodes:
  - id: "context_builder"
    type: "context_builder"
    description: "Build context"
    dependencies: []
    parallel_execution: true
    
  - id: "memory_search"
    type: "tool_call"
    description: "Search memory"
    dependencies: ["context_builder"]
    parallel_execution: true
    
  - id: "web_search"
    type: "search_web"
    description: "Search web"
    dependencies: ["context_builder"]
    parallel_execution: true
```

### Error Handling and Retries

Configure retry policies for robust execution:

```yaml
nodes:
  - id: "llm_worker"
    type: "llm_worker"
    description: "Generate response"
    dependencies: ["context_builder"]
    timeout_seconds: 300
    max_retries: 3
    retry_count: 0
```

## Workflow Management API

### Define Workflows Programmatically

```python
from ai.workflows.engine import WorkflowEngine
from ai.workflows.definitions import WorkflowDefinition

# Create workflow definition
definition = WorkflowDefinition(
    workflow_id="custom_workflow",
    nodes=[
        # Node definitions
    ],
    edges=[
        # Edge definitions
    ]
)

# Register workflow
engine = WorkflowEngine()
engine.register_workflow(definition)
```

### Execute Workflows via API

```python
from fastapi import FastAPI
from ai.workflows.engine import WorkflowEngine

app = FastAPI()
engine = WorkflowEngine()

@app.post("/api/v1/workflow/execute")
async def execute_workflow(workflow_id: str, context: TaskContext):
    """Execute a registered workflow"""
    result = await engine.execute_workflow_by_id(workflow_id, context)
    return result

@app.get("/api/v1/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow execution status"""
    status = await engine.get_workflow_status(workflow_id)
    return status
```

### List Available Workflows

```python
@app.get("/api/v1/workflows")
async def list_workflows():
    """List all available workflows"""
    workflows = await engine.list_workflows()
    return {"workflows": workflows}
```

## Best Practices

### 1. Keep Workflows Focused

Each workflow should have a single, well-defined purpose:

- **Good**: `code_review_workflow` - Reviews code for quality and security
- **Bad**: `general_workflow` - Tries to handle everything

### 2. Use Appropriate Node Types

- Use `router` for intent classification
- Use `context_builder` for context assembly
- Use `validator` for output validation
- Use `tool_call` for external tool integration
- Use `human_approval` for high-risk operations

### 3. Implement Proper Error Handling

- Set appropriate timeouts for each node
- Configure retry policies for unreliable operations
- Use validation gates to catch errors early
- Implement graceful degradation for failed nodes

### 4. Optimize for Performance

- Use parallel execution for independent nodes
- Minimize context size to reduce LLM costs
- Cache expensive operations when possible
- Use local processing when privacy allows

### 5. Ensure Security

- Validate all inputs at the `router` node
- Sanitize all outputs at the `validator` node
- Use `human_approval` for sensitive operations
- Implement rate limiting for external tool calls

## Troubleshooting

### Common Issues

1. **Workflow Execution Fails**
   - Check node dependencies are correctly defined
   - Verify all required services are running
   - Review error logs for specific failure reasons

2. **Context Size Too Large**
   - Implement context summarization in `context_builder`
   - Remove unnecessary context fields
   - Use context pruning strategies

3. **Performance Issues**
   - Enable parallel execution for independent nodes
   - Optimize LLM model selection
   - Implement caching for expensive operations

4. **Validation Failures**
   - Review validation rules in `validator` nodes
   - Check input format and content
   - Implement custom validation logic if needed

### Debugging Workflows

Enable detailed logging to debug workflow execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed execution logs
result = await engine.execute_workflow(task_context)
```

### Monitoring Workflows

Use the observability features to monitor workflow performance:

```python
from ai.observability import workflow_tracer

# Trace workflow execution
with workflow_tracer.trace_workflow("workflow_id"):
    result = await engine.execute_workflow(task_context)
```

## Examples

### Simple Chat Bot

```yaml
# workflows/simple_chat_bot.yaml
workflow_id: "simple_chat_bot"
nodes:
  - id: "router"
    type: "router"
    description: "Classify as chat"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build chat context"
    dependencies: ["router"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Generate response"
    dependencies: ["context_builder"]
    
  - id: "validator"
    type: "validator"
    description: "Validate response"
    dependencies: ["llm_worker"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
```

### Research Assistant

```yaml
# workflows/research_assistant.yaml
workflow_id: "research_assistant"
nodes:
  - id: "router"
    type: "router"
    description: "Identify research task"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build research context"
    dependencies: ["router"]
    
  - id: "search_web"
    type: "search_web"
    description: "Search for information"
    dependencies: ["context_builder"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Synthesize findings"
    dependencies: ["search_web"]
    
  - id: "validator"
    type: "validator"
    description: "Validate research output"
    dependencies: ["llm_worker"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "search_web"
  - from: "search_web"
    to: "tool_call"
  - from: "tool_call"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
```

### Code Reviewer

```yaml
# workflows/code_reviewer.yaml
workflow_id: "code_reviewer"
nodes:
  - id: "router"
    type: "router"
    description: "Identify code review task"
    dependencies: []
    
  - id: "context_builder"
    type: "context_builder"
    description: "Build code review context"
    dependencies: ["router"]
    
  - id: "llm_worker"
    type: "llm_worker"
    description: "Analyze code"
    dependencies: ["context_builder"]
    
  - id: "validator"
    type: "validator"
    description: "Validate code quality"
    dependencies: ["llm_worker"]
    
  - id: "tool_call"
    type: "tool_call"
    description: "Run static analysis"
    dependencies: ["validator"]

edges:
  - from: "router"
    to: "context_builder"
  - from: "context_builder"
    to: "llm_worker"
  - from: "llm_worker"
    to: "validator"
  - from: "validator"
    to: "tool_call"
```

## Conclusion

JARVISv3's workflow architecture provides a powerful and flexible foundation for building sophisticated AI assistants. By using the declarative YAML approach, non-developers can create and modify workflows, while the Python API enables complex, programmatic workflow management.

The key to success is understanding the node types, designing focused workflows, and implementing proper error handling and validation. With these principles, you can build workflows that are robust, maintainable, and scalable.

For more information, refer to the API documentation and explore the built-in workflow examples in the `workflows/` directory.

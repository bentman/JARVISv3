# JARVISv3 Workflow Documentation

This guide explains how JARVISv3 coordinates different agents using workflows to handle your daily tasks.

## Overview

JARVISv3 uses a **Workflow Graph** architecture. Instead of a simple chat loop, it breaks tasks down into specific steps (nodes) like searching the web, checking your local memory, or reviewing code.

This setup makes the assistant more reliable because:
- **Task-Specific Steps**: Each part of a task is handled by a node designed for that job.
- **Error Recovery**: If a search fails or a model times out, the graph can retry or use a fallback.
- **Visible Logic**: You can see exactly how the assistant reached an answer.

## Workflow Architecture

### Core Components

1. **Workflow Engine**: The background process that runs the graph.
2. **Nodes**: Individual steps (e.g., "Search", "Summarize", "Validate").
3. **Context**: The shared data that flows between nodes so they stay in sync.
4. **Validation**: Checks that happen at each step to catch hallucinations or errors.

### Node Types

| Node Type | Purpose | Description |
|-----------|---------|-------------|
| `router` | Intent Check | Figures out what you want to do (chat, code, research). |
| `context_builder` | Data Gathering | Pulls in your past conversations and local notes. |
| `llm_worker` | AI Processing | The actual "brain" that generates a response. |
| `validator` | Quality Check | Makes sure the answer isn't a hallucination and fits the format. |
| `tool_call` | Using Tools | Accesses your files or runs a script. |
| `search_web` | Finding Info | Grabs results from Google, Bing, or DuckDuckGo. |

## Creating Workflows

### Custom Node Logic (For Developers)

If you want to add a custom step to a workflow, you can do it in two ways:

#### 1. Quick Python Functions
Pass an `async` function to a `WorkflowNode`. 

```python
async def my_custom_logic(context: TaskContext, results: Dict[str, Any]) -> Dict[str, Any]:
    query = context.workflow_context.initiating_query
    return {"status": "ok", "custom_data": query}

node = WorkflowNode(
    id="custom_step",
    type=NodeType.LLM_WORKER,
    execute_func=my_custom_logic
)
```

#### 2. Declarative YAML
Define your workflow structure in a YAML file for easy reading:

```yaml
workflow_id: "my_daily_routine"
nodes:
  - id: "router"
    type: "router"
    dependencies: []
  - id: "search"
    type: "search_web"
    dependencies: ["router"]
  - id: "brain"
    type: "llm_worker"
    dependencies: ["search"]
```

## Built-in Workflows

### 1. Chat Workflow
The default path for general questions. It pulls context from your memory and generates a response with local models.

### 2. Research Workflow
Uses the `search_web` node to dig through multiple sources and the `validator` node to ensure the summary is accurate.

### 3. Code Assistant
Optimized for `coding` tasks. Uses specialized models (like Qwen2.5-Coder) and can audit local files without sending them to the cloud.

### 4. Voice Session
A high-performance path for hands-free use. It coordinates the STT, Chat Workflow, and TTS in one smooth loop.

## Best Practices

1. **Keep it Simple**: Use the `router` to keep workflows small and focused.
2. **Use Retries**: Enable retries for nodes that rely on the web or heavy models.
3. **Trust the Validator**: Always add a `validator` node for tasks where accuracy is critical.
4. **Run Locally**: Set your privacy level to "High" to ensure all nodes run on your own CPU/GPU.

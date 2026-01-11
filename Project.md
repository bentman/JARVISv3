# Project JARVISv3: Personal AI Assistant

## Vision
A personal AI assistant that runs on my hardware (Desktop + Laptop), handles my daily workflows (code, research, notes), and stays completely offline whenever possible. It's not a chatbot; it's a set of specialized agents that actually get things done.

*Note: This document represents the complete vision for JARVISv3. Current implementation progress is tracked in [SYSTEM_INVENTORY.md](./SYSTEM_INVENTORY.md).*

### Core Philosophy
- **Runs on My Hardware**: Optimized for my Desktop (x86_64 + NVIDIA GPU) and my Laptop (ARM + NPU).
- **Offline-First**: Default to local models. Internet is only for explicit web searches.
- **Task-Specific Brains**: Uses different LLMs for different tasks (e.g., small fast models for chat, larger capable models for code).
- **No Friction**: Starting it should be one command. Using it should be one hotkey.

## Core Capabilities

| Capability | Description |
| :--- | :--- |
| **Voice Interface** | Wake word (Jarvis), STT, and TTS pipeline. |
| **Code Assistant** | Specialized workflows for code review and refactoring. |
| **Research Node** | Aggregated search (DDG/Bing) with local memory context. |
| **Context Memory** | Remembers conversation history via local FAISS store. |
| **Hardware Routing**| Detects GPU/NPU availability and routes to optimal models. |
| **Conversation API**| Full history management and retrieval. |
| **Web Client** | React 18+ interface for browser access. |
| **Desktop Wrapper** | Tauri integration for system-level shortcuts. |

*For current implementation status, see [SYSTEM_INVENTORY.md](./SYSTEM_INVENTORY.md).*

### Vision vs. Reality

This document represents the **complete vision** for JARVISv3. The current implementation status is tracked separately in [SYSTEM_INVENTORY.md](./SYSTEM_INVENTORY.md).

**What's Working Now**: Backend infrastructure, workflow engine, core services, web client, and foundational AI workflows.

**What's in Progress**: Voice interface and desktop integration requires local testing.

**What's Next**: Full end-to-end functionality with local model execution and seamless hardware integration.

## System Architecture

JARVISv3 implements the "Unified Golden Stack" with a focus on defense-in-depth and observable execution.

### Workflow Architecture
JARVISv3 uses a **Workflow Graph** architecture. Instead of a simple chat loop, it breaks tasks down into specific steps (nodes) like searching the web, checking your local memory, or reviewing code.

This setup makes the assistant more reliable because:
- **Task-Specific Steps**: Each part of a task is handled by a node designed for that job.
- **Error Recovery**: If a search fails or a model times out, the graph can retry or use a fallback.
- **Visible Logic**: You can see exactly how the assistant reached an answer.

#### Core Components
1. **Workflow Engine**: The background process that runs the graph.
2. **Nodes**: Individual steps (e.g., "Search", "Summarize", "Validate").
3. **Context**: The shared data that flows between nodes so they stay in sync.
4. **Validation**: Checks that happen at each step to catch hallucinations or errors.

#### Node Types
| Node Type | Purpose | Description |
|-----------|---------|-------------|
| `router` | Intent Check | Figures out what you want to do (chat, code, research). |
| `context_builder` | Data Gathering | Pulls in your past conversations and local notes. |
| `llm_worker` | AI Processing | The actual "brain" that generates a response. |
| `validator` | Quality Check | Makes sure the answer isn't a hallucination and fits the format. |
| `tool_call` | Using Tools | Accesses your files or runs a script. |
| `search_web` | Finding Info | Grabs results from Google, Bing, or DuckDuckGo. |

#### Built-in Workflows
1. **Chat Workflow**: The default path for general questions. It pulls context from your memory and generates a response with local models.
2. **Research Workflow**: Uses the `search_web` node to dig through multiple sources and the `validator` node to ensure the summary is accurate.
3. **Code Assistant**: Optimized for `coding` tasks. Uses specialized models and can audit local files without sending them to the cloud.
4. **Voice Session**: A high-performance path for hands-free use. It coordinates the STT, Chat Workflow, and TTS in one smooth loop.

- **The Graph**: Tasks are DAGs. Reliability comes from retries and checkpoints.
- **Context**: Everything is typed (Pydantic). No "context drift" or messy strings.
- **Model Router**: Detects NVIDIA (CUDA), Apple (Metal), Intel/AMD (GPU), or Intel/AMD (CPU only) and routes accordingly.
- **Privacy**: Local-first isn't a feature; it's the architecture. PII is redacted before anything hits a web-search provider.

## Verification Pillar
System functionality is verified periodically via `scripts/validate_backend.py`, the authoritative backend validation tool. `scripts/validate_backend.py` remains the primary source of truth for backend validation.

- **Backend**: Core functionality validated through comprehensive, dynamically discovered test suite with per-test visibility.
- **Web Client**: Vitest suites for UI components.
- **Intelligence**: E2E smoke tests for real model inference.

---
*For development history, see [CHANGE_LOG.md](./CHANGE_LOG.md).*
*For future plans, see [CHANGE_ROADMAP.md](./CHANGE_ROADMAP.md).*

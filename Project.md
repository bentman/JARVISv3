# Project JARVISv3: Personal AI Assistant

## Vision
A personal AI assistant that runs on my hardware (Desktop + Laptop), handles my daily workflows (code, research, notes), and stays completely offline whenever possible. It's not a chatbot; it's a set of specialized agents that actually get things done.

### Core Philosophy
- **Runs on My Hardware**: Optimized for my Desktop (x86_64 + NVIDIA GPU) and my Laptop (ARM + NPU).
- **Offline-First**: Default to local models. Internet is only for explicit web searches.
- **Task-Specific Brains**: Uses different LLMs for different tasks (e.g., small fast models for chat, larger capable models for code).
- **No Friction**: Starting it should be one command. Using it should be one hotkey.

## Daily Workflow Capabilities

| Capability | Status | Description |
| :--- | :--- | :--- |
| **Voice Interface** | Implemented | Wake word (Jarvis), STT, and TTS. Requires external voice models. |
| **Code Assistant** | Implemented | Specialized node for code review and refactoring. Requires external LLM. |
| **Research Node** | Implemented | Aggregated search (DDG/Bing) with local memory context. Requires external search APIs. |
| **Context Memory** | Implemented and Exercised | Remembers what we talked about via local FAISS store. |
| **Hardware Routing**| Implemented | Detects GPU (NVIDIA/AMD/Intel) or CPU-only and picks the right model. Requires actual hardware. |
| **Conversation API**| Implemented and Exercised | Full history management via API. |
| **Desktop Wrapper** | Implemented | Tauri-based window with system shortcuts. |

## Development Status: All Roadmap Phases Completed

All 9 roadmap phases have been completed and validated through comprehensive testing:
- ✅ **Backend Infrastructure**: 112 tests passing across Unit, Integration, and Agentic categories
- ✅ **Workflow Architecture**: Agentic graph with dynamic planning, checkpoints, and state management
- ✅ **Context Management**: Typed, versioned context with lifecycle, validation, and lineage tracking
- ✅ **Hardware Optimization**: Resource-aware execution with dynamic memory management and graceful degradation
- ✅ **Human-AI Collaboration**: Approval nodes integrated into workflows with configurable criteria
- ✅ **Observability**: Comprehensive metrics, tracing, health monitoring, and circuit breaker patterns
- ✅ **Security & Privacy**: Multi-layer validation, PII redaction, and audit trails
- ✅ **Workflow Composability**: Template-based composition system with reusable workflow patterns
- ✅ **Production Readiness**: Full test coverage, error handling, and deployment optimization

See CHANGE_LOG.md for detailed implementation history and CHANGE_ROADMAP.md for the complete development plan.

## Technical "Under the Hood"

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
3. **Code Assistant**: Optimized for `coding` tasks. Uses specialized models (like Qwen2.5-Coder) and can audit local files without sending them to the cloud.
4. **Voice Session**: A high-performance path for hands-free use. It coordinates the STT, Chat Workflow, and TTS in one smooth loop.

- **The Graph**: Tasks are DAGs. Reliability comes from retries and checkpoints.
- **Context**: Everything is typed (Pydantic). No "context drift" or messy strings.
- **Model Router**: Detects NVIDIA (CUDA), Apple (Metal), Intel/AMD (GPU), or Intel/AMD (CPU only) and routes accordingly.
- **Privacy**: Local-first isn't a feature; it's the architecture. PII is redacted before anything hits a web-search provider.

## Verification Pillar
System functionality is verified periodically via `scripts/validate_backend.py`, the authoritative backend validation tool.
- **Backend**: Core functionality validated through comprehensive, dynamically discovered test suite with per-test visibility. (Maintenance: Ongoing - Deprecations removed opportunistically).
- **Frontend**: Vitest suites for UI components.
- **Intelligence**: E2E smoke tests for real model inference.

## Development Roadmap

See CHANGE_ROADMAP.md for the complete, sequenced development plan. All 9 roadmap phases have been completed and validated:

- ✅ **Phase 1-2**: Foundation (Workflow Architecture, Agent Registry)
- ✅ **Phase 3-4**: Context Management (Code-Driven Context, Layered Context Model)
- ✅ **Phase 5**: Operational Trustworthiness (Validation, Observability, Security)
- ✅ **Phase 6**: Contextual Intelligence (Active Memory, Context Evolution)
- ✅ **Phase 7**: Workflow Composability (Template System, Instant Composition)
- ✅ **Phase 8**: Resource-Aware Execution (Dynamic Memory, Hardware Optimization)
- ✅ **Phase 9**: Human-AI Collaboration (Approval Nodes, Decision Boundaries)

The roadmap provides explicit completion criteria, dependencies, and success metrics for each development phase. See CHANGE_LOG.md for implementation details and validation results.

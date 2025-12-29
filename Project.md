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

## 3-Month Personal Roadmap

### Month 1: Make It Usable
**Goal**: Use it daily without thinking about the setup.
- [x] Single-command startup (Docker for Desktop, Local for Laptop).
- [x] Hardware-aware model selection (GPU vs NPU) - requires actual hardware.
- [ ] System tray integration (always accessible).
- [ ] Auto-start on boot configuration.
- [ ] Cross-device conversation sync (SQLite + simple sync service).

### Month 2: Make It Reliable
**Goal**: Works perfectly even when conditions aren't.
- [x] Persistent memory across sessions (FAISS + metadata) - locally exercised.
- [x] Offline-first workflows (everything core works without internet) - requires external dependencies for AI features.
- [ ] Model download automation (checks, downloads, verifies) - requires external dependencies.
- [ ] Global hotkey for voice input (works app-unfocused) - requires external voice models.
- [ ] Degraded mode indicator (shows what works when offline) - requires external dependencies.

### Month 3: Make It Essential
**Goal**: Feel handicapped working without it.
- [ ] Task-specific model optimization (different LLMs for different nodes) - requires external LLMs.
- [ ] Research summarization workflow (deep dive + memory) - requires external LLMs.
- [x] Context pinning ("remember this project permanently") - implemented and locally exercised.
- [ ] Global conversation search ("what did we discuss about X?") - requires external dependencies.
- [ ] Voice-first UI refinement (faster than typing) - requires external voice models.

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
System functionality is verified periodically via `validation/validate_backend.py`.
- **Backend**: Core functionality validated through comprehensive test suite. (Maintenance: Ongoing - Deprecations removed in Sprint 1).
- **Frontend**: Vitest suites for UI components.
- **Intelligence**: E2E smoke tests for real model inference.

## 4-Phase Evolution Plan (2025-2026)

### Phase 1: Validation & Codebase Hygiene (COMPLETED)
- [x] Modernize dependencies (ddgs update).
- [x] Eliminate deprecation warnings (datetime.now(UTC), model_dump()).
- [x] Fix async/await issues in testing suite.
- [x] Standardize modern Python standards in AGENTS.md.

### Phase 2: Operationalizing Cyclic State Machines (COMPLETED)
- [x] Upgrade WorkflowEngine to support conditional cyclic edges.
- [x] Implement Reflector nodes for self-correction.
- [x] Refactor Code Assistant to use cyclic Write-Test-Fix loop.

### Phase 3: Dynamic Supervisor Routing (COMPLETED)
- [x] Implement Supervisor Agent for dynamic plan generation.
- [x] Enable runtime DAG construction based on user intent (Plan Queue).

### Phase 4: Deep Memory & Active Learning (COMPLETED)
- [x] Implement Active Memory nodes for mid-task learning - implemented but not exercised.
- [x] Add Context Pinning for long-term project persistence - implemented and locally exercised.

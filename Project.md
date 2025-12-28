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
| **Voice Interface** | Ready | Wake word (Jarvis), STT, and TTS. Works while I'm away from the keyboard. |
| **Code Assistant** | Working | Specialized node for code review and refactoring. |
| **Research Node** | Working | Aggregated search (DDG/Bing) with local memory context. |
| **Context Memory** | Ready | Remembers what we talked about via local FAISS store. |
| **Hardware Routing**| Ready | Detects GPU (NVIDIA/AMD/Intel) or CPU-only and picks the right model. |
| **Conversation API**| Ready | Full history management via API. |
| **Desktop Wrapper** | Working | Tauri-based window with system shortcuts. |

## 3-Month Personal Roadmap

### Month 1: Make It Usable
**Goal**: Use it daily without thinking about the setup.
- [x] Single-command startup (Docker for Desktop, Local for Laptop).
- [x] Hardware-aware model selection (GPU vs NPU).
- [ ] System tray integration (always accessible).
- [ ] Auto-start on boot configuration.
- [ ] Cross-device conversation sync (SQLite + simple sync service).

### Month 2: Make It Reliable
**Goal**: Works perfectly even when conditions aren't.
- [x] Persistent memory across sessions (FAISS + metadata).
- [ ] Offline-first workflows (everything core works without internet).
- [ ] Model download automation (checks, downloads, verifies).
- [ ] Global hotkey for voice input (works app-unfocused).
- [ ] Degraded mode indicator (shows what works when offline).

### Month 3: Make It Essential
**Goal**: Feel handicapped working without it.
- [ ] Task-specific model optimization (different LLMs for different nodes).
- [ ] Research summarization workflow (deep dive + memory).
- [ ] Context pinning ("remember this project permanently").
- [ ] Global conversation search ("what did we discuss about X?").
- [ ] Voice-first UI refinement (faster than typing).

## Technical "Under the Hood"
- **The Graph**: Tasks are DAGs. Reliability comes from retries and checkpoints.
- **Context**: Everything is typed (Pydantic). No "context drift" or messy strings.
- **Model Router**: Detects NVIDIA (CUDA), Apple (Metal), Intel/AMD (GPU), or Intel/AMD (CPU only) and routes accordingly.
- **Privacy**: Local-first isn't a feature; it's the architecture. PII is redacted before anything hits a web-search provider.

## Verification Pillar
System functionality is verified periodically via `scripts/validate_production.py`.
- **Backend**: 23+ tests covering core logic and feature parity. (Maintenance: Ongoing - Deprecations removed in Sprint 1).
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
- [x] Implement Active Memory nodes for mid-task learning.
- [x] Add Context Pinning for long-term project persistence.

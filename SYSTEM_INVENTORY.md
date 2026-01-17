# SYSTEM_INVENTORY.md

This file is the authoritative ledger of system capability state.  
Only what is listed here is considered true. All other documentation must defer to this file.

A capability must be classified into exactly one of the following states.

**Implemented and Locally Exercised**  
The capability has been executed end-to-end at least once in a real local runtime context, either through tests or direct execution.

**Implemented but Not Exercised**  
The capability exists in code but has not yet been executed end-to-end in a real local runtime context.

**Requires External Dependency**  
The capability exists in code but cannot be executed without external services, models, hardware, or infrastructure. Capabilities in this state must not be promoted using mocks, stubs, or simulated responses.

Promotion between states occurs only through an explicit completed mini-phase and must be reflected here immediately. Passive existence does not imply priority or readiness.

---

## State 1: Implemented and Locally Exercised

**Backend Infrastructure**
- FastAPI application with routing and middleware
- SQLite database with CRUD operations for conversations, messages, users
- Database tables: users, workflows, context_objects, budget_tracking, observability_logs, conversations, messages, workflow_checkpoints
- Conversation management: create, retrieve, delete, tag, statistics

**Workflow Engine**
- Agentic graph architecture with cyclic state machine
- Node execution with dependency management and error handling
- Workflow checkpoint saving and recovery

**Core Services**
- Memory service with conversation persistence
- Database manager with table creation and data operations
- Context schemas with validation
- Security validator with PII detection, SQL injection prevention, XSS protection, and input sanitization
- Ollama model provider with availability checking and response generation via a running Ollama server

**API Endpoints**
- GET /api/v1/conversations
- GET /api/v1/conversation/{id}
- DELETE /api/v1/conversation/{id}
- Health check endpoint

**Security and Validation**
- Budget tracking and cloud escalation
- Authentication and permission systems
- Privacy level controls

**Web Client (React)**
- React 18 + TypeScript application (Node 20, Port 3000)
- UI components (BudgetSummary, HardwareIndicator, SettingsModal, VoiceRecorder, WorkflowVisualizer)
- API service integration

**Desktop Wrapper (Tauri)**
- Tauri configuration and build setup
- System tray and global shortcut integration points
- Verified local exercise on Windows with `cd frontend && npm run tauri dev`, 2026-01-11 UTC, HEAD cf13e3cf

**AI Workflows**
- ChatWorkflow node graph
- Research workflow
- Development workflow (Code Assistant)
- Voice workflow integration (nodes and context)
- Agent registry and collaboration
- Search node functionality

**Model Management**
- ModelRouter provider selection (remaining providers)
- ModelManager download and verification logic
- Hardware detection and routing logic
- Model profiles (light, medium, heavy)

**Advanced Features**
- Distributed node registration and communication
- MCP server integration
- Hardware profiling and model selection
- Vector store with semantic search

**Contextual Intelligence**
- Active Memory nodes integrated into workflow execution
- Context evolution during multi-step task execution
- Intelligent adaptation based on learned execution patterns
- Pattern recognition and workflow optimization
- Dynamic context enhancement from memory operations

**Workflow Composability**
- Template-based workflow composition system
- Library of validated workflow templates (research, code_review, analysis)
- Instant composition of complex workflows from reusable components
- Parameter substitution and inter-template connections
- Template extension patterns for custom workflows

**Observability**
- Comprehensive metrics collection (requests, workflows, nodes, models, errors, resources)
- Prometheus-compatible metrics endpoint
- Workflow tracing with execution path visibility
- Health monitoring with automated checks
- Circuit breaker pattern for external service resilience
- Resource usage tracking (memory, CPU)

**Resource-Aware Execution**
- Dynamic GPU memory allocation and management across hardware types (NVIDIA CUDA, AMD, Intel)
- Hardware-specific acceleration detection (NPU variants: Apple Silicon M-series, Qualcomm ARM64, Intel)
- Graceful degradation handling with resource exhaustion detection
- Optimized model configurations based on detected hardware capabilities
- Cross-platform deployment optimization for CPU, GPU, and NPU environments

**Hardware Routing**
- CPU/GPU/NPU environment detection and tier classification (cpu/cloud verified)
- Dynamic model selection strategy based on available hardware resources

**Human-AI Collaboration**
- Approval nodes integrated into high-stakes workflows with workflow pausing/resuming
- Risk-based approval criteria evaluation (high-stakes operations always require approval)
- Auto-approval for low-risk operations with configurable confidence thresholds
- Comprehensive approval request data structures with context and decision criteria
- Workflow state management for approval-dependent execution flow

**Embedding Reliability Enhancement**
- Feature hashing embedding service with zero external dependencies for offline semantic search
- Unified embedding service with automatic fallback from transformer to feature hashing embeddings
- Deterministic embeddings suitable for approximate semantic search with L2 normalization
- Enhanced vector store with embedding strategy metadata and fallback search capability
- Seamless embedding strategy switching while maintaining search quality and reliability

**Infrastructure**
- Docker deployment configurations (Verified Dev/Prod parity)

---

## State 2: Implemented but Not Exercised

## State 3: Requires External Dependency

**AI Model Execution**
- llama.cpp provider with local GGUF model execution (Fixed: deterministic single-shot via `llama-completion`)
- Cloud LLM providers (API keys required)
- Model downloads from Hugging Face (network required)
- Local inference validated end-to-end

**Voice Services (Execution)**
- Voice Wake Word Detection (Requires openwakeword; ONNX models persisted to ./models)
- Voice Speech-to-Text (STT) (Requires Whisper executable and model weights)
- Voice Text-to-Speech (TTS) (Requires Piper executable and voice models)

**External Services**
- Search providers (DuckDuckGo, Bing, Google, Tavily)
- Redis caching
- SQLite database support

**Hardware Dependencies (Advanced)**
- Advanced hardware profiling (deep inspection beyond tier detection)

---

## Removed Capabilities (Historical)

These capabilities were removed or reverted. This section serves as historical context, not active capability state.

**Model Integrity Assurance** (REVERTED - Not Pursued)
- SHA256 checksum-based model integrity verification for corruption detection
- Pre-inference model validation with automatic corrupted file removal
- Checksum storage and retrieval system with persistent checksums.json
- Automated integrity checking during model loading across all providers
- Clear error reporting for corrupted model files with recovery guidance

*Rationale*: Implementation was incomplete (4/12 tests failing), introduced unnecessary complexity without proportional value for local development context, and provided false confidence rather than genuine reliability improvement. Reverted to simplify system and maintain operational stability.

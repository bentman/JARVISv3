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

---

## State 2: Implemented but Not Exercised

**Model Management**
- ModelRouter provider selection (remaining providers)
- ModelManager download and verification logic
- Hardware detection and routing logic
- Model profiles (light, medium, heavy)

**Voice Services**
- VoiceService STT and TTS pipeline
- Wake word detection
- Audio quality assessment
- Piper and espeak fallback mechanisms

**Security and Validation**
- Budget tracking and cloud escalation
- Authentication and permission systems
- Privacy level controls

**AI Workflows**
- ChatWorkflow node graph
- Research workflow
- Development workflow
- Agent registry and collaboration
- Search node functionality

**Advanced Features**
- Distributed node registration and communication
- MCP server integration
- Hardware profiling and model selection
- Vector store with semantic search

---

## State 3: Requires External Dependency

**AI Model Execution**
- llama.cpp provider (models and binaries required)
- Cloud LLM providers (API keys required)
- Model downloads from Hugging Face (network required)

**Voice Model Execution**
- Whisper STT models
- Piper TTS models
- Audio processing dependencies

**External Services**
- Search providers (DuckDuckGo, Bing, Google, Tavily)
- Redis caching
- PostgreSQL database support

**Hardware Dependencies**
- GPU/NPU detection and utilization
- Advanced hardware profiling

**Infrastructure**
- Docker deployment configurations
- Frontend React application

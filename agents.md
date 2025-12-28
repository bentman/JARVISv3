# JARVISv3: Agentic AI Coding Instructions & Rules

This document serves as the **source of truth** for all AI coding assistants (Cline, Gemini-CLI, Claude-Code, etc.) contributing to this repository. It defines the architectural "North Star," engineering standards, and operational protocols required to maintain the integrity of the JARVISv3 framework.

---

## 1. Architectural Mental Model: The Agentic Graph

Assistants must move beyond the "chatbot" paradigm. In JARVISv3, every interaction is a structured traversal of a **Workflow Graph**.

-   **Workflow Graph**: A Directed Acyclic Graph (DAG) or State Machine where nodes represent specific logic steps (Routing, Context Building, LLM Worker, Validation).
-   **Code-Driven Context**: Context is not an ad-hoc string; it is a typed, validated **Golden Context** object (Pydantic) managed through a lifecycle of assembly, summarization, and archival.
-   **Execution & Routing**: Intelligence is distributed. The system profiles hardware (CPU/GPU/NPU) and selects the optimal provider (Ollama, llama.cpp, or cloud) for each node.

---

## 2. Project Structure (v3 Implementation)

-   **`backend/ai/`**: The Core Intelligence module.
    -   `workflows/`: Engine, node definitions, and DAG orchestration.
    -   `context/`: Pydantic schemas and lifecycle management.
    -   `generators/`: Pluggable context builders (Memory, Hardware, Budget).
    -   `validators/`: Quality gates (Security, Budget, Code Check).
-   **`backend/core/`**: Integrated Services.
    -   `hardware.py`, `privacy.py`, `budget.py`, `voice.py`, `memory.py`.
-   **`backend/main.py`**: FastAPI entry point and API definitions.
-   **`frontend/`**: React 18 + TypeScript + Tailwind UI.
-   **`scripts/`**: Operational tools (deployment, validation, model management).
-   **`tests/`**: Integration and system-wide validation.

---

## 3. Engineering Standards (Mandatory)

To maintain high-fidelity implementation, assistants must adhere to:

### 3.1 Code Quality & Style (Naming Conventions)
-   **Python (Backend)**:
    -   Strict **PEP 8** compliance.
    -   File/Directory Names: `snake_case` (e.g., `model_router.py`).
    -   Class Names: `PascalCase` (e.g., `WorkflowEngine`).
    -   Function/Variable Names: `snake_case`.
    -   Typing: Mandatory type hints for all function signatures and complex variables.
    -   **Modernization**:
        -   Use `datetime.now(UTC)` instead of `datetime.utcnow()` (Deprecated).
        -   Use Pydantic `v2.model_dump()` instead of `v1.dict()` (Deprecated).
    -   Linting: Use `black` and `flake8` (refer to `.flake8` and `pyproject.toml`).
-   **TypeScript/React (Frontend)**:
    -   Component Names: `PascalCase` (e.g., `HardwareIndicator.tsx`).
    -   Hook/Function/Variable Names: `camelCase`.
    -   File Names: Match component name or `camelCase` for utilities.

### 3.2 Architectural Rigor (SOLID)
-   **Single Responsibility**: Keep Workflow Nodes and Services focused on one task.
-   **Dependency Injection**: Pass service instances into builders/workers to ensure testability. Avoid direct singleton imports inside deep logic where possible.
-   **Interface Segregation**: Use base classes (e.g., `BaseValidator`) to enforce consistent interfaces for pluggable components.

### 3.3 Security (OWASP & Privacy)
-   **Validation**: Every external input must be validated against a Pydantic schema.
-   **Cryptography**: Never use custom crypto. Rely on standard libraries (`passlib`, `cryptography`).
-   **Privacy**: Favor local-first processing. All cloud escalation must be gated by `PrivacyLevel` and `Budget` checks.

### 3.4 Quality Model (ISO 25010)
-   Prioritize **Maintainability** (modular code) and **Reliability** (error handling/graceful degradation) in every PR.

### 3.5 Anti-Sprawl & Test Fidelity (Mandatory)
-   **Zero Redundancy**: Do not create new scripts for functionality that exists in `validate_production.py`. Consolidate legacy or redundant tests into the unified suite.
-   **High Fidelity Testing**: Tests must not be "pass-through" placeholders. 
    -   Frontend tests must verify component mounting and dependency resolution (e.g., `QueryClient`).
    -   Backend tests must verify state transitions, database side-effects, and schema adherence.
-   **Honest Reporting**: Use `SKIPPED` status for tests missing local hardware/models (e.g., AI inference). Never mark a component as `PASSED` if it was only mocked without architectural verification.

---

## 4. Operational Protocols

### 4.1 Operational Command Suite (Executable Instructions)
Agents must use the following commands for validation and execution:

| Action | Command |
| :--- | :--- |
| **System Validation** | `./backend/.venv/Scripts/python scripts/validate_production.py` |
| **Run Unit Tests** | `./backend/.venv/Scripts/python -m pytest backend/tests/` |
| **Linting (Python)** | `./backend/.venv/Scripts/flake8 backend/` |
| **Formatting (Python)**| `./backend/.venv/Scripts/black backend/` |
| **Start Backend** | `./backend/.venv/Scripts/python backend/main.py` |
| **Start Frontend** | `cd frontend && npm run dev` |

### 4.2 Environment Isolation (Critical)
**Always** use the project virtual environment. Never install packages globally.
-   **Activate**: `backend/.venv` (Windows: `Scripts/Activate.ps1`, Unix: `bin/activate`).
-   **Execution**: Prefix Python commands with the venv path (e.g., `./backend/.venv/Scripts/python` on Windows or `./backend/.venv/bin/python` on Unix). 
-   **Note**: All commands in this document use Windows paths by default for compatibility with the primary development environment.

### 4.2 Documentation & Script Integrity (No Sprawl)
-   **Documents**: Avoid creating new markdown files. Update `Project.md`, `README.md`, or `agents.md` instead.
-   **Scripts**: Do not create ad-hoc verification scripts (e.g., `verify_stage.py`). Integrate all system checks into `scripts/validate_production.py` to maintain a single source of truth for system health.

### 4.3 Common Implementation Tasks
-   **Adding a Workflow Node**:
    1.  Define node in `backend/ai/workflows/`.
    2.  Update `NodeType` enum in `schemas.py`.
    3.  Register in `WorkflowEngine`.
-   **Adding a Context Generator**:
    1.  Implement in `backend/ai/generators/`.
    2.  Register with `ContextBuilder`.
-   **New API Endpoint**: Add to `backend/main.py` using versioned paths (`/api/v1/...`).

### 4.3 Definition of Done (DoD)
A feature is only `ready-to-use` if it satisfies the **Unified Validation Pillars**:
1.  **Backend Core**: All `pytest` suites pass and `database_manager` is initialized.
2.  **Frontend UX**: `npm run test` passes with zero failures in the React layer.
3.  **AI Intelligence**: `backend/tests/test_e2e_model.py` passes (or is documented as SKIPPED if models are missing, but never FAILED).
4.  **Linting**: Static analysis (`mypy`, `flake8`) shows zero critical errors.
5.  **Security**: Input validation is enforced; no hardcoded secrets or insecure crypto.
6.  **Documentation**: `Project.md` roadmap is updated; code is self-documenting.

**Note**: All pillars are verified simultaneously by running `./backend/.venv/Scripts/python scripts/validate_production.py`.

---

## 5. Agent Readiness & Truthfulness

When reporting status, use this taxonomy:
-   `ready-to-use`: Implemented, tested, and meets all standards in Sec 3.
-   `partial`: Functional but missing specific quality gates or tests.
-   `planned`: Documented in `Project.md` but not yet implemented.

*Note: Never claim "ready-to-use" without providing verification evidence (test outputs/logs).*

---

## 6. Governance & Conflict Resolution

### 6.1 Constraint Handling Protocol
If a user request conflicts with the engineering standards defined in this document or the architectural intent in `Project.md`:
1.  **Flag the Trade-off**: Explicitly state: *"This approach trades [standard/attribute] for [user's goal]"*.
2.  **Offer Alternative**: Briefly present the standard-compliant approach.
3.  **Wait for Confirmation**: Do not proceed with sub-standard implementation without explicit user acknowledgment of the risk.

### 6.2 Documentation First
No architectural changes or new core services should be implemented without first updating the corresponding section in `Project.md` or `agents.md`.

---

## 6. Reference Material

-   **[Project.md](Project.md)**: Current requirements, roadmap, and core capabilities.
-   **[JARVISv2 Reference](https://github.com/bentman/JARVISv2)**: An optional legacy reference containing earlier logic and patterns from the initial refactoring phase. Use it only for inspiration on service‑level behavior while following JARVISv3’s workflow architecture. For local development, it is commonly exposed via a symlink inside the JARVISv3 folder as `.\JARVISv2_ref`.

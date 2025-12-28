# JARVISv3: Agentic AI Coding Instructions & Rules

This document serves as the **source of truth** for all AI coding assistants (Cline, Gemini-CLI, Claude-Code, etc.) contributing to this repository. It defines the architectural "North Star", engineering standards, and operational protocols required to maintain the integrity of the JARVISv3 framework.

---

## 1. Architectural Mental Model: The Agentic Graph

Assistants must move beyond the "chatbot" paradigm. In JARVISv3, every interaction is a structured traversal of a **Workflow Graph**.

* **Workflow Graph**
  A Directed Acyclic Graph (DAG) or State Machine where nodes represent specific logic steps (Routing, Context Building, LLM Worker, Validation).

* **Code-Driven Context**
  Context is not an ad-hoc string; it is a typed, validated **Golden Context** object (Pydantic) managed through a lifecycle of assembly, summarization, and archival.

* **Execution & Routing**
  Intelligence is distributed. The system profiles hardware (CPU/GPU/NPU) and selects the optimal provider (Ollama, llama.cpp, or cloud) for each node.

---

## 2. Project Structure (v3 Implementation)

* **`./backend/ai/`** — Core intelligence
  * `./workflows/` — Engine, node definitions, DAG orchestration
  * `./context/` — Pydantic schemas and lifecycle management
  * `./generators/` — Pluggable context builders (Memory, Hardware, Budget)
  * `./validators/` — Quality gates (Security, Budget, Code Check)

* **`./backend/core/`** — Integrated services
  hardware.py, privacy.py, budget.py, voice.py, memory.py

* **`./backend/main.py`** — FastAPI entry point and API definitions

* **`./frontend/`** — React 18 + TypeScript + Tailwind UI

* **`./scripts/`** — Operational tools (deployment, validation, model management)

* **`./tests/`** — Integration and system-wide validation

* **`./reports/`** — Integration and system-wide validation test results
---

## 3. Engineering Standards (Mandatory)

### 3.1 Code Quality & Style

**Python (Backend)**

* Strict PEP-8 compliance
* File and directory names: snake_case
* Class names: PascalCase
* Functions and variables: snake_case
* Mandatory type hints for all public functions and complex variables

**Modernization Requirements**

* Use `datetime.now(UTC)` instead of `datetime.utcnow()` (deprecated)
* Use Pydantic v2 `model_dump()` instead of v1 `dict()`

**Tooling**

* Formatting: black
* Linting: flake8
* Configuration: `./.flake8`, `./pyproject.toml`

**TypeScript / React (Frontend)**

* Component Names: `PascalCase` (e.g., `HardwareIndicator.tsx`).
* Hook/Function/Variable Names: `camelCase`.
* File Names: Match component name or `camelCase` for utilities.

### 3.2 Architectural Rigor (SOLID)

* **Single Responsibility**
  Workflow nodes and services must do exactly one thing.

* **Dependency Injection**
  Pass services explicitly to enable testability; avoid deep singleton imports.

* **Interface Segregation**
  Use base classes (e.g., BaseValidator) for pluggable components.

### 3.3 Security (OWASP & Privacy)

* All external input must be validated via Pydantic schemas.
* Never implement custom cryptography.
* Use standard libraries (passlib, cryptography).
* Favor local-first processing.
* All cloud escalation must be gated by PrivacyLevel and Budget checks.

### 3.4 Quality Model (ISO-25010)

Prioritize **Maintainability** (modular code) and **Reliability** (error handling/graceful degradation) in every PR.

### 3.5 Anti-Sprawl & Test Fidelity

* **Zero Redundancy**
  Do not create new scripts for functionality covered by `validation/validate_backend.py` or `validation/validate_frontend.py`.

* **High-Fidelity Testing**

  * Backend tests must verify state transitions, DB side-effects, and schema adherence.
  * Frontend tests must verify component mounting and dependency resolution.

* **Honest Reporting**

  * Use SKIPPED for missing hardware or models.
  * Never mark PASSED if functionality was only mocked.

---

## 4. Operational Protocols

### 4.1 Operational Command Suite (Executable Instructions)

| Action              | Command                                                       |
| ------------------- | --------------------------------------------------------------- |
| System Validation   | `./backend/.venv/Scripts/python validation/validate_backend.py` |
| Run Backend Tests   | `./backend/.venv/Scripts/python -m pytest backend/tests/`       |
| Linting (Python)    | `./backend/.venv/Scripts/flake8 backend/`                       |
| Formatting (Python) | `./backend/.venv/Scripts/black backend/`                        |
| Start Backend       | `./backend/.venv/Scripts/python backend/main.py`                |
| Start Frontend      | `pushd frontend && npm run dev`                                 |

---

### 4.2 Environment Isolation (Critical)
**Always** use the project virtual environment. Never install packages globally.
-   **Activate**: `backend/.venv` (Windows: `Scripts/Activate.ps1`, Unix: `bin/activate`).
-   **Execution**: Prefix Python commands with the venv path (e.g., `./backend/.venv/Scripts/python` on Windows or `./backend/.venv/bin/python` on Unix). 

### 4.3 Documentation & Script Integrity

* Avoid creating new markdown files unnecessarily.
* Update Project.md, README.md, or AGENTS.md instead.
* Do not create ad-hoc verification scripts.
* All validation flows through `validation/validate_backend.py` or `validation/validate_frontend.py`.

---

### 4.4 Common Implementation Tasks

**Adding a Workflow Node**

1. Implement in `./backend/ai/workflows/`
2. Update `NodeType` enum in `schemas.py`.
3. Register in `WorkflowEngine`

**Adding a Context Generator**

1. Implement in `/backend/ai/generators/`
2. Register with `ContextBuilder`

**Adding an API Endpoint**

* Implement in `./backend/main.py` using versioned paths (`/api/v1/…`)

---

### 4.5 Definition of Done (DoD)

A feature is ready-to-use when:

1. Backend tests pass and database initializes
2. Frontend tests pass
3. AI tests pass or are explicitly SKIPPED due to missing models
4. Linting shows no critical errors
5. Security validation is enforced
6. Documentation reflects actual implementation

All pillars are verified via `/validation/validate_backend.py` or `/validation/validate_frontend.py`.

---

## 5. Agent Readiness & Truthfulness

Use the following taxonomy:

* `ready-to-use` — Implemented, exercised, and dependency-complete
* `partial` — Implemented but missing dependencies or gates
* `planned` — Documented but not implemented

*!IMPORTANT!: Never claim "ready-to-use" without verification evidence (test outputs/logs).*

---

## 6. Governance & Conflict Resolution

### 6.1 Constraint Handling Protocol

If a request conflicts with standards or architectural intent:

1. Flag the trade-off
2. Offer a compliant alternative
3. Wait for explicit user confirmation before proceeding

---

### 6.2 Documentation First

No architectural changes or new core services may be implemented without first updating `Project.md` or `AGENTS.md`.

---

## 7. Non-Negotiable Invariants

### 7.1 System State Classification

The authoritative system inventory is maintained in `./SYSTEM_INVENTORY.md`.
Any capability whose state changes must be updated there immediately.

All system capabilities must be classified into exactly one factual state:
- **Implemented and Locally Exercised**: Code exists and has been executed end-to-end at least once in a real local runtime (tests or direct execution).
- **Implemented but Not Exercised**: Code exists but has not yet been executed end-to-end locally.
- **Requires External Dependency**: Code exists but cannot be executed without external services, models, hardware, or infrastructure.

No additional states, percentages, or readiness language are permitted.

### 7.2 Documentation Accuracy

* Documentation must not exceed the inventory.
* Do not claim “works", “ready", or “complete” when dependencies exist.
* Use precise, factual language only: “Implemented”, “Locally Exercised", or “Requires External Dependency”.
* If documentation conflicts with the inventory, the inventory is correct and documentation must be corrected.

### 7.3 Validation Requirements

* All claims must be backed by observable execution.
* Capabilities without execution remain Implemented but Not Exercised.
* External dependencies must not be promoted using mocks, stubs, placeholders, or simulated responses.

### 7.4 Change Recording

* Completed changes affecting behavior or capability state must be recorded in `./CHANGE_LOG.md`.
* Capability promotions must update both `./SYSTEM_INVENTORY.md` and `./CHANGE_LOG.md` in the same change set.

---

## 8. Reference Material

* [Project.md](Project.md) — Current requirements, roadmap, and core capabilities
* [JARVISv2](https://github.com/bentman/JARVISv3) Reference — Legacy patterns for inspiration only
* [jarvis](https://github.com/bentman/jarvis) Historical only - Start of the journey (do not use!)

# JARVISv3: Agentic AI Coding Instructions & Rules

This document establishes standards for developers and AI agents working on JARVISv3. It defines project structure, engineering practices, and operational protocols to ensure consistency and reliability.

---

## 1. Project Structure

### Directory Layout
```
backend/
├── ai/           # Core intelligence (workflows, context, generators, validators)
├── core/         # Integrated services (hardware, privacy, budget, voice, memory)
├── main.py       # FastAPI application entry point
└── requirements.txt

frontend/         # React 18 + TypeScript application
scripts/          # Operational tools (validation, deployment)
tests/            # Integration and unit tests
reports/          # Validation output and logs
```

### File Naming Conventions
- **Python**: snake_case for files, functions, variables; PascalCase for classes
- **TypeScript**: PascalCase for components; camelCase for functions, variables
- **Scripts**: snake_case, descriptive names, idempotent operations

---

## 2. Engineering Standards

### Code Quality
- **Python**: PEP 8 compliance, mandatory type hints, no critical flake8/mypy errors
- **TypeScript**: Strict typing, PascalCase components, camelCase functions
- **KISS Principle**: Keep code simple and direct
- **Separation of Concerns**: One responsibility per module, function, or class
- **Technical-Debt Hygiene**: Deprecation warnings, obsolete APIs, and forward-incompatibility risks must be resolved opportunistically when encountered, with priority on correctness and minimal scope

### Architecture Principles
- **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- **Dependency Injection**: Pass services explicitly to enable testability
- **Idempotency**: Operations must be safe to re-run without side effects
- **Deterministic Operations**: Prioritize repeatability and predictability

### Security & Privacy
- **Input Validation**: All external input validated via Pydantic schemas
- **Local-First**: Process data locally when possible, minimize external dependencies
- **PII Protection**: Automatic redaction of personally identifiable information

### Testing Standards
- **No Pass-Throughs**: Tests must verify actual state transitions and side effects
- **Honest Reporting**: Use SKIPPED for missing dependencies, never PASSED on mocks
- **YAGNI Principle**: Add tests only for current functionality, avoid speculative coverage

---

## 3. Operational Protocols

### Command Suite
Use these exact commands for consistent operations:
- System Validation: `./backend/.venv/Scripts/python scripts/validate_backend.py`
- Unit Tests: `./backend/.venv/Scripts/python -m pytest tests/`
- Backend Start: `./backend/.venv/Scripts/python backend/main.py`
- Frontend Start: `cd frontend && npm run dev`

### Environment Management
- **Virtual Environment**: Always use `backend/.venv` for Python operations
- **Dependency Isolation**: Never install packages globally
- **Path Consistency**: Use relative paths from project root

### Definition of Done
A feature is complete when:
1. Backend tests pass
2. Frontend tests pass (zero failures)
3. Linting shows no critical errors
4. Documentation updated
5. Validation script confirms functionality

### Anti-Sprawl Rules
- **Zero Redundancy**: Do not create duplicate scripts or functionality
- **Single Source of Truth**: Use designated files for each purpose
- **Script Simplicity**: Keep scripts focused on one task, avoid complexity

---

## 4. System State Classification

### Required States
Every capability must be classified in exactly one state in `SYSTEM_INVENTORY.md`:
- **Implemented and Locally Exercised**: Executed end-to-end in local runtime with test coverage
- **Implemented but Not Exercised**: Code exists but not validated locally
- **Requires External Dependency**: Code exists but needs external services/models/hardware

### Promotion Rules
- Capabilities promoted only through completed mini-phases
- Update `SYSTEM_INVENTORY.md` immediately after promotion
- One capability per mini-phase maximum
- No promotion using mocks, stubs, or simulated responses for external dependencies
- Granularity must match automated test coverage

### Documentation Discipline
- Documentation must not exceed inventory claims
- Use precise language: "Implemented", "Locally Exercised", or "Requires External Dependency"
- Inventory takes precedence over all other documentation

---

## 5. Governance

### Conflict Resolution
When requests conflict with standards:
1. Flag the specific trade-off
2. Offer compliant alternative
3. Wait for explicit confirmation before proceeding

### Documentation First
Update `Project.md` or `AGENTS.md` before implementing architectural changes.

### Change Logging
Maintain `CHANGE_LOG.md` with factual records of completed changes:
- Append-only entries
- Short, factual descriptions
- Timestamp + change + evidence format
- Log capability promotions and observable behavior changes only

---

## 6. Mini-Phase Technique

### Purpose
Mini-phases control agent behavior and prevent scope creep through narrowly scoped instructions.

### Core Rules
- **Action-Only**: Execute only explicitly stated actions
- **Single Purpose**: One goal per mini-phase
- **No Implied Authority**: Forbidden actions must be stated
- **Explicit Stop Condition**: Clear completion criteria

### Format Requirements
- Clear title describing intent
- Paragraph-form instructions
- Explicit prohibitions when needed
- Integration with existing rules, not additions

### Example
**Mini-Phase: Promote Security Validator**

Select the security validator from State 2 in `SYSTEM_INVENTORY.md`. Execute end-to-end tests for PII detection, SQL injection prevention, XSS protection, and input sanitization. Update inventory and change log upon successful test completion. Do not modify other files.

---

## 7. Reference Material

- [Project.md](Project.md) — Requirements and roadmap
- [SYSTEM_INVENTORY.md](SYSTEM_INVENTORY.md) — Capability state authority
- [CHANGE_LOG.md](CHANGE_LOG.md) — Historical change record
- [.roo/rules/](.roo/rules/) — Source rule definitions

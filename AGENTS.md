# AGENTS.md — JARVISv3 Working Agreements (Canonical)

This file is the single authoritative instruction set for AI agents and contributors working in this repository. AGENTS.md is intended to function like a “README for agents” (clear, predictable, and tool-agnostic), and should stay concise to avoid drift and tool-specific sprawl.

## 0 Non-Negotiables

Do not guess or assume - use research if in question.
If you cannot verify something from the repo, say what you checked and ask what to do next.
No “completed/verified” claims without reproducible evidence.
No scope expansion. Do the minimum required for the requested outcome, then stop.
Do not introduce new repo artifacts unless explicitly requested.
Avoid documentation "sprawl" - integrate where appropriate, not append.
Avoid code "sprawl" - reuse existing patterns/files where appropriate, not append.
Architecture claims must describe the system as it exists, not as intended.
No parallel architectures, experimental stacks, or shadow systems.

## 1 Governance and Truth Sources

Authoritative sources (priority order):

* AGENTS.md — rules for work in this repo (this file).
* SYSTEM_INVENTORY.md — capability truth ledger (“what is true now”).
* CHANGE_LOG.md — append-only history of completed work, with evidence.
* CHANGE_ROADMAP.md — forward plan only; must not contradict inventory truth.
* Project.md — intent and design framing; must not exceed inventory truth.

## 2 Capability State Classification (SYSTEM_INVENTORY.md)

Every capability must be listed in exactly one state in SYSTEM_INVENTORY.md:

* Implemented and Locally Exercised
* Implemented but Not Exercised
* Requires External Dependency

Rules:

* No readiness language and no percentages.
* Promotions require automated test execution or real local runtime execution.
* Do not promote capabilities based on mocks/stubs when the real dependency is missing.
* Partial execution does not promote the whole capability; keep granularity honest.
* Reverted or intentionally skipped work must be reflected explicitly (do not invent a fourth state).

## 3 CHANGE_LOG.md Rules (Slim Format)

CHANGE_LOG.md records what changed after it was done. It is not a plan, status report, or justification.

Core rules:

* Append-only: never rewrite, reorder, or delete existing entries.
* Log after completion: do not pre-log planned work.
* Completed and committed changes only.
* No future tense or speculative language.
* Keep entries factual and lightweight; include a brief evidence note.

Entry header format (required): `### YYYY-MM-DD HH:MM UTC - Short description`

Entry body (required, slim):

* One or two short lines describing what changed.
* Evidence note (one line): the command(s) run and the outcome (PASS / PASS_WITH_SKIPS / SKIPPED / FAIL), or a precise static verification if runtime is not possible.

If nothing materially changed, do not log.
Promotions, reversions, and corrections must be logged.

If you suspect CHANGE_LOG.md contains an inaccurate entry:

* do not edit history
* append a corrective entry stating what was claimed, what is actually true, and the evidence

Correction/Reversion Entry Patterns:

For corrections of inaccurate prior entries:
```
### YYYY-MM-DD HH:MM UTC — Correction: [brief description]
- Prior entry [timestamp] claimed [what was inaccurately claimed].
- Actual state: [factual current truth].
- Evidence: [command/outcome or static verification].
```

For reversions of completed changes:
```
### YYYY-MM-DD HH:MM UTC — Reverted: [change description]
- Reversion reason: [why change was reverted].
- Changes undone: [what was removed/reverted].
- Evidence: [command/outcome showing reversion].
```

## 4 Git Safety Rules

Never run destructive git commands without explicit approval:

* `git restore`
* `git reset`
* `git clean`
* `git rebase`
* history rewrites

If rollback is requested, first determine whether changes are committed or uncommitted and propose the safest approach.

## 5 Mini-Phase Technique

All work is done via mini-phases.

Mini-phase title convention: `Mini-Phase: <short intent>`

Mini-phase requirements:

* One purpose.
* Action-only (no speculative planning mixed in).
* Explicit stop condition.
* No implicit follow-on work.
* Complete one mini-phase before starting another.

Completion rules:

* Do not log completion until the scope-appropriate validation has passed (see Validation section).
* Do not promote capabilities early.
* If a change is reverted, it must be explicitly recorded.

## 6 Execution Harness (Control Against Churn)

For any non-trivial work:

Discover
Identify the smallest relevant set of files and the authoritative commands. Note what you inspected.

Verify
Reproduce the issue or confirm current behavior using the smallest valid run (or a static trace if runtime isn’t available). Capture the key evidence.

Explain
State the root cause and what “fixed” must look like (observable outcomes).

Propose (approval gate)
Before editing, provide:

* exact files you will touch
* exact commands you will run
* exact evidence you expect to observe
  Then stop and wait for explicit approval.

Implement
Apply only what was approved.

Validate and report
Run only the agreed commands, capture outcomes, and stop. If the failure mode does not change, stop and report (do not iterate silently).

## 7 Environment Management and Python Execution

### Environment Setup
- All Python commands must use the project virtual environment: `./backend/.venv/Scripts/python`
- Initialize venv if missing: `python -m venv backend/.venv`
- Install dependencies: `./backend/.venv/Scripts/python -m pip install -r backend/requirements.txt`
- Platform note: Works on Windows (PowerShell 7), macOS, and Linux

### Command Execution Pattern
- Always use full venv path: `./backend/.venv/Scripts/python [command]`
- Examples:
  - Run backend: `./backend/.venv/Scripts/python backend/main.py`
  - Run tests: `./backend/.venv/Scripts/python -m pytest tests/`
  - Run validation: `./backend/.venv/Scripts/python scripts/validate_backend.py`

### Why Virtual Environment
- Isolates project dependencies from system Python
- Ensures consistent behavior across development environments
- Prevents conflicts with global packages
- Required for proper dependency resolution

## 8 Validation and Execution (Authoritative)

Standards:

* Unit tests are offline, deterministic, and dependency-free.
* Integration tests may use real dependencies but must SKIP cleanly when unavailable.
* Agentic tests must not silently depend on external services; if dependencies are required, they must SKIP cleanly.
* Tests validate observable behavior (not implementation details). Skips are not failures.
* Deprecation warnings are technical debt and must be tracked (do not ignore).

Test layout:

* `tests/unit/`, `tests/integration/`, `tests/agentic/`

Workflow for any code change (including troubleshooting):

* Add or update the appropriate unit/integration/agentic test for the change.
* Iterate locally on the smallest target until it passes:

  * `./backend/.venv/Scripts/python -m pytest <path_to_test> -q`
* Before claiming completion, run the full test suite:

  * `./backend/.venv/Scripts/python -m pytest tests/`
* If backend validation applies to the scope, run it last to generate the authoritative evidence report in `./reports/`:

  * `./backend/.venv/Scripts/python scripts/validate_backend.py`
* Do not log completion until the scope-appropriate tests pass and (when applicable) the validation report exists.

Authoritative run commands (repo root):

* Backend start: `./backend/.venv/Scripts/python backend/main.py`
* Frontend start: `pushd frontend && npm run dev`

Notes:

* Do not assume `make` is installed on Windows. If a Makefile exists, treat it as optional convenience.
* When verifying servers, start, confirm readiness via logs or a simple request, then stop unless instructed to keep it running.

## 9 Data and Artifact Conventions

Runtime artifacts must not live in the repo root.

Preferred location is `./data/` for persistent runtime artifacts such as:

* SQLite DB
* vector store files
* checkpoints
* caches and similar generated state

Rules:

* Do not introduce new storage locations without explicit agreement.
* Ensure ignores cover generated artifacts (git + docker contexts) without being overly broad.

## 10 Docker Conventions (Dev vs Hardened)

This repo uses two compose tracks:

* `docker-compose.dev.yml` (development runner)

  * Fast iteration, relaxed posture
  * Intended to be left running for ongoing development visibility
* `docker-compose.yml` (hardened runner)

  * Security posture and operational constraints

Parity rule:

* Dev and hardened should match core capabilities and wiring; posture differences only.
* If they drift, propose the minimal changes needed to restore parity.

Preferences:

* SQLite-first for default persistence unless the repo explicitly documents otherwise.
* Redis (if present) must degrade gracefully when unavailable and must not block startup.

## 11 Documentation Discipline

Documentation must not exceed SYSTEM_INVENTORY.md claims.
Do not update docs “because it should be true” only because "there is evidance supporting it".
Update docs only when behavior is evidenced and the document is an authoritative consumer of that claim.

## 12 Output Format

When reporting back:

* Summary (1–3 sentences)
* Files inspected and/or touched
* Commands executed and outcomes
* Evidence excerpt(s) (minimal, copy/pasteable)
* Proposal for next step (scoped), or stop if complete

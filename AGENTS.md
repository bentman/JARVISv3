# AGENTS.md — JARVISv3 Working Agreements (Canonical)
This file is the single authoritative instruction set for AI agents and contributors working in this repository. AGENTS.md is intended to function like a “README for agents” (clear, predictable, and tool-agnostic), and should stay concise to avoid drift and tool-specific sprawl.

## 0 Non-Negotiables
Do not guess. If you cannot verify something from the repo, say what you checked and ask what to do next.
No “completed/verified” claims without reproducible evidence.
No scope expansion. Do the minimum required for the requested outcome, then stop.
Avoid sprawl. Reuse existing patterns/files; do not introduce new repo artifacts unless explicitly requested.

## 1 Governance and Truth Sources
Authoritative sources (priority order):
- AGENTS.md — rules for work in this repo (this file).
- SYSTEM_INVENTORY.md — capability truth ledger (“what is true now”).
- CHANGE_LOG.md — append-only history of completed work, with evidence.
- CHANGE_ROADMAP.md — forward plan only; must not contradict inventory truth.
- Project.md — intent and design framing; must not exceed inventory truth.
If reference rules conflict with AGENTS.md, follow AGENTS.md and call out the mismatch.

## 2 Capability State Classification
Every capability must be listed in exactly one state in SYSTEM_INVENTORY.md:
- Implemented and Locally Exercised
- Implemented but Not Exercised
- Requires External Dependency
Rules:
- No readiness language and no percentages.
- Do not promote capabilities based on mocks/stubs when the real dependency is missing.
- Promotions require reproducible evidence on the current repo state.

## 3 Mini-Phase Technique
All work is done via mini-phases.
Mini-phase title convention:
`Mini-Phase: <short intent>`
Mini-phase requirements:
- One purpose.
- Action-only (explicit actions; no implied authority).
- Explicit stop condition.
- No implicit follow-on work.

## 4 Execution Harness (Control Against Churn)
For any non-trivial work, use this workflow:
Discover: Inspect repo state, locate the relevant files, and identify the authoritative commands.
Analyze: Provide a root-cause narrative grounded in observed evidence (or static trace if runtime isn’t available).
Troubleshoot (optional): Only if required to confirm the diagnosis; keep it minimal.
Verify conditions: Confirm expected vs actual behavior using the smallest valid run.
Summary: State what is wrong, why, and what must be true after the fix.
Proposal (approval gate):
Before editing, provide:
- exact file list you will touch
- exact commands you will run
- exact evidence you expect to observe
Then stop and wait for explicit approval.
Implement:
Apply only what was approved.
Verify and evidence: Run only the agreed commands and capture outcomes.
Document: Update CHANGE_LOG.md and/or SYSTEM_INVENTORY.md only when evidence supports the claim.
Hard stop rule: If a fix attempt does not change the failure mode, stop and report. Do not iterate silently.

## 5 Git Safety Rules
Never run destructive git commands without explicit approval:
- `git restore`
- `git reset`
- `git clean`
- `git rebase`
- history rewrites
If rollback is requested, first determine whether changes are committed or uncommitted and propose the safest approach.
If you suspect CHANGE_LOG.md contains an inaccurate entry:
- do not edit history
- append a corrective entry (see CHANGE_LOG rules below)

## 6 CHANGE_LOG.md Rules (Formatting + Evidence)
CHANGE_LOG.md is a human-readable record of noteworthy, completed work. Maintaining a changelog is a common best practice, but it only stays useful if it is factual and readable. :contentReference[oaicite:1]{index=1}
Core rules:
- Append-only: never rewrite or reorder existing entries.
- Log after completion: do not pre-log planned work.
- Factual, minimal, and evidence-backed: include the smallest evidence that proves the claim.
- Prefer clarity over volume: log the outcome and verification, not a diary.
Entry header format (required):
`### YYYY-MM-DD HH:MM UTC - Short description`
Entry body (required fields):
- Scope: what area changed (backend/frontend/tests/docker/docs/etc.)
- Change: what objectively changed (no hype, no “should”)
- Evidence: commands run and outcomes, or precise static verification if runtime is not possible
- Impact: what this enables/changes for users or contributors (1–2 sentences)
Correction entries:
- If correcting a prior entry, append a new entry with:
  - what the prior entry claimed
  - what is actually true
  - evidence supporting the correction
What to log:
- Behavior changes (runtime or API semantics)
- Validation semantics changes (test runner behavior, skip rules, gating rules)
- Inventory promotions/demotions
- Docker/compose changes that affect how the system runs
- Authoritative documentation changes that alter “repo truth” (AGENTS.md, SYSTEM_INVENTORY.md, Project.md, CHANGE_ROADMAP.md)
What not to log:
- Pure formatting changes with no meaning change (unless required for clarity of authority)
- Temporary experiments that are reverted before completion
- Unverified claims, “it seems,” or “should work” statements

## 7 Validation and Testing Standards
Dependency boundaries:
- Unit tests must not require external dependencies (no network services, no local model services, no external binaries).
- Integration tests must SKIP cleanly when dependencies are absent.
- Never claim “pass” based on mocks when the real dependency is missing.
Verification behavior:
- Run the smallest relevant check first (single test file, targeted command).
- Escalate only when needed or when the change is cross-cutting.
- Report exactly what was run and the observed outcomes.

## 8 Authoritative Commands (Repo Root)
Backend:
- Authoritative backend validation:
  - `./backend/.venv/Scripts/python scripts/validate_backend.py`
- Unit tests:
  - `./backend/.venv/Scripts/python -m pytest tests/`
- Start backend (local):
  - `./backend/.venv/Scripts/python backend/main.py`
Frontend:
- Start frontend (local):
  - `pushd frontend && npm run dev`
Notes:
- Do not assume `make` is installed on Windows. If a Makefile exists, treat it as optional convenience, not the authoritative interface.
- When verifying servers, start, confirm readiness via logs or a simple request, then stop unless instructed to keep it running.

## 9 Data and Artifact Conventions
Runtime artifacts must not live in the repo root.
Preferred location is `./data/` for persistent runtime artifacts such as:
- SQLite DB
- vector store files
- checkpoints
- caches and similar generated state
Rules:
- Do not introduce new storage locations without explicit agreement.
- Ensure ignores cover generated artifacts (git + docker contexts) without being overly broad.

## 10 Docker Conventions (Dev vs Hardened)
This repo uses two compose tracks:
- `docker-compose.dev.yml` (development runner)
  - Fast iteration, relaxed posture
  - Intended to be left running for ongoing development visibility
- `docker-compose.yml` (hardened runner)
  - Security posture and operational constraints
Parity rule:
- Dev and hardened should match core capabilities and wiring; posture differences only.
- If they drift, propose the minimal changes needed to restore parity.
Preferences:
- SQLite-first for default persistence unless the repo explicitly documents otherwise.
- Redis (if present) must degrade gracefully when unavailable and must not block startup.

## 11 Documentation Discipline
Documentation must not exceed SYSTEM_INVENTORY.md claims.
Do not update docs “because it should be true.”
Update docs only when behavior is evidenced and the document is an authoritative consumer of that claim.

## 12 Output Format
When reporting back:
- Summary (1–3 sentences)
- Files inspected and/or touched
- Commands executed and outcomes
- Evidence excerpt(s) (minimal, copy/pasteable)
- Proposal for next step (scoped), or stop if complete
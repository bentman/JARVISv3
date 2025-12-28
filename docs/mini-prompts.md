# Mini-Phase Prompting Technique

## What This Is

This project uses a **mini-phase prompting technique** to control agent behavior and prevent scope creep, drift, or accidental authority escalation. A mini-phase is a narrowly scoped, single-purpose instruction that defines *exactly* what an agent is allowed to do during a specific moment in the workflow.

Mini-phases are not general prompts, plans, or conversations. They are **execution boundaries**.

---

## Key Principle

Mini-phases are not about speed or convenience.
They exist to ensure that **every change has a clear boundary, a clear owner, and a clear record**.

If a mini-phase feels restrictive, it is working as intended.
---

## Why This Exists

As projects grow, agents (and humans) naturally begin to:

* combine execution with explanation,
* infer authority from prior context,
* expand scope beyond what was requested,
* retain state in memory instead of committing it to the repo.

Mini-phases exist to counteract those tendencies by:

* making authority explicit,
* limiting scope to one outcome,
* enforcing discipline around promotions and documentation,
* ensuring changes are intentional, reviewable, and auditable.

This technique keeps the system truthful, boring, and stable.

---

## Core Rules of a Mini-Phase

Every mini-phase prompt must follow these constraints:

* **Action-only**: The agent may only perform the actions explicitly stated.
* **Single purpose**: One goal per mini-phase.
* **No implied authority**: Anything not stated is forbidden.
* **No analysis unless requested**: Execution and reasoning are separate phases.
* **Explicit stop condition**: The agent must stop when the task is complete.

Mini-phases are designed to be composable but never overlapping.

---

## Required Format

Mini-phase prompts must use:

* A **clear title** that describes the intent.
* **Paragraph-form instructions**, not bullet lists.
* Explicit statements about what must *not* be done when needed.
* Language that integrates with existing rules and documents, not appends to them.

This format reduces misinterpretation and prevents agents from “helpfully” doing more than asked.

---

## Example: Promoting a Capability in SYSTEM_INVENTORY.md

**Mini-Phase: Promote one capability to State 1**

Select exactly one capability currently listed in State 2 of `SYSTEM_INVENTORY.md`. Exercise that capability through an automated test integrated into the existing backend test suite and executed via `scripts/validate_backend.py`. Do not refactor, generalize, or expand scope beyond what is required for the test.

Once the automated test passes and produces repeatable results, update `SYSTEM_INVENTORY.md` to move only that capability into State 1. Record the promotion in `CHANGE_LOG.md` in the same change set. Do not modify any other files. Stop immediately after the inventory and change log are updated.

## Relevant Goal Concepts (Conceptual, not detailed)

* **Workflow Architecture vs. Static Configuration**
  Shift from static, declarative configuration to executable workflow graphs with retries, checkpoints, and state.

* **Code-Driven Context**
  Context treated as typed, versioned data with lifecycle, validation, and lineage—not prompt strings.

* **Layered Context Model**
  Distinct scopes (system, workflow, node, tool) with explicit lifetimes and responsibilities.

* **Validation as a First-Class Concern**
  Schema validation, semantic checks, budget enforcement, and security gates at every mutation point.

* **Human-in-the-Loop Checkpoints**
  Explicit approval nodes integrated into workflows, not out-of-band decisions.

* **Hardware-Aware Execution**
  Scheduling and routing decisions based on live hardware state and constraints.

* **Observability & Replayability**
  Checkpointing, replay, and audit trails as core system capabilities.

* **Incremental Maturity Model**
  Foundation → context generation → advanced workflows → hardening, with success criteria at each stage.

* **Testability & Evolvability**
  Workflows and context objects designed to be unit-testable and safely extensible.


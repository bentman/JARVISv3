# JARVISv3: Advanced Agentic Framework

JARVISv3 is a sophisticated AI assistant framework that transforms linear prompt-response loops into structured, state-managed **Agentic Graphs**. By implementing the **"Unified Golden Stack"** architecture, it treats Workflow Orchestration and Code-Driven Context as first-class concerns, enabling reliable, multi-agent collaboration on local hardware.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)
[![Status](https://img.shields.io/badge/status-Advanced%20Prototype-success)](#)

---

## 🚀 Key Features

*   **Workflow Engine**: Execute tasks as Directed Acyclic Graphs (DAGs) with explicit nodes for routing, context building, and validation.
*   **Code-Driven Context**: Context as typed, validated Pydantic objects, preventing "context drift" in complex conversations.
*   **Local-First Intelligence**: Native integration with **Ollama** and **llama.cpp** for privacy-preserving, high-performance local inference.
*   **Hardware-Aware Routing**: Automatic profiling of CPU, GPU, and NPU to select the optimal model for every task.
*   **Multi-Agent Collaboration**: Specialized agent registries for complex workflows (e.g., Software Architect -> Senior Coder -> Security Audit).
*   **Model Context Protocol (MCP)**: Standardized tool integration for file access, web search, and secure code execution.
*   **Unified Search Aggregator**: Privacy-aware search across local memory and multiple web providers (DuckDuckGo, Bing, Google, Tavily) with Redis caching.
*   **Semantic Memory & Tagging**: FAISS-powered long-term memory with conversation tagging and data portability (Export/Import).
*   **Headless Voice Loop**: Standalone Python client for always-on, low-latency voice interaction with barge-in support and local fallbacks.

---

## 🏗️ Engineering Standards (Moving Forward)

To ensure the long-term sustainability and reliability of the JARVISv3 ecosystem, all ongoing development strictly adheres to the following industry standards:

-   **Code Quality (PEP 8 & Type Safety)**: We mandate strict PEP 8 compliance and comprehensive type hinting. Every new implementation is verified via static analysis (`mypy`, `flake8`) to maintain a clean, maintainable codebase.
-   **Architectural Rigor (SOLID)**: Every component is designed following SOLID principles. We prioritize Single Responsibility and Dependency Injection to ensure the framework remains modular and easily testable.
-   **Security by Design (OWASP)**: Our security architecture is informed by OWASP guidelines. Moving forward, we enforce strict schema-based input validation and leverage industry-standard cryptographic libraries for all sensitive operations.
-   **Quality Framework (ISO 25010)**: We evaluate every phase of development against ISO 25010 attributes, ensuring a consistent focus on *Functional Suitability*, *Reliability*, and *Maintainability*.

---

## 🏁 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### Fast Track
1.  **Clone & Configure**:
    ```bash
    git clone https://github.com/bentman/JARVISv3.git
    cd JARVISv3
    cp .env.example .env
    ```
2.  **Start Services**:
    ```bash
    docker-compose up --build
    ```
    *Access the UI at `http://localhost:3000` and API docs at `http://localhost:8000/api/docs`.*

### 🎙️ Headless Voice Mode
For always-on voice interaction on your host machine or specialized hardware (Raspberry Pi):
1.  **Install Dependencies**:
    ```bash
    pip install sounddevice numpy simpleaudio requests
    ```
2.  **Run the Loop**:
    ```bash
    python scripts/voice_loop.py --host http://localhost:8000
    ```

---

## 📐 Architecture Overview

JARVISv3 implements a layered architecture designed for modularity and observability:

1.  **Artifact Layer**: Typed schemas for system and workflow context.
2.  **Context Engine**: Just-in-time assembly and lifecycle management (summarization/archival).
3.  **Workflow Graph**: Declarative orchestration of LLM and Tool nodes.
4.  **Execution & Routing**: Hardware-aware provider selection (Local vs. Cloud).
5.  **Governance & Observability**: Validation gates and end-to-end tracing.

---

## 🤝 Contributions Welcome!

We are building the future of local agentic AI, and your help is appreciated! Whether it's adding new workflow templates, improving hardware detection, or refining the UI, we welcome contributions from the community.

### How to Contribute
- **Report Issues**: Use GitHub Issues for bugs or feature requests.
- **Submit PRs**: Follow our **[Agent Guidelines](agents.md)** for engineering standards and operational protocols.
- **Join the Discussion**: Check out our Discussions page to brainstorm new ideas.

---

## 🛡️ Security & Privacy

*   **Local-First**: All data processing is local by default. Cloud escalation is strictly opt-in and budget-gated.
*   **Privacy Controls**: Automatic PII detection and redaction are integrated into the core validation pipeline.
*   **Compliance**: Built with GDPR/CCPA alignment in mind.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🌟 Acknowledgments

JARVISv3 represents the evolution of **[JARVISv2](https://github.com/bentman/JARVISv2)**, refactoring its production-proven service logic into a modern, workflow-oriented framework.

# JARVISv3: Personal AI Assistant

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-Windows_|_macOS_|_Linux-lightgrey)](#)
[![Status](https://img.shields.io/badge/status-Active_Development-green)](#)

JARVISv3 is a local-first AI assistant designed for daily workflows on your machines. It's built on a **Workflow Graph** architecture, which means instead of just chatting, it coordinates specialized agents to handle things like search, deep research, code creation, code review, and memory retrieval — all while running on your own hardware.

## 🔍 Reality Check: What works?

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Basic Chat** | ✅ Works | Local inference via Ollama/llama.cpp is stable. |
| **Voice Interaction**| ✅ Works | Wake word and STT/TTS are functional and reliable. |
| **Memory** | ✅ Works | Semantic search across past conversations (FAISS). |
| **Web Research** | ✅ Works | Aggregated search with privacy redaction. |
| **Multi-Machine** | ⚠️ Needs Work | Runs on Win/Mac/Linux, but cross-device sync is manual. |
| **Daily Utility** | ⚠️ Needs Work | Needs better system integration (tray, global hotkeys). |

---

## 🚀 Quick Start (5 Minutes)

### 🖥️ Desktop (with NVIDIA GPU)
Use Docker for the easiest setup with GPU acceleration:
1.  **Configure**: `cp .env.example .env` (Add API keys for web search if needed).
2.  **Launch**: `docker-compose up --build`
3.  **Access**: UI at `http://localhost:3000`, API at `http://localhost:8000`.

### 💻 Laptop (Mac M-Series or NPU)
Run natively for best performance and NPU access:
1.  **Backend**: 
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python main.py
    ```
2.  **Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

---

## 🛠️ Daily Workflows

*   **Search**: Grab results from the web without the tracking.
*   **Deep Research**: Dig deep into a topic and get a clean summary.
*   **Code Creation**: Spin up logic locally—your code stays on your drive.
*   **Code Review**: Catch bugs and style issues before you commit.
*   **Memory Retrieval**: "What was that idea I had last Tuesday?"

---

## ✅ System Validation
Run the system validation suite periodically to check core functionality:
```bash
python scripts/validate_production.py
```
*Checks Backend Core logic, Feature Parity, and Frontend build.*

---

## 🛡️ Privacy First
- **Zero Cloud by Default**: Your conversations stay on your disk.
- **Redaction**: PII is automatically scrubbed before any optional cloud escalation.
- **Audit Logs**: See exactly what data was processed and where.

---

## 🤝 Contributions Welcome!
Whether it's adding new workflow templates, improving hardware detection, or refining the UI, contributions are welcome. See our **[Agent Guidelines](AGENTS.md)** for standards.

---

## 📜 License
Distributed under the **MIT License**. See **[LICENSE](LICENSE)** for more information.

---

## 🌟 Acknowledgments
Refactored from the proven logic of **[JARVISv2](https://github.com/bentman/JARVISv2)**.

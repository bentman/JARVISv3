# JARVISv3: Personal AI Assistant

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

### 1. Offline Research
Ask "Jarvis, summarize my notes on project X" or use `/research [topic]` to pull from both local history and the web.

### 2. Code Review
Use the specialized coding agents to audit local files for security issues or refactoring opportunities without sending your code to the cloud.

### 3. Voice-First Interaction
Run `python scripts/voice_loop.py` on your laptop to have an always-on assistant while you work, supporting barge-in and local wake word detection.

---

## ✅ System Validation
We verify the entire system logic daily. To run the full suite and generate a report:
```bash
python scripts/validate_production.py
```
*Checks Backend Core, Feature Parity, Frontend UI, and AI Inference.*

---

## 🛡️ Privacy First
- **Zero Cloud by Default**: Your conversations stay on your disk.
- **Redaction**: PII is automatically scrubbed before any optional cloud escalation.
- **Audit Logs**: See exactly what data was processed and where.

---

## 🌟 Acknowledgments
Refactored from the proven logic of **[JARVISv2](https://github.com/bentman/JARVISv2)**.

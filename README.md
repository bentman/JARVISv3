# JARVISv3: Personal AI Assistant

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-Windows_|_macOS_|_Linux-lightgrey)](#)
[![Status](https://img.shields.io/badge/status-Active_Development-yellow)](#)

JARVISv3 is a local-first AI assistant designed for daily workflows on your machines. It's built on a **Workflow Graph** architecture, which means instead of just chatting, it coordinates specialized agents to handle things like search, deep research, code creation, code review, and memory retrieval — all while running on your own hardware.

## 🔍 Reality Check: What works?

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Basic Chat** | ⚠️ Implemented | Requires external LLM provider (Ollama/llama.cpp). |
| **Voice Interaction**| ⚠️ Implemented | Wake word and STT/TTS are implemented but require external voice models. |
| **Memory** | ✅ Exercised | Semantic search across past conversations (FAISS) - locally functional. |
| **Web Research** | ⚠️ Implemented | Aggregated search with privacy redaction. Requires external search APIs. |
| **Multi-Machine** | ⚠️ Implemented | Runs on Win/Mac/Linux with distributed node support. |
| **Daily Utility** | ⚠️ Implemented | System integration available but requires configuration. |

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
    ./.venv/Scripts/python main.py
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

## 📖 User Guide

### Getting Started

After following the deployment guide for your platform, JARVISv3 will be accessible at:

- **Web Interface**: `http://localhost:3000`
- **Desktop App**: Via the Tauri wrapper (run `npm run tauri dev` in `frontend/`)
- **API Documentation**: `http://localhost:8000/api/docs`

### First-Time Configuration

1. **Hardware Detection**: JARVISv3 automatically detects your hardware capabilities and selects the optimal model
2. **Privacy Settings**: Configure your privacy preferences in Settings → Privacy
3. **Voice Setup**: Configure wake word sensitivity and voice preferences in Settings → Voice

### Understanding the Interface

The JARVISv3 interface consists of:

- **Conversation Area**: Main chat interface with streaming responses
- **Hardware Indicators**: Real-time system status and model selection
- **Voice Controls**: Wake word activation and feedback indicators
- **Settings Panel**: Comprehensive customization options

### Voice Mode

#### Wake Word Activation

JARVISv3 supports hands-free activation using wake word detection:

1. **Default Wake Word**: "Jarvis" (configurable in Settings)
2. **Activation**: Say the wake word followed by your command
3. **Feedback**: Visual and audio feedback confirms activation

#### Voice Commands

##### Basic Commands
```
"Jarvis, what's the weather?"
"Jarvis, set a reminder for tomorrow"
"Jarvis, search for information about AI"
```

##### Advanced Commands
```
"Jarvis, start a research task about renewable energy"
"Jarvis, create a workflow for code review"
"Jarvis, search my memory for project notes"
```

### Text Mode

#### Chat Interface

The text interface provides:

- **Rich Text Display**: Markdown support with code syntax highlighting
- **Streaming Responses**: Real-time response generation with progress indicators
- **Keyboard Shortcuts**: Quick actions and navigation
- **Message History**: Persistent conversation history

#### Text Commands

##### Basic Chat
```
Hello, how are you?
What can you help me with?
Tell me about yourself.
```

##### Task Commands
```
/summarize [text or document]
/search [query]
/research [topic]
/code [programming task]
```

##### Workflow Commands
```
/start workflow [workflow_name]
/list workflows
/status workflow [workflow_id]
```

### Hybrid Mode

JARVISv3 supports seamless switching between voice and text modes:

1. **Context Preservation**: Your conversation context is maintained across mode switches
2. **Adaptive Interface**: The interface adapts to your current mode
3. **Unified History**: All interactions are stored in a unified conversation history

### Memory and Search

#### Semantic Memory

JARVISv3 uses FAISS vector store for semantic memory:

- **Automatic Indexing**: Conversations are automatically indexed for search
- **Semantic Search**: Find related conversations based on meaning, not just keywords
- **Memory Persistence**: Conversations are stored across sessions

#### Search Capabilities

##### Local Memory Search
```
/search memory for project requirements
/find conversations about machine learning
/lookup previous discussion about budget
```

##### Unified Search
JARVISv3 can search both local memory and the web.

### Privacy and Security

#### Privacy Levels

JARVISv3 offers three privacy levels:

##### Low Privacy
- **Local Processing**: All processing happens locally
- **No Cloud**: No external API calls
- **Maximum Privacy**: Complete data isolation

##### Medium Privacy (Default)
- **Local First**: Primary processing is local
- **Selective Cloud**: Cloud features only when explicitly enabled
- **Budget Control**: Automatic budget management for cloud usage

##### High Privacy
- **Strict Local**: Only essential local processing
- **Minimal Data**: Aggressive data minimization
- **Enhanced Security**: Additional security measures

### Hardware and Performance

#### Hardware Detection

JARVISv3 automatically detects your hardware and optimizes accordingly.

#### Hardware Profiles
- **Light**: CPUs with 4-8 cores, 8-16GB RAM
- **Medium**: CPUs with 8+ cores, 16-32GB RAM, entry-level GPUs
- **Heavy**: High-end CPUs, 32GB+ RAM, powerful GPUs
- **NPU-Optimized**: Systems with Neural Processing Units

### Advanced Features

#### Workflow Management

JARVISv3 includes several built-in workflows:

- **Chat**: General conversation and Q&A
- **Research**: Complex research tasks with web search
- **Code Review**: Code analysis and review workflows
- **Task Management**: Task creation and tracking

### Troubleshooting

#### Common Issues

##### JARVISv3 Won't Start
1. **Check Dependencies**: Ensure all dependencies are installed
2. **Port Conflicts**: Check for port conflicts (8000, 3000)
3. **Permissions**: Verify file and directory permissions
4. **Logs**: Check application logs for error details

##### Voice Not Working
1. **Microphone**: Check microphone connection and permissions
2. **Wake Word**: Verify wake word detection is enabled
3. **Audio Settings**: Check audio input/output settings
4. **Background Noise**: Reduce background noise

##### Slow Performance
1. **Hardware**: Verify hardware meets requirements
2. **Memory**: Check available memory
3. **Model Size**: Use smaller models for better performance
4. **Background Apps**: Close other resource-intensive applications

### Best Practices

#### Security Best Practices
1. **Strong Authentication**: Use strong passwords and 2FA
2. **Regular Updates**: Keep JARVISv3 updated with latest security patches
3. **Access Control**: Limit access to authorized users only
4. **Audit Logs**: Regularly review audit logs for suspicious activity
5. **Data Minimization**: Only store necessary data

#### Performance Best Practices
1. **Hardware Optimization**: Ensure hardware meets or exceeds requirements
2. **Regular Maintenance**: Clean up old data and optimize regularly
3. **Resource Monitoring**: Monitor resource usage and adjust settings
4. **Model Selection**: Choose appropriate models for your use case
5. **Caching**: Use caching for frequently accessed data

---

## ✅ System Validation
Run the authoritative backend validation suite to check core functionality:
```bash
./backend/.venv/Scripts/python scripts/validate_backend.py
```
This tool automatically discovers and runs all tests across Unit, Integration, and Agentic categories, providing per-test visibility with status indicators. Generates terminal summary plus timestamped reports in `reports/` for detailed results.

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

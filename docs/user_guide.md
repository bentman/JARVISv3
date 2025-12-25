# JARVISv3 User Guide

Welcome to JARVISv3, your advanced AI assistant with sophisticated workflow architecture and code-driven context management.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Voice Mode](#voice-mode)
3. [Text Mode](#text-mode)
4. [Hybrid Mode](#hybrid-mode)
5. [Memory and Search](#memory-and-search)
6. [Privacy and Security](#privacy-and-security)
7. [Hardware and Performance](#hardware-and-performance)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Getting Started

### Initial Setup

After following the [deployment guide](./deployment/) for your platform, JARVISv3 will be accessible at:

- **Web Interface**: `http://localhost:3000`
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

## Voice Mode

### Wake Word Activation

JARVISv3 supports hands-free activation using wake word detection:

1. **Default Wake Word**: "Jarvis" (configurable in Settings)
2. **Activation**: Say the wake word followed by your command
3. **Feedback**: Visual and audio feedback confirms activation

### Voice Commands

#### Basic Commands
```
"Jarvis, what's the weather?"
"Jarvis, set a reminder for tomorrow"
"Jarvis, search for information about AI"
```

#### Advanced Commands
```
"Jarvis, start a research task about renewable energy"
"Jarvis, create a workflow for code review"
"Jarvis, search my memory for project notes"
```

### Voice Settings

Access Settings → Voice to configure:

- **Wake Word Sensitivity**: Adjust detection sensitivity
- **Voice Selection**: Choose from available TTS voices
- **Audio Quality**: Configure input/output quality
- **Barge-in**: Enable/disable interruption capability

### Voice Best Practices

1. **Clear Speech**: Speak clearly and at a moderate pace
2. **Quiet Environment**: Minimize background noise for better accuracy
3. **Microphone Quality**: Use a good quality microphone for optimal results
4. **Wake Word Training**: Allow JARVISv3 to learn your voice for better detection

## Text Mode

### Chat Interface

The text interface provides:

- **Rich Text Display**: Markdown support with code syntax highlighting
- **Streaming Responses**: Real-time response generation with progress indicators
- **Keyboard Shortcuts**: Quick actions and navigation
- **Message History**: Persistent conversation history

### Text Commands

#### Basic Chat
```
Hello, how are you?
What can you help me with?
Tell me about yourself.
```

#### Task Commands
```
/summarize [text or document]
/search [query]
/research [topic]
/code [programming task]
```

#### Workflow Commands
```
/start workflow [workflow_name]
/list workflows
/status workflow [workflow_id]
```

### Keyboard Shortcuts

- **Ctrl/Cmd + Enter**: Send message
- **Ctrl/Cmd + K**: Clear conversation
- **Ctrl/Cmd + S**: Save conversation
- **Ctrl/Cmd + H**: Toggle voice mode
- **Ctrl/Cmd + /**: Show help

## Hybrid Mode

### Seamless Mode Switching

JARVISv3 supports seamless switching between voice and text modes:

1. **Context Preservation**: Your conversation context is maintained across mode switches
2. **Adaptive Interface**: The interface adapts to your current mode
3. **Unified History**: All interactions are stored in a unified conversation history

### When to Use Each Mode

#### Voice Mode
- **Hands-free situations**: Cooking, driving, multitasking
- **Natural conversation**: Casual queries and commands
- **Accessibility**: For users with mobility or vision challenges

#### Text Mode
- **Complex tasks**: Detailed instructions or multi-step processes
- **Quiet environments**: Libraries, meetings, shared spaces
- **Precision**: When exact wording is important

#### Hybrid Mode
- **Complex workflows**: Start with voice, refine with text
- **Research tasks**: Voice for initial queries, text for detailed follow-up
- **Creative work**: Voice for brainstorming, text for refinement

## Memory and Search

### Semantic Memory

JARVISv3 uses FAISS vector store for semantic memory:

- **Automatic Indexing**: Conversations are automatically indexed for search
- **Semantic Search**: Find related conversations based on meaning, not just keywords
- **Memory Persistence**: Conversations are stored across sessions

### Search Capabilities

#### Local Memory Search
```
/search memory for project requirements
/find conversations about machine learning
/lookup previous discussion about budget
```

#### Unified Search
JARVISv3 can search both local memory and the web:

```
/search web and memory for latest AI trends
/unified search for Python best practices
```

### Memory Management

#### Automatic Cleanup
- **Summarization**: Long conversations are automatically summarized
- **Pruning**: Old, irrelevant memories are removed
- **Archival**: Important memories are archived for long-term storage

#### Manual Management
- **Delete Conversations**: Remove specific conversations from memory
- **Tag Conversations**: Add tags for better organization
- **Export Memory**: Export conversation history for backup

## Privacy and Security

### Privacy Levels

JARVISv3 offers three privacy levels:

#### Low Privacy
- **Local Processing**: All processing happens locally
- **No Cloud**: No external API calls
- **Maximum Privacy**: Complete data isolation

#### Medium Privacy (Default)
- **Local First**: Primary processing is local
- **Selective Cloud**: Cloud features only when explicitly enabled
- **Budget Control**: Automatic budget management for cloud usage

#### High Privacy
- **Strict Local**: Only essential local processing
- **Minimal Data**: Aggressive data minimization
- **Enhanced Security**: Additional security measures

### Data Protection

#### Encryption
- **At Rest**: All stored data is encrypted
- **In Transit**: All network communication is encrypted
- **In Memory**: Sensitive data is encrypted in memory when possible

#### PII Detection
JARVISv3 automatically detects and protects personally identifiable information:

- **Automatic Redaction**: PII is automatically redacted from logs
- **User Notification**: You'll be notified when PII is detected
- **Compliance**: GDPR/CCPA compliant data handling

### Security Features

#### Input Validation
- **Security Scanning**: All inputs are scanned for security issues
- **Injection Prevention**: SQL injection and XSS prevention
- **Malware Detection**: File uploads are scanned for malware

#### Access Control
- **Authentication**: Secure user authentication
- **Authorization**: Role-based access control
- **Audit Logs**: Complete audit trail of all actions

## Hardware and Performance

### Hardware Detection

JARVISv3 automatically detects your hardware and optimizes accordingly:

#### Hardware Profiles
- **Light**: CPUs with 4-8 cores, 8-16GB RAM
- **Medium**: CPUs with 8+ cores, 16-32GB RAM, entry-level GPUs
- **Heavy**: High-end CPUs, 32GB+ RAM, powerful GPUs
- **NPU-Optimized**: Systems with Neural Processing Units

#### Automatic Optimization
- **Model Selection**: Automatically selects the best model for your hardware
- **Resource Management**: Optimizes memory and CPU usage
- **Performance Monitoring**: Real-time performance tracking

### Performance Tuning

#### Manual Configuration
Access Settings → Performance to configure:

- **Model Selection**: Manually select models
- **Memory Limits**: Set memory usage limits
- **Processing Priority**: Adjust processing priority
- **Cache Settings**: Configure caching behavior

#### Performance Monitoring
- **Real-time Metrics**: CPU, memory, and GPU usage
- **Response Times**: Track response time performance
- **Resource Usage**: Monitor resource consumption
- **Health Status**: Overall system health

### Troubleshooting Performance Issues

#### Slow Response Times
1. **Check Hardware**: Verify your hardware meets requirements
2. **Close Applications**: Close other resource-intensive applications
3. **Adjust Settings**: Lower model complexity or memory usage
4. **Clear Cache**: Clear cached data that may be consuming memory

#### High Memory Usage
1. **Enable Pruning**: Enable automatic memory pruning
2. **Reduce Context**: Limit conversation context size
3. **Clear History**: Clear old conversation history
4. **Upgrade Hardware**: Consider hardware upgrades if needed

## Advanced Features

### Workflow Management

#### Built-in Workflows
JARVISv3 includes several built-in workflows:

- **Chat**: General conversation and Q&A
- **Research**: Complex research tasks with web search
- **Code Review**: Code analysis and review workflows
- **Task Management**: Task creation and tracking

#### Custom Workflows
Create custom workflows using the workflow system:

1. **Define Workflow**: Use YAML or Python to define workflow steps
2. **Configure Nodes**: Set up routing, context building, and validation
3. **Test Workflow**: Test and refine your workflow
4. **Deploy Workflow**: Make your workflow available for use

#### Workflow Examples

##### Research Workflow
```yaml
workflow_id: "research_workflow"
nodes:
  - id: "router"
    type: "router"
    description: "Identify research intent"
  - id: "search_web"
    type: "search_web"
    description: "Search the web for information"
  - id: "synthesize"
    type: "llm_worker"
    description: "Synthesize findings"
  - id: "validate"
    type: "validator"
    description: "Validate research output"
```

### MCP Integration

#### Tool Access
JARVISv3 integrates with Model Context Protocol (MCP) tools:

- **File System Access**: Read and write files
- **Code Execution**: Execute code in secure environments
- **Web Search**: Access web search capabilities
- **System Information**: Access system information

#### Custom Tools
Develop and integrate custom MCP tools:

1. **Tool Definition**: Define tool interface and capabilities
2. **Implementation**: Implement tool functionality
3. **Registration**: Register tool with JARVISv3
4. **Usage**: Use tool in workflows and conversations

### Voice Integration

#### Advanced Voice Features
- **Emotion Detection**: Detect emotion in your voice
- **Multi-language Support**: Support for multiple languages
- **Voice Commands**: Custom voice commands for specific tasks
- **Audio Quality**: High-quality audio processing

#### Voice Customization
- **Wake Word**: Customize the wake word
- **Voice Selection**: Choose from multiple TTS voices
- **Audio Settings**: Configure input and output audio quality
- **Barge-in Control**: Configure interruption behavior

## Troubleshooting

### Common Issues

#### JARVISv3 Won't Start
1. **Check Dependencies**: Ensure all dependencies are installed
2. **Port Conflicts**: Check for port conflicts (8000, 3000)
3. **Permissions**: Verify file and directory permissions
4. **Logs**: Check application logs for error details

#### Voice Not Working
1. **Microphone**: Check microphone connection and permissions
2. **Wake Word**: Verify wake word detection is enabled
3. **Audio Settings**: Check audio input/output settings
4. **Background Noise**: Reduce background noise

#### Slow Performance
1. **Hardware**: Verify hardware meets requirements
2. **Memory**: Check available memory
3. **Model Size**: Use smaller models for better performance
4. **Background Apps**: Close other resource-intensive applications

#### Privacy Concerns
1. **Privacy Level**: Set appropriate privacy level
2. **Data Deletion**: Use data deletion features
3. **Local Processing**: Enable local-first processing
4. **Audit Logs**: Review audit logs for data access

### Getting Help

#### Documentation
- **User Guide**: This document for user information
- **API Documentation**: `/api/docs` for API reference
- **Workflow Documentation**: `/docs/workflows.md` for workflow details

#### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Community discussions and help
- **Documentation**: Comprehensive documentation

## Best Practices

### Security Best Practices

1. **Strong Authentication**: Use strong passwords and 2FA
2. **Regular Updates**: Keep JARVISv3 updated with latest security patches
3. **Access Control**: Limit access to authorized users only
4. **Audit Logs**: Regularly review audit logs for suspicious activity
5. **Data Minimization**: Only store necessary data

### Performance Best Practices

1. **Hardware Optimization**: Ensure hardware meets or exceeds requirements
2. **Regular Maintenance**: Clean up old data and optimize regularly
3. **Resource Monitoring**: Monitor resource usage and adjust settings
4. **Model Selection**: Choose appropriate models for your use case
5. **Caching**: Use caching for frequently accessed data

### Privacy Best Practices

1. **Privacy Settings**: Configure privacy settings according to your needs
2. **Data Review**: Regularly review stored data and delete unnecessary information
3. **Local Processing**: Prefer local processing when possible
4. **PII Protection**: Be careful with personally identifiable information
5. **Compliance**: Ensure compliance with relevant regulations (GDPR, CCPA)

### Workflow Best Practices

1. **Clear Objectives**: Define clear objectives for each workflow
2. **Error Handling**: Implement proper error handling and recovery
3. **Testing**: Thoroughly test workflows before deployment
4. **Documentation**: Document workflows for maintainability
5. **Monitoring**: Monitor workflow performance and usage

### Voice Best Practices

1. **Clear Speech**: Speak clearly and at a moderate pace
2. **Environment**: Use in quiet environments when possible
3. **Microphone Quality**: Use quality microphones for better accuracy
4. **Wake Word Training**: Allow time for wake word training
5. **Privacy**: Be mindful of privacy when using voice features

## Conclusion

JARVISv3 provides a powerful AI assistant framework with advanced features for both personal and professional use. By following this guide and the best practices outlined, you can maximize the benefits of JARVISv3 while maintaining security, privacy, and optimal performance.

For additional support and information:
- Visit the [GitHub repository](https://github.com/bentman/JARVISv3)
- Check the [API documentation](http://localhost:8000/api/docs)
- Join the [community discussions](https://github.com/bentman/JARVISv3/discussions)

Remember to regularly update JARVISv3 to benefit from the latest features, improvements, and security updates.

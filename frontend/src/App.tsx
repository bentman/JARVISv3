import React, { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Toaster, toast } from 'react-hot-toast';
import { MessageCircle, Send, Settings, User, Bot, Plus, Menu, X, Volume2 } from 'lucide-react';
import HardwareIndicator from './components/HardwareIndicator';
import BudgetSummary from './components/BudgetSummary';
import VoiceRecorder from './components/VoiceRecorder';
import SettingsModal from './components/SettingsModal';
import WorkflowVisualizer from './components/WorkflowVisualizer';
import { voiceService } from './services/api';

// Define TypeScript interfaces
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  tokensUsed?: number;
  workflowId?: string;
}

interface WorkflowStatus {
  workflow_id: string;
  status: string;
  progress: number;
  completed_nodes: string[];
  failed_nodes: string[];
  execution_time: number;
}

interface SystemHealth {
  status: string;
  timestamp: string;
  version: string;
  modules: Record<string, string>;
}

// API service functions
const apiService = {
  // Chat API
  chat: (user_id: string, query: string) => 
    axios.post('/api/v1/chat', { user_id, query }),
  
  // Health check
  health: () => axios.get('/health'),
  
  // Workflow status
  getWorkflowStatus: (workflow_id: string) =>
    axios.get<WorkflowStatus>(`/api/v1/workflow/${workflow_id}/status`),
  
  // Context building
  buildContext: (user_id: string, query: string) =>
    axios.post('/api/v1/context/build', { user_id, query }),
};


function App() {
  const [userInput, setUserInput] = useState('');
  const [userId, setUserId] = useState('user_123');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Hello! I'm JARVISv3, an advanced agentic system with workflow architecture and code-driven context. How can I assist you today?",
      timestamp: new Date(),
    }
  ]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [autoPlayAudio, setAutoPlayAudio] = useState(true);
  const [activeAudio, setActiveAudio] = useState<HTMLAudioElement | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [preferredModel, setPreferredModel] = useState('auto');
  const [privacyLevel, setPrivacyLevel] = useState('medium');
  const [searchEnabled, setSearchEnabled] = useState(true);
  const [searchProviders, setSearchProviders] = useState('duckduckgo');
  const [monthlyBudgetLimit, setMonthlyBudgetLimit] = useState(100);
  const [currentNodeId, setCurrentNodeId] = useState<string | null>(null);
  const [completedNodes, setCompletedNodes] = useState<string[]>([]);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Query for health check
  const { data: healthData, isLoading: isHealthLoading } = useQuery<SystemHealth>({
    queryKey: ['health'],
    queryFn: async () => {
      const response: any = await apiService.health();
      return response.data;
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });


  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!userInput.trim() || isProcessing) return;

    const currentInput = userInput;
    // Add user message to chat
    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: currentInput,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsProcessing(true);

    const assistantMessageId = `msg_${Date.now() + 1}`;
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, assistantMessage]);
    setCompletedNodes([]);
    setCurrentNodeId(null);

    try {
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user_id: userId, 
          query: currentInput,
          user_preferences: {
            preferred_model: preferredModel,
            privacy_level: privacyLevel,
            search_enabled: searchEnabled,
            search_providers: searchProviders,
            monthly_budget_limit: monthlyBudgetLimit
          }
        }),
      });

      if (!response.ok) throw new Error('Failed to start stream');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();
      let fullContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.type === 'node_start') {
                setCurrentNodeId(data.node_id);
              } else if (data.type === 'node_end') {
                setCurrentNodeId(null);
                setCompletedNodes(prev => [...prev, data.node_id]);
              } else if (data.type === 'stream_chunk') {
                fullContent += data.chunk;
                setMessages(prev => prev.map(m => 
                  m.id === assistantMessageId ? { ...m, content: fullContent } : m
                ));
              } else if (data.type === 'workflow_completed') {
                  setMessages(prev => prev.map(m => 
                    m.id === assistantMessageId ? { ...m, workflowId: data.workflow_id } : m
                  ));
                  if (autoPlayAudio && data.final_response) handleSpeak(data.final_response);
              } else if (data.type === 'error') {
                  toast.error(data.message);
              }
            } catch (e) {
               // Ignore parse errors for incomplete JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      toast.error('Chat processing failed');
      setMessages(prev => prev.map(m => 
        m.id === assistantMessageId ? { ...m, content: 'Sorry, I encountered an error processing your request.' } : m
      ));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleTranscription = async (audioBlob: Blob) => {
    toast.loading("Transcribing...");
    try {
      const { text } = await voiceService.transcribe(audioBlob as File);
      setUserInput(text);
      toast.dismiss();
    } catch (error) {
      toast.dismiss();
      toast.error("Transcription failed.");
    }
  };

  const handleSpeak = async (text: string) => {
    try {
      // Stop current audio if playing
      if (activeAudio) {
        activeAudio.pause();
      }

      const audioBlob = await voiceService.speak(text);
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      setActiveAudio(audio);
      audio.onended = () => setActiveAudio(null);
      audio.play();
    } catch (error) {
        toast.error("Failed to play audio response.")
    }
  };

  // Barge-in: Stop audio when recording starts
  useEffect(() => {
    if (isRecording && activeAudio) {
      activeAudio.pause();
      setActiveAudio(null);
      toast('Interrupting...', { icon: '🛑', duration: 1000 });
    }
  }, [isRecording, activeAudio]);

  const startNewChat = () => {
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: "Hello! I'm JARVISv3, an advanced agentic system with workflow architecture and code-driven context. How can I assist you today?",
        timestamp: new Date(),
      }
    ]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      {/* Mobile sidebar toggle */}
      <div className="md:hidden fixed top-4 left-4 z-50">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
        >
          {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Sidebar */}
      <div className={`fixed md:relative z-40 w-64 bg-gray-800 h-full transform transition-transform duration-300 ease-in-out ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } md:translate-x-0`}>
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center">
              <MessageCircle size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold">JARVISv3</h1>
              <p className="text-xs text-gray-400">Agentic Graph System</p>
            </div>
          </div>
        </div>
        
        <div className="p-4">
          <button
            onClick={startNewChat}
            className="w-full flex items-center space-x-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <Plus size={18} />
            <span>New Chat</span>
          </button>
          
          <div className="mt-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">System Status</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">API:</span>
                <span className={healthData ? 'text-green-400' : 'text-red-400'}>
                  {isHealthLoading ? 'Checking...' : healthData ? 'Online' : 'Offline'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Version:</span>
                <span className="text-gray-300">{healthData?.version || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Budget:</span>
                <span className="text-gray-300">
                  {healthData?.modules?.budget_state || 'N/A'}
                </span>
              </div>
            </div>
          </div>
          
          <div className="mt-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">User</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">ID:</span>
                <span className="text-gray-300">{userId}</span>
              </div>
              <div>
                <label className="text-gray-400 text-xs">User ID:</label>
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="w-full px-2 py-1 text-xs bg-gray-700 rounded mt-1"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="md:ml-64 flex flex-col h-screen">
        {/* Header */}
        <header className="border-b border-gray-700 bg-gray-800/50 backdrop-blur-sm p-4">
          <div className="flex items-center justify-between">
            <div className="hidden md:block">
              <h1 className="text-xl font-bold">JARVISv3</h1>
              <p className="text-sm text-gray-400">Advanced Agentic System</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm hidden sm:block">
                <span className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  isHealthLoading ? 'bg-yellow-500' : healthData ? 'bg-green-500' : 'bg-red-500'
                }`}></span>
                <span>
                  {isHealthLoading ? 'Checking...' : healthData ? 'Healthy' : 'Offline'}
                </span>
              </div>
              
              <div className="flex flex-col items-end gap-1">
                <HardwareIndicator />
                <BudgetSummary />
              </div>
              
              <button 
                onClick={() => setIsSettingsOpen(true)}
                className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
              >
                <Settings size={20} />
              </button>
            </div>
          </div>
        </header>

        {/* Chat messages */}
        <div className="flex-1 overflow-y-auto p-4 pb-20">
          <div className="max-w-4xl mx-auto space-y-4">
            {(isProcessing || currentNodeId || completedNodes.length > 0) && isProcessing && (
              <div className="mb-6 bg-gray-800/30 p-3 rounded-xl border border-gray-700/50 flex flex-col items-center">
                <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-2 font-bold">Workflow Pipeline</div>
                <WorkflowVisualizer 
                  currentNodeId={currentNodeId} 
                  completedNodes={completedNodes}
                  failedNodes={[]} 
                />
              </div>
            )}
            {messages.map((message) => (
              <div
                key={message.id}
                className={`p-4 rounded-lg max-w-3xl ${
                  message.role === 'user'
                    ? 'bg-blue-900/20 border border-blue-800/50 ml-auto'
                    : 'bg-gray-800/50 border border-gray-700/50'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                      message.role === 'user' ? 'bg-blue-600' : 'bg-purple-600'
                    }`}
                  >
                    {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium mb-1">
                      {message.role === 'user' ? 'You' : 'JARVISv3'}
                    </div>
                    <div className="text-gray-300 whitespace-pre-wrap">
                      {message.content}
                    </div>
                    <div className="text-xs text-gray-500 mt-2 flex items-center space-x-2">
                      <span>{message.timestamp.toLocaleTimeString()}</span>
                      {message.tokensUsed && (
                        <span className="bg-gray-700 px-2 py-1 rounded">
                          {message.tokensUsed} tokens
                        </span>
                      )}
                      {message.workflowId && (
                        <span className="bg-gray-700 px-2 py-1 rounded">
                          {message.workflowId.substring(0, 8)}...
                        </span>
                      )}
                    </div>
                  </div>
                  {message.role === 'assistant' && (
                    <button 
                      onClick={() => handleSpeak(message.content)}
                      className="text-gray-400 hover:text-white transition-colors ml-2"
                    >
                      <Volume2 size={16} />
                    </button>
                  )}
                </div>
              </div>
            ))}
            
            {(isProcessing || isRecording) && (
              <div className="p-4 rounded-lg max-w-3xl bg-gray-800/50 border border-gray-700/50">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold bg-purple-600">
                    <Bot size={16} />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium mb-1">JARVISv3</div>
                    <div className="text-gray-300 flex items-center animate-pulse">
                      {isRecording ? 'Recording...' : 'Processing...'}
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="border-t border-gray-700 bg-gray-800/50 backdrop-blur-sm p-4 sticky bottom-0">
          <div className="max-w-4xl mx-auto">
            <div className="flex space-x-3">
              <div className="flex-1 relative">
                <textarea
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask JARVISv3 anything..."
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={2}
                  disabled={isProcessing}
                />
              </div>
              <VoiceRecorder
                onTranscription={handleTranscription}
                isRecording={isRecording}
                setIsRecording={setIsRecording}
              />
              <button
                onClick={handleSendMessage}
                disabled={isProcessing || !userInput.trim()}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
              >
                <Send size={18} />
                <span className="hidden sm:inline">Send</span>
              </button>
            </div>
            
            <div className="mt-3 text-xs text-gray-500 flex justify-between">
              <div>Workflow Architecture • Code-Driven Context • MCP Integration</div>
              <div>Budget: {healthData?.modules?.budget_state || 'N/A'}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Toast notifications */}
      <Toaster position="bottom-right" />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={{
          userId,
          preferredModel,
          privacyLevel,
          autoPlayAudio,
          searchEnabled,
          searchProviders,
          monthlyBudgetLimit
        }}
        onSave={(newSettings) => {
          setUserId(newSettings.userId);
          setPreferredModel(newSettings.preferredModel);
          setPrivacyLevel(newSettings.privacyLevel);
          setAutoPlayAudio(newSettings.autoPlayAudio);
          setSearchEnabled(newSettings.searchEnabled);
          setSearchProviders(newSettings.searchProviders);
          setMonthlyBudgetLimit(newSettings.monthlyBudgetLimit);
          toast.success('Settings saved');
        }}
      />

      {/* Overlay for mobile sidebar */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
          onClick={() => setSidebarOpen(false)}
        ></div>
      )}
    </div>
  );
}

export default App;

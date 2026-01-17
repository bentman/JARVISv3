import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface ChatResponse {
  response: string;
  workflow_id: string;
  tokens_used: number;
  validation_passed: boolean;
  execution_time: number;
}

export const chatService = {
  sendMessage: async (userId: string, query: string): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/api/v1/chat', {
      user_id: userId,
      query: query,
    });
    return response.data;
  },
};

export interface HardwareStatus {
  gpu_usage: number;
  memory_available_gb: number;
  cpu_usage: number;
  available_tiers: string[];
  current_load: number;
}

export interface BudgetStatus {
  cloud_spend_usd: number;
  monthly_limit_usd: number;
  remaining_pct: number;
  daily_spending: number;
}

export const systemService = {
  getHardwareStatus: async (): Promise<HardwareStatus> => {
    const response = await api.get<HardwareStatus>('/api/v1/hardware/status');
    return response.data;
  },
  getBudgetStatus: async (userId: string = "admin_123"): Promise<BudgetStatus> => {
    const response = await api.get<BudgetStatus>(`/api/v1/budget/status?user_id=${userId}`);
    return response.data;
  },
};

export const voiceService = {
  transcribe: async (file: File): Promise<{ text: string; confidence: number }> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/v1/voice/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  speak: async (text: string): Promise<Blob> => {
    const response = await api.post('/api/v1/voice/speak', { text }, {
      responseType: 'blob',
    });
    return response.data;
  },
};

export default api;

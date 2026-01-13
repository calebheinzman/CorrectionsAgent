import { create } from 'zustand';

const mockChats = [
  {
    id: 'chat-1',
    title: 'Drug activity investigation - Block C',
    lastUpdated: new Date(Date.now() - 1000 * 60 * 30),
    tags: ['Case 12-409', 'Drug use'],
  },
  {
    id: 'chat-2',
    title: 'Incident report analysis - Fight 01/05',
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 2),
    tags: ['Incident-linked'],
  },
  {
    id: 'chat-3',
    title: 'Prisoner transfer records review',
    lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24),
    tags: [],
  },
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

function mapQueryResponseToRun({ question, createdAt, runId, responseJson }) {
  const citations = (responseJson.citations || []).map((c, idx) => ({
    citation_number: idx + 1,
    source_type: c.source_type,
    source_id: c.source_id,
    snippet: c.excerpt || '',
    used_in: '',
  }));

  const rawToolCalls = responseJson.agent_trace?.tool_calls || [];
  const toolCalls = rawToolCalls.map((t) => {
    const toolName = t.tool_name || t.tool || t.name || 'unknown_tool';
    const params = t.inputs || t.params || t.args || {};
    const success = typeof t.success === 'boolean' ? t.success : t.status !== 'error';

    const latencySeconds =
      typeof t.latency_ms === 'number'
        ? t.latency_ms / 1000
        : typeof t.latency === 'number'
          ? t.latency
          : null;

    const resultsSummary =
      t.results_summary ||
      (typeof t.output_size === 'number'
        ? t.output_size > 0
          ? `Output size: ${t.output_size} chars`
          : success
            ? 'Success'
            : 'Error'
        : success
          ? 'Success'
          : 'Error');

    return {
      tool_name: toolName,
      status: success ? 'success' : 'error',
      latency: latencySeconds ?? 0,
      record_count: typeof t.record_count === 'number' ? t.record_count : null,
      params,
      results_summary: resultsSummary,
      error: t.error || null,
    };
  });

  return {
    run_id: runId,
    created_at: createdAt,
    user_message: question,
    assistant_message: responseJson.answer || '',
    tool_calls: toolCalls,
    contexts: [],
    citations,
    logs: {
      orchestrator_status: responseJson.status,
      safety: responseJson.safety,
      relevance: responseJson.relevance,
      model_info: responseJson.agent_trace?.model_info,
    },
  };
}

export const useChatStore = create((set, get) => ({
  // State
  chats: mockChats,
  currentChatId: 'chat-1',
  runsByChatId: {},
  runs: [],
  selectedRunId: null,
  sidebarCollapsed: false,
  searchQuery: '',

  // Actions
  setCurrentChat: (chatId) => {
    const state = get();
    const runs = state.runsByChatId[chatId] || [];
    set({
      currentChatId: chatId,
      runs,
      selectedRunId: runs.length > 0 ? runs[runs.length - 1].run_id : null,
    });
  },
  
  setSelectedRun: (runId) => set({ selectedRunId: runId }),
  
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  
  setSearchQuery: (query) => set({ searchQuery: query }),

  getSelectedRun: () => {
    const state = get();
    return state.runs.find((r) => r.run_id === state.selectedRunId);
  },

  getCurrentChat: () => {
    const state = get();
    return state.chats.find((c) => c.id === state.currentChatId);
  },

  getFilteredChats: () => {
    const state = get();
    let filtered = state.chats;

    if (state.searchQuery) {
      const query = state.searchQuery.toLowerCase();
      filtered = filtered.filter(
        (c) =>
          c.title.toLowerCase().includes(query) ||
          c.tags.some((t) => t.toLowerCase().includes(query))
      );
    }

    return filtered;
  },

  createNewChat: () => {
    const newChat = {
      id: `chat-${Date.now()}`,
      title: 'New Investigation',
      lastUpdated: new Date(),
      tags: [],
    };
    set((state) => ({
      chats: [newChat, ...state.chats],
      currentChatId: newChat.id,
      runsByChatId: { ...state.runsByChatId, [newChat.id]: [] },
      runs: [],
      selectedRunId: null,
    }));
  },

  sendMessage: async (message) => {
    const chatId = get().currentChatId;
    const createdAt = new Date();
    const tempRunId = `run-${Date.now()}`;

    const optimisticRun = {
      run_id: tempRunId,
      created_at: createdAt,
      user_message: message,
      assistant_message: 'Processing your request...',
      tool_calls: [],
      contexts: [],
      citations: [],
      logs: { status: 'pending' },
    };

    set((state) => {
      const updatedRuns = [...state.runs, optimisticRun];
      return {
        runs: updatedRuns,
        runsByChatId: { ...state.runsByChatId, [chatId]: updatedRuns },
        selectedRunId: tempRunId,
      };
    });

    try {
      const resp = await fetch(`${API_BASE_URL}/v1/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: message,
          // session_id can be wired to chat/thread id later
          session_id: get().currentChatId,
        }),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.detail || data?.message || `HTTP ${resp.status}`);
      }

      const runId = data.request_id || tempRunId;
      const mapped = mapQueryResponseToRun({
        question: message,
        createdAt,
        runId,
        responseJson: data,
      });

      set((state) => {
        const updatedRuns = state.runs.map((r) => (r.run_id === tempRunId ? mapped : r));
        return {
          runs: updatedRuns,
          runsByChatId: { ...state.runsByChatId, [chatId]: updatedRuns },
          selectedRunId: runId,
        };
      });
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      set((state) => {
        const updatedRuns = state.runs.map((r) =>
          r.run_id === tempRunId
            ? {
                ...r,
                assistant_message: `Error calling API: ${errorMessage}`,
                logs: { status: 'error', error: errorMessage },
              }
            : r
        );

        return {
          runs: updatedRuns,
          runsByChatId: { ...state.runsByChatId, [chatId]: updatedRuns },
        };
      });
    }
  },
}));

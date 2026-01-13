import { useEffect, useRef } from 'react';
import { useChatStore } from '../../store/chatStore';
import { MessageUser } from './MessageUser';
import { MessageAssistant } from './MessageAssistant';

export function ChatThread() {
  const { runs, selectedRunId, setSelectedRun } = useChatStore();
  const threadRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [runs.length]);

  if (runs.length === 0) {
    return (
      <div
        ref={threadRef}
        data-chat-thread
        className="flex-1 overflow-y-auto flex items-center justify-center"
      >
        <div className="text-center max-w-md px-4">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold text-2xl">CA</span>
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Corrections Agent
          </h2>
          <p className="text-gray-500 text-sm">
            Ask questions about conversations, incidents, and reports. All responses include citations to source evidence.
          </p>
          <div className="mt-6 grid grid-cols-1 gap-2">
            <button className="text-left p-3 bg-white border border-gray-200 rounded-xl hover:border-accent transition-colors">
              <p className="text-sm font-medium text-gray-900">
                Search conversations
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                "What conversations mention drug activity in Block C?"
              </p>
            </button>
            <button className="text-left p-3 bg-white border border-gray-200 rounded-xl hover:border-accent transition-colors">
              <p className="text-sm font-medium text-gray-900">
                Review incidents
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                "Show me incident reports from last week"
              </p>
            </button>
            <button className="text-left p-3 bg-white border border-gray-200 rounded-xl hover:border-accent transition-colors">
              <p className="text-sm font-medium text-gray-900">
                Cross-reference data
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                "Link conversations to incident IR-2026-0042"
              </p>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={threadRef}
      data-chat-thread
      className="flex-1 overflow-y-auto px-4 py-6"
    >
      <div className="max-w-chat mx-auto">
        {runs.map((run) => (
          <div key={run.run_id}>
            {/* User message */}
            <MessageUser
              message={run.user_message}
              timestamp={run.created_at}
            />

            {/* Assistant response */}
            <MessageAssistant
              run={run}
              isSelected={selectedRunId === run.run_id}
              onSelect={() => setSelectedRun(run.run_id)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

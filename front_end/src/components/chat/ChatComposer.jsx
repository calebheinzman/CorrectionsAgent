import { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import { Send } from 'lucide-react';
import { useChatStore } from '../../store/chatStore';

export function ChatComposer() {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);
  const { sendMessage } = useChatStore();

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    sendMessage(message.trim());
    setMessage('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      <div className="max-w-chat mx-auto">
        {/* Composer */}
        <form onSubmit={handleSubmit} className="relative">
          <div className="flex items-end gap-2 bg-gray-50 rounded-2xl border border-gray-200 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20">

            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about conversations, incidents, or reports..."
              rows={1}
              className="flex-1 py-3 pl-4 bg-transparent border-0 resize-none focus:ring-0 text-sm text-gray-900 placeholder-gray-500 max-h-48"
            />

            {/* Right actions */}
            <div className="flex items-center gap-1 pr-2 pb-2">
              <button
                type="submit"
                disabled={!message.trim()}
                className={clsx(
                  'p-2 rounded-xl transition-colors',
                  message.trim()
                    ? 'bg-accent text-white hover:bg-blue-700'
                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                )}
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </form>

        {/* Disclaimer */}
        <p className="text-center text-xs text-gray-400 mt-2">
          Responses include citations to source evidence. Always verify critical information.
        </p>
      </div>
    </div>
  );
}

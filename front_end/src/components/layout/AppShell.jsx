import { useEffect } from 'react';
import { useChatStore } from '../../store/chatStore';
import { Sidebar } from './Sidebar';
import { ChatPage } from '../chat/ChatPage';

export function AppShell() {
  const { setSelectedRun, runs } = useChatStore();

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger shortcuts when typing in inputs
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'g':
          // Scroll to bottom
          const chatThread = document.querySelector('[data-chat-thread]');
          if (chatThread) {
            chatThread.scrollTop = chatThread.scrollHeight;
          }
          break;
        case 'j':
          // Next message
          e.preventDefault();
          const currentIndex = runs.findIndex(
            (r) => r.run_id === useChatStore.getState().selectedRunId
          );
          if (currentIndex < runs.length - 1) {
            setSelectedRun(runs[currentIndex + 1].run_id);
          }
          break;
        case 'k':
          // Previous message
          e.preventDefault();
          const currIdx = runs.findIndex(
            (r) => r.run_id === useChatStore.getState().selectedRunId
          );
          if (currIdx > 0) {
            setSelectedRun(runs[currIdx - 1].run_id);
          }
          break;
        default:
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [runs, setSelectedRun]);

  return (
    <div className="h-screen flex overflow-hidden bg-background-light">
      <Sidebar />
      <ChatPage />
    </div>
  );
}

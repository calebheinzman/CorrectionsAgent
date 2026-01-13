import { ChatTopBar } from './ChatTopBar';
import { ChatThread } from './ChatThread';
import { ChatComposer } from './ChatComposer';

export function ChatPage() {
  return (
    <div className="flex-1 flex flex-col min-w-0 bg-background-light">
      <ChatTopBar />
      <ChatThread />
      <ChatComposer />
    </div>
  );
}

import { clsx } from 'clsx';
import {
  Plus,
  Search,
  ChevronLeft,
  ChevronRight,
  User,
} from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { Chip, Tooltip } from '../primitives';

function formatRelativeTime(date) {
  const now = new Date();
  const diff = now - date;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
}

export function Sidebar() {
  const {
    sidebarCollapsed,
    toggleSidebar,
    searchQuery,
    setSearchQuery,
    getFilteredChats,
    currentChatId,
    setCurrentChat,
    createNewChat,
  } = useChatStore();

  const chats = getFilteredChats();

  if (sidebarCollapsed) {
    return (
      <div className="w-16 h-full bg-white border-r border-gray-200 flex flex-col items-center py-4">
        <Tooltip content="Expand sidebar" position="right">
          <button
            onClick={toggleSidebar}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg mb-4"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </Tooltip>

        <Tooltip content="New Chat" position="right">
          <button
            onClick={createNewChat}
            className="p-2 text-white bg-accent hover:bg-blue-700 rounded-lg mb-4"
          >
            <Plus className="w-5 h-5" />
          </button>
        </Tooltip>

        <div className="flex-1" />
      </div>
    );
  }

  return (
    <div className="w-72 h-full bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">CA</span>
            </div>
            <span className="font-semibold text-gray-900">Corrections Agent</span>
          </div>
          <button
            onClick={toggleSidebar}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>

        <button
          onClick={createNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent text-white rounded-btn hover:bg-blue-700 transition-colors font-medium"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Search */}
      <div className="p-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-accent/40 text-gray-900 placeholder-gray-500"
          />
        </div>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto px-2">
        {chats.map((chat) => (
          <button
            key={chat.id}
            onClick={() => setCurrentChat(chat.id)}
            className={clsx(
              'w-full text-left p-3 rounded-lg mb-1 transition-colors',
              currentChatId === chat.id
                ? 'bg-gray-100'
                : 'hover:bg-gray-50'
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-900 truncate">
                    {chat.title}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {formatRelativeTime(chat.lastUpdated)}
                </div>
              </div>
            </div>
            {chat.tags.length > 0 && (
              <div className="flex gap-1 mt-2 flex-wrap">
                {chat.tags.slice(0, 2).map((tag) => (
                  <Chip key={tag} className="text-[10px] px-1.5 py-0">
                    {tag}
                  </Chip>
                ))}
                {chat.tags.length > 2 && (
                  <span className="text-[10px] text-gray-400">+{chat.tags.length - 2}</span>
                )}
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
              <User className="w-4 h-4 text-gray-600" />
            </div>
            <div>
              <div className="text-sm font-medium text-gray-900">
                Investigator
              </div>
              <div className="text-xs text-gray-500 flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                Production
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

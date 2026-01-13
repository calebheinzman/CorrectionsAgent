import { clsx } from 'clsx';
import {
  Wrench,
  FileText,
  Quote,
  ScrollText,
  ChevronRight,
  ChevronLeft,
} from 'lucide-react';
import { useChatStore } from '../../store/chatStore';
import { ToolsTab } from '../trace/ToolsTab';
import { CitationsTab } from '../trace/CitationsTab';
import { RunLogsTab } from '../trace/RunLogsTab';

const tabs = [
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'citations', label: 'Citations', icon: Quote },
  { id: 'logs', label: 'Run Logs', icon: ScrollText },
];

export function TracePanel() {
  const {
    tracePanelCollapsed,
    toggleTracePanel,
    activeTraceTab,
    setActiveTraceTab,
    getSelectedRun,
  } = useChatStore();

  const selectedRun = getSelectedRun();

  if (tracePanelCollapsed) {
    return (
      <div className="w-12 h-full bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 flex flex-col items-center py-4">
        <button
          onClick={toggleTracePanel}
          className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg mb-4"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setActiveTraceTab(tab.id);
              toggleTracePanel();
            }}
            className={clsx(
              'p-2 rounded-lg mb-1',
              activeTraceTab === tab.id
                ? 'bg-accent/10 text-accent'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800'
            )}
          >
            <tab.icon className="w-5 h-5" />
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="w-96 h-full bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Trace
        </h2>
        <button
          onClick={toggleTracePanel}
          className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-800">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTraceTab(tab.id)}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 text-sm font-medium transition-colors',
              activeTraceTab === tab.id
                ? 'text-accent border-b-2 border-accent'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            )}
          >
            <tab.icon className="w-4 h-4" />
            <span className="hidden lg:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto">
        {!selectedRun ? (
          <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400 text-sm">
            Select a message to view trace details
          </div>
        ) : (
          <>
            {activeTraceTab === 'tools' && <ToolsTab run={selectedRun} />}
            {activeTraceTab === 'citations' && <CitationsTab run={selectedRun} />}
            {activeTraceTab === 'logs' && <RunLogsTab run={selectedRun} />}
          </>
        )}
      </div>
    </div>
  );
}

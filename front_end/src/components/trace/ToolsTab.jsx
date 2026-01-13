import { useState } from 'react';
import { clsx } from 'clsx';
import {
  CheckCircle,
  AlertCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  MessageSquare,
  FileText,
  AlertTriangle,
  RefreshCw,
  Shield,
} from 'lucide-react';
import { Chip } from '../primitives';

function ToolCard({ tool }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const statusConfig = {
    success: {
      icon: CheckCircle,
      color: 'text-green-600 dark:text-green-400',
      bg: 'bg-green-100 dark:bg-green-900/30',
      label: 'Success',
    },
    partial: {
      icon: AlertCircle,
      color: 'text-amber-600 dark:text-amber-400',
      bg: 'bg-amber-100 dark:bg-amber-900/30',
      label: 'Partial',
    },
    error: {
      icon: AlertTriangle,
      color: 'text-red-600 dark:text-red-400',
      bg: 'bg-red-100 dark:bg-red-900/30',
      label: 'Error',
    },
    running: {
      icon: RefreshCw,
      color: 'text-blue-600 dark:text-blue-400',
      bg: 'bg-blue-100 dark:bg-blue-900/30',
      label: 'Running',
    },
  };

  const status = statusConfig[tool.status] || statusConfig.success;
  const StatusIcon = status.icon;

  const toolIcons = {
    'Conversations Search': MessageSquare,
    'Incident Reports Search': AlertTriangle,
    'User Reports Search': FileText,
    'Conversation Details': MessageSquare,
  };

  const ToolIcon = toolIcons[tool.tool_name] || FileText;

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-panel overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex-shrink-0">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>

        <div className="flex-shrink-0 w-8 h-8 bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
          <ToolIcon className="w-4 h-4 text-gray-600 dark:text-gray-400" />
        </div>

        <div className="flex-1 text-left">
          <div className="text-sm font-medium text-gray-900 dark:text-white">
            {tool.tool_name}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {tool.latency}s
            </span>
            <span className="text-xs text-gray-400">•</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {tool.record_count} records
            </span>
          </div>
        </div>

        <Chip
          variant={tool.status === 'success' ? 'success' : tool.status === 'error' ? 'error' : 'warning'}
          className="flex items-center gap-1"
        >
          <StatusIcon className={clsx('w-3 h-3', tool.status === 'running' && 'animate-spin')} />
          {status.label}
        </Chip>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-100 dark:border-gray-800">
          <div className="pt-3 space-y-3">
            {/* Query Parameters */}
            <div>
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1.5">
                Parameters
              </div>
              <pre className="text-xs bg-gray-50 dark:bg-gray-800 p-2 rounded-lg overflow-x-auto text-gray-700 dark:text-gray-300">
                {JSON.stringify(tool.params, null, 2)}
              </pre>
            </div>

            {/* Results Summary */}
            <div>
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1.5">
                Results
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {tool.results_summary}
              </p>
            </div>

            {/* Tool Policy */}
            <div className="flex items-start gap-2 p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Shield className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-blue-700 dark:text-blue-300">
                This tool returns only records you have access to. Sensitive fields are redacted by policy.
              </p>
            </div>

            {tool.error && (
              <div className="flex items-start gap-2 p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs text-red-700 dark:text-red-300">
                    {tool.error}
                  </p>
                  <button className="text-xs text-red-600 dark:text-red-400 underline mt-1">
                    Retry tool call
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function ToolsTab({ run }) {
  if (!run.tool_calls || run.tool_calls.length === 0) {
    return (
      <div className="p-4">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-3">
            <Clock className="w-6 h-6 text-gray-400" />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No tool calls for this response
          </p>
        </div>
      </div>
    );
  }

  const successCount = run.tool_calls.filter((t) => t.status === 'success').length;
  const errorCount = run.tool_calls.filter((t) => t.status === 'error').length;
  const totalLatency = run.tool_calls.reduce((sum, t) => sum + (t.latency || 0), 0);

  return (
    <div className="p-4">
      {/* Summary */}
      <div className="flex items-center gap-4 mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900 dark:text-white">
            {run.tool_calls.length}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Calls</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-green-600 dark:text-green-400">
            {successCount}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Success</div>
        </div>
        {errorCount > 0 && (
          <div className="text-center">
            <div className="text-lg font-semibold text-red-600 dark:text-red-400">
              {errorCount}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Errors</div>
          </div>
        )}
        <div className="text-center">
          <div className="text-lg font-semibold text-gray-900 dark:text-white">
            {totalLatency.toFixed(1)}s
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Total</div>
        </div>
      </div>

      {/* Tool Cards */}
      <div className="space-y-3">
        {run.tool_calls.map((tool, index) => (
          <ToolCard key={index} tool={tool} />
        ))}
      </div>
    </div>
  );
}

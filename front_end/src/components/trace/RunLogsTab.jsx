import { clsx } from 'clsx';
import {
  Clock,
  Cpu,
  Zap,
  Shield,
  Download,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';
import { Chip } from '../primitives';

function LogSection({ title, icon: Icon, children }) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-gray-500 dark:text-gray-400" />
        <h3 className="text-sm font-medium text-gray-900 dark:text-white">{title}</h3>
      </div>
      <div className="ml-6">{children}</div>
    </div>
  );
}

function LogItem({ label, value, subtext }) {
  return (
    <div className="flex items-baseline justify-between py-1.5 border-b border-gray-100 dark:border-gray-800 last:border-0">
      <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
      <div className="text-right">
        <span className="text-sm font-medium text-gray-900 dark:text-white">{value}</span>
        {subtext && (
          <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">{subtext}</span>
        )}
      </div>
    </div>
  );
}

export function RunLogsTab({ run }) {
  const logs = run.logs || {};

  const formatTimestamp = (date) => {
    if (!date) return 'N/A';
    return new Date(date).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div className="p-4">
      {/* Timeline */}
      <LogSection title="Timeline" icon={Clock}>
        <div className="relative pl-4 border-l-2 border-gray-200 dark:border-gray-700 space-y-3">
          <div className="relative">
            <div className="absolute -left-[21px] w-3 h-3 bg-accent rounded-full" />
            <div className="text-sm">
              <span className="font-medium text-gray-900 dark:text-white">User prompt</span>
              <span className="text-gray-500 dark:text-gray-400 ml-2">
                {formatTimestamp(run.created_at)}
              </span>
            </div>
          </div>
          {run.tool_calls?.map((tool, index) => (
            <div key={index} className="relative">
              <div
                className={clsx(
                  'absolute -left-[21px] w-3 h-3 rounded-full',
                  tool.status === 'success' ? 'bg-green-500' : 'bg-red-500'
                )}
              />
              <div className="text-sm">
                <span className="font-medium text-gray-900 dark:text-white">{tool.tool_name}</span>
                <span className="text-gray-500 dark:text-gray-400 ml-2">{tool.latency}s</span>
              </div>
            </div>
          ))}
          <div className="relative">
            <div className="absolute -left-[21px] w-3 h-3 bg-green-500 rounded-full" />
            <div className="text-sm">
              <span className="font-medium text-gray-900 dark:text-white">Response complete</span>
              <span className="text-gray-500 dark:text-gray-400 ml-2">
                {logs.total_latency ? `${logs.total_latency}s total` : ''}
              </span>
            </div>
          </div>
        </div>
      </LogSection>

      {/* Model Info */}
      <LogSection title="Model" icon={Cpu}>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
          <LogItem label="Version" value={logs.model_version || 'N/A'} />
          <LogItem label="Temperature" value={logs.temperature ?? 'N/A'} />
        </div>
      </LogSection>

      {/* Token Usage */}
      <LogSection title="Token Usage" icon={Zap}>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
          <LogItem label="Prompt tokens" value={logs.prompt_tokens?.toLocaleString() || 'N/A'} />
          <LogItem label="Completion tokens" value={logs.completion_tokens?.toLocaleString() || 'N/A'} />
          <LogItem
            label="Total tokens"
            value={logs.total_tokens?.toLocaleString() || 'N/A'}
          />
        </div>
      </LogSection>

      {/* Safety Filters */}
      <LogSection title="Safety Filters" icon={Shield}>
        {logs.safety_filters && logs.safety_filters.length > 0 ? (
          <div className="space-y-2">
            {logs.safety_filters.map((filter, index) => (
              <div
                key={index}
                className="flex items-start gap-2 p-2 bg-amber-50 dark:bg-amber-900/20 rounded-lg"
              >
                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                    {filter.category}
                  </p>
                  <p className="text-xs text-amber-700 dark:text-amber-300">{filter.reason}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-2 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
            <span className="text-sm text-green-700 dark:text-green-300">
              No safety filters triggered
            </span>
          </div>
        )}
      </LogSection>

      {/* Export */}
      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <button className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors">
          <Download className="w-4 h-4" />
          Export run logs (JSON)
        </button>
      </div>
    </div>
  );
}

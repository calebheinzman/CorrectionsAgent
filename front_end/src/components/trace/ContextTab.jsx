import { useState } from 'react';
import { clsx } from 'clsx';
import {
  MessageSquare,
  FileText,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Pin,
  StickyNote,
  Clock,
  User,
  MapPin,
} from 'lucide-react';
import { Chip } from '../primitives';

const sourceTypeConfig = {
  conversation: {
    icon: MessageSquare,
    label: 'Conversation',
    color: 'text-blue-600 dark:text-blue-400',
    bg: 'bg-blue-100 dark:bg-blue-900/30',
  },
  user_report: {
    icon: FileText,
    label: 'User Report',
    color: 'text-purple-600 dark:text-purple-400',
    bg: 'bg-purple-100 dark:bg-purple-900/30',
  },
  incident_report: {
    icon: AlertTriangle,
    label: 'Incident Report',
    color: 'text-amber-600 dark:text-amber-400',
    bg: 'bg-amber-100 dark:bg-amber-900/30',
  },
};

function RelevanceMeter({ level }) {
  const levels = {
    high: { width: 'w-full', color: 'bg-green-500', label: 'High match' },
    medium: { width: 'w-2/3', color: 'bg-amber-500', label: 'Medium match' },
    low: { width: 'w-1/3', color: 'bg-gray-400', label: 'Low match' },
  };

  const config = levels[level] || levels.medium;

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full', config.width, config.color)} />
      </div>
      <span className="text-xs text-gray-500 dark:text-gray-400">{config.label}</span>
    </div>
  );
}

function ContextItem({ context }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const config = sourceTypeConfig[context.source_type] || sourceTypeConfig.conversation;
  const Icon = config.icon;

  const formatTimestamp = (ts) => {
    if (!ts) return '';
    const date = new Date(ts);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-panel overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-start gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors text-left"
      >
        <div className="flex-shrink-0 mt-0.5">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>

        <div className={clsx('flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center', config.bg)}>
          <Icon className={clsx('w-4 h-4', config.color)} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {context.source_id}
            </span>
            <Chip variant="default" className="text-[10px]">
              {config.label}
            </Chip>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
            {context.snippet}
          </p>
          <div className="mt-2">
            <RelevanceMeter level={context.relevance} />
          </div>
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-100 dark:border-gray-800">
          <div className="pt-3 space-y-3">
            {/* Full Snippet */}
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                {context.snippet}
              </p>
            </div>

            {/* Metadata */}
            {context.metadata && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                {context.metadata.timestamp && (
                  <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
                    <Clock className="w-3.5 h-3.5" />
                    {formatTimestamp(context.metadata.timestamp)}
                  </div>
                )}
                {context.metadata.facility && (
                  <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
                    <MapPin className="w-3.5 h-3.5" />
                    {context.metadata.facility}
                    {context.metadata.cell && `, ${context.metadata.cell}`}
                  </div>
                )}
                {context.metadata.speakers && (
                  <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400 col-span-2">
                    <User className="w-3.5 h-3.5" />
                    {context.metadata.speakers.join(', ')}
                  </div>
                )}
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-2 pt-2">
              <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <ExternalLink className="w-3.5 h-3.5" />
                Open full record
              </button>
              <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Pin className="w-3.5 h-3.5" />
                Pin snippet
              </button>
              <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <StickyNote className="w-3.5 h-3.5" />
                Add note
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ContextGroup({ type, contexts }) {
  const [isExpanded, setIsExpanded] = useState(true);
  const config = sourceTypeConfig[type] || sourceTypeConfig.conversation;
  const Icon = config.icon;

  return (
    <div className="mb-4">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 w-full text-left mb-2"
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400" />
        )}
        <Icon className={clsx('w-4 h-4', config.color)} />
        <span className="text-sm font-medium text-gray-900 dark:text-white">
          {config.label}s
        </span>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          ({contexts.length})
        </span>
      </button>

      {isExpanded && (
        <div className="space-y-2 ml-6">
          {contexts.map((context, index) => (
            <ContextItem key={index} context={context} />
          ))}
        </div>
      )}
    </div>
  );
}

export function ContextTab({ run }) {
  if (!run.contexts || run.contexts.length === 0) {
    return (
      <div className="p-4">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-3">
            <FileText className="w-6 h-6 text-gray-400" />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
            No context retrieved
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Try broadening your search filters
          </p>
        </div>
      </div>
    );
  }

  // Group contexts by source type
  const grouped = run.contexts.reduce((acc, context) => {
    const type = context.source_type;
    if (!acc[type]) acc[type] = [];
    acc[type].push(context);
    return acc;
  }, {});

  return (
    <div className="p-4">
      {/* Summary */}
      <div className="flex items-center gap-4 mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
        <div className="text-sm text-gray-600 dark:text-gray-400">
          <span className="font-semibold text-gray-900 dark:text-white">
            {run.contexts.length}
          </span>{' '}
          sources retrieved
        </div>
      </div>

      {/* Grouped Contexts */}
      {Object.entries(grouped).map(([type, contexts]) => (
        <ContextGroup key={type} type={type} contexts={contexts} />
      ))}
    </div>
  );
}

import { clsx } from 'clsx';
import {
  MessageSquare,
  FileText,
  AlertTriangle,
  ExternalLink,
  ArrowRight,
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

function CitationCard({ citation }) {
  const config = sourceTypeConfig[citation.source_type] || sourceTypeConfig.conversation;
  const Icon = config.icon;

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-panel p-4">
      {/* Citation Number & Source */}
      <div className="flex items-start gap-3 mb-3">
        <div className="flex-shrink-0 w-7 h-7 bg-accent text-white rounded-full flex items-center justify-center text-sm font-semibold">
          {citation.citation_number}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <div className={clsx('w-5 h-5 rounded flex items-center justify-center', config.bg)}>
              <Icon className={clsx('w-3 h-3', config.color)} />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {citation.source_id}
            </span>
            <Chip variant="default" className="text-[10px]">
              {config.label}
            </Chip>
          </div>
        </div>
      </div>

      {/* Snippet */}
      <div className="mb-3 p-2.5 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <p className="text-sm text-gray-700 dark:text-gray-300 italic">
          "{citation.snippet}"
        </p>
      </div>

      {/* Used In */}
      <div className="flex items-start gap-2 text-xs text-gray-500 dark:text-gray-400">
        <ArrowRight className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
        <p className="line-clamp-2">{citation.used_in}</p>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
        <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-accent hover:bg-accent/10 rounded-lg transition-colors">
          <ExternalLink className="w-3.5 h-3.5" />
          View source
        </button>
        <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
          Show in chat
        </button>
      </div>
    </div>
  );
}

export function CitationsTab({ run }) {
  if (!run.citations || run.citations.length === 0) {
    return (
      <div className="p-4">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mb-3">
            <FileText className="w-6 h-6 text-gray-400" />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No citations in this response
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4">
      {/* Summary */}
      <div className="flex items-center gap-4 mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
        <div className="text-sm text-gray-600 dark:text-gray-400">
          <span className="font-semibold text-gray-900 dark:text-white">
            {run.citations.length}
          </span>{' '}
          citations used
        </div>
      </div>

      {/* Citation Cards */}
      <div className="space-y-3">
        {run.citations.map((citation) => (
          <CitationCard key={citation.citation_number} citation={citation} />
        ))}
      </div>
    </div>
  );
}

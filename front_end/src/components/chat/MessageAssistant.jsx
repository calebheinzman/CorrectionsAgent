import { useState, isValidElement, cloneElement } from 'react';
import { clsx } from 'clsx';
import ReactMarkdown from 'react-markdown';
import {
  Copy,
  FileText,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Shield,
  Check,
} from 'lucide-react';
import { Tooltip } from '../primitives';

function CitationInline({ number, onClick }) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-semibold bg-accent/10 text-accent hover:bg-accent/20 rounded ml-0.5 transition-colors"
    >
      {number}
    </button>
  );
}

function EvidenceStrip({ run, isExpanded, onToggle }) {
  const toolErrors = run.tool_calls?.filter((t) => t.status === 'error').length || 0;
  const totalLatency = run.logs?.total_latency || 0;
  const toolCallCount = run.tool_calls?.length || 0;

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <button
        onClick={onToggle}
        className="flex items-center justify-between w-full text-left"
      >
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span>{toolCallCount} tool calls</span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
          <span>Last tool run: {totalLatency}s</span>
          {toolErrors > 0 ? (
            <span className="text-red-500">{toolErrors} errors</span>
          ) : (
            <span className="text-green-600">no errors</span>
          )}
        </div>
      )}
    </div>
  );
}

function DetailsSection({ title, children, defaultOpen = false, anchorId }) {
  return (
    <details
      className="border border-gray-200 rounded-xl bg-white"
      open={defaultOpen}
      id={anchorId}
    >
      <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-gray-900">
        {title}
      </summary>
      <div className="px-3 pb-3">{children}</div>
    </details>
  );
}

function ToolsDetails({ run }) {
  const toolCalls = run.tool_calls || [];

  if (toolCalls.length === 0) {
    return <div className="text-sm text-gray-500">No tool calls.</div>;
  }

  return (
    <div className="space-y-2">
      {toolCalls.map((tool, idx) => (
        (() => {
          const toolName = tool.tool_name || tool.name || tool.tool || 'unknown_tool';
          const params = tool.params || tool.inputs || tool.args || {};

          const status =
            tool.status ||
            (typeof tool.success === 'boolean' ? (tool.success ? 'success' : 'error') : undefined) ||
            'success';

          const latencySeconds =
            typeof tool.latency === 'number'
              ? tool.latency
              : typeof tool.latency_ms === 'number'
                ? tool.latency_ms / 1000
                : 0;

          const recordCount = typeof tool.record_count === 'number' ? tool.record_count : null;

          const resultsSummary =
            tool.results_summary ||
            (typeof tool.output_size === 'number'
              ? tool.output_size > 0
                ? `Output size: ${tool.output_size} chars`
                : status === 'success'
                  ? 'Success'
                  : 'Error'
              : status === 'success'
                ? 'Success'
                : 'Error');

          return (
            <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-gray-50 flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium text-gray-900 truncate">{toolName}</div>
                  <div className="text-xs text-gray-500">
                    {latencySeconds}s • {recordCount === null ? 'n/a' : recordCount} records • {status}
                  </div>
                </div>
              </div>
              <div className="p-3 space-y-2">
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Parameters</div>
                  <pre className="text-xs bg-gray-50 p-2 rounded-lg overflow-x-auto text-gray-700">{JSON.stringify(params, null, 2)}</pre>
                </div>
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Results</div>
                  <div className="text-sm text-gray-700">{resultsSummary}</div>
                </div>
                {tool.error && (
                  <div className="text-sm text-red-600">{tool.error}</div>
                )}
              </div>
            </div>
          );
        })()
      ))}
    </div>
  );
}

function CitationsDetails({ run }) {
  const citations = run.citations || [];
  if (citations.length === 0) return <div className="text-sm text-gray-500">No citations.</div>;

  return (
    <div className="space-y-2">
      {citations.map((c) => (
        <div key={c.citation_number} className="border border-gray-200 rounded-lg p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-sm font-semibold text-gray-900">[{c.citation_number}] {c.source_id}</div>
            <div className="text-xs text-gray-500">{c.source_type}</div>
          </div>
          <div className="mt-2 text-sm text-gray-700 italic">"{c.snippet}"</div>
          <div className="mt-2 text-xs text-gray-500">Used in: {c.used_in}</div>
        </div>
      ))}
    </div>
  );
}

export function MessageAssistant({ run, isSelected, onSelect }) {
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCitationClick = () => {
    onSelect();
    setDetailsExpanded(true);
    setTimeout(() => {
      const el = document.getElementById(`citations-${run.run_id}`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 0);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(run.assistant_message);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyWithCitations = async () => {
    let text = run.assistant_message;
    if (run.citations?.length > 0) {
      text += '\n\n---\nSources:\n';
      run.citations.forEach((c) => {
        text += `[${c.citation_number}] ${c.source_id}\n`;
      });
    }
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Check if this is a safety refusal
  const isSafetyRefusal = run.logs?.safety_filters?.length > 0;

  const renderTextWithCitations = (text) => {
    const citationRegex = /\[(\d+)\](?!\()/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = citationRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }

      parts.push(
        <CitationInline
          key={`citation-${match[1]}-${match.index}`}
          number={match[1]}
          onClick={() => handleCitationClick()}
        />
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  const processMarkdownChildren = (node) => {
    if (typeof node === 'string') {
      return renderTextWithCitations(node);
    }

    if (Array.isArray(node)) {
      return node.flatMap((n) => processMarkdownChildren(n));
    }

    if (isValidElement(node)) {
      const nextChildren = processMarkdownChildren(node.props?.children);
      return cloneElement(node, { ...node.props }, nextChildren);
    }

    return node;
  };

  // Parse markdown and inject citation buttons
  const renderContent = () => {
    return (
      <div className="markdown-content text-sm text-gray-900 leading-relaxed">
        <ReactMarkdown
          components={{
            p: ({ children }) => {
              return <p className="mb-3 last:mb-0">{processMarkdownChildren(children)}</p>;
            },
            li: ({ children }) => {
              return <li>{processMarkdownChildren(children)}</li>;
            },
            blockquote: ({ children }) => {
              return <blockquote>{processMarkdownChildren(children)}</blockquote>;
            },
            h1: ({ children }) => {
              return <h1>{processMarkdownChildren(children)}</h1>;
            },
            h2: ({ children }) => {
              return <h2>{processMarkdownChildren(children)}</h2>;
            },
            h3: ({ children }) => {
              return <h3>{processMarkdownChildren(children)}</h3>;
            },
            h4: ({ children }) => {
              return <h4>{processMarkdownChildren(children)}</h4>;
            },
            h5: ({ children }) => {
              return <h5>{processMarkdownChildren(children)}</h5>;
            },
            h6: ({ children }) => {
              return <h6>{processMarkdownChildren(children)}</h6>;
            },
          }}
        >
          {run.assistant_message}
        </ReactMarkdown>
      </div>
    );
  };

  return (
    <div
      className={clsx(
        'mb-6 transition-all',
        isSelected && 'ring-2 ring-accent/30 rounded-xl'
      )}
      onClick={onSelect}
    >
      <div className="flex gap-3">
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
          <span className="text-white font-bold text-xs">CA</span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Safety indicator */}
          {isSafetyRefusal && (
            <div className="flex items-center gap-2 mb-2">
              <Tooltip content="Request blocked by safety policy">
                <div className="flex items-center gap-1.5 px-2 py-1 bg-amber-100 rounded-lg">
                  <Shield className="w-4 h-4 text-amber-600" />
                  <span className="text-xs font-medium text-amber-700">
                    Safety filter applied
                  </span>
                </div>
              </Tooltip>
            </div>
          )}

          {/* Message content */}
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
            {renderContent()}

            {/* Evidence strip */}
            {run.tool_calls?.length > 0 && (
              <EvidenceStrip
                run={run}
                isExpanded={evidenceExpanded}
                onToggle={() => setEvidenceExpanded(!evidenceExpanded)}
              />
            )}

            {/* Details dropdown */}
            <div className="mt-3 pt-3 border-t border-gray-100">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setDetailsExpanded((v) => !v);
                }}
                className="w-full flex items-center justify-between text-left"
              >
                <span className="text-sm font-medium text-gray-900">Details</span>
                {detailsExpanded ? (
                  <ChevronUp className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                )}
              </button>

              {detailsExpanded && (
                <div className="mt-3 space-y-3">
                  <DetailsSection title="Tools" defaultOpen>
                    <ToolsDetails run={run} />
                  </DetailsSection>
                  <DetailsSection title="Citations" anchorId={`citations-${run.run_id}`}>
                    <CitationsDetails run={run} />
                  </DetailsSection>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 mt-2">
            <Tooltip content={copied ? 'Copied!' : 'Copy'}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleCopy();
                }}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </button>
            </Tooltip>

            <Tooltip content="Copy with citations">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleCopyWithCitations();
                }}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <FileText className="w-4 h-4" />
              </button>
            </Tooltip>

            <Tooltip content="Regenerate">
              <button
                onClick={(e) => e.stopPropagation()}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </Tooltip>
          </div>
        </div>
      </div>
    </div>
  );
}

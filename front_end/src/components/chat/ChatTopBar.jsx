import {
  Download,
  MoreHorizontal,
  Edit2,
  Trash2,
} from 'lucide-react';
import { useState } from 'react';
import { jsPDF } from 'jspdf';
import { useChatStore } from '../../store/chatStore';
import { Tooltip } from '../primitives';

export function ChatTopBar() {
  const { getCurrentChat, runs } = useChatStore();
  const [showMenu, setShowMenu] = useState(false);
  const chat = getCurrentChat();

  const exportPdf = () => {
    const doc = new jsPDF({ unit: 'pt', format: 'letter' });
    const margin = 48;
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const maxWidth = pageWidth - margin * 2;

    const title = chat?.title || 'Chat Export';

    let y = margin;

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(16);
    const titleLines = doc.splitTextToSize(title, maxWidth);
    doc.text(titleLines, margin, y);
    y += titleLines.length * 18 + 10;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.text(`Exported: ${new Date().toLocaleString()}`, margin, y);
    y += 18;

    const ensureSpace = (linesCount) => {
      const needed = linesCount * 14 + 18;
      if (y + needed > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
    };

    const addBlock = (label, text) => {
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(11);
      ensureSpace(2);
      doc.text(label, margin, y);
      y += 14;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(11);
      const lines = doc.splitTextToSize(text || '', maxWidth);
      ensureSpace(lines.length);
      doc.text(lines, margin, y);
      y += lines.length * 14 + 12;
    };

    (runs || []).forEach((r) => {
      addBlock('User', r.user_message);
      addBlock('Assistant', r.assistant_message);
    });

    const safeName = (title || 'chat')
      .toLowerCase()
      .replace(/[^a-z0-9\-\s]/g, '')
      .trim()
      .replace(/\s+/g, '-');

    doc.save(`${safeName || 'chat'}-export.pdf`);
  };

  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white">
      <div className="flex items-center gap-3">
        <h1 className="text-base font-semibold text-gray-900 truncate max-w-md">
          {chat?.title || 'New Chat'}
        </h1>
        <button className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded">
          <Edit2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex items-center gap-1">
        <Tooltip content="Export PDF">
          <button
            onClick={exportPdf}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Download className="w-4 h-4" />
          </button>
        </Tooltip>

        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>

          {showMenu && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setShowMenu(false)}
              />
              <div className="absolute right-0 top-full mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden z-20">
                <button className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50">
                  <Edit2 className="w-4 h-4" />
                  Rename chat
                </button>
                <button
                  onClick={() => {
                    setShowMenu(false);
                    exportPdf();
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  <Download className="w-4 h-4" />
                  Export as PDF
                </button>
                <div className="border-t border-gray-100" />
                <button className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50">
                  <Trash2 className="w-4 h-4" />
                  Delete chat
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

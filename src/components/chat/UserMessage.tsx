import { useState } from 'react';
import { Copy, Trash2, Check } from 'lucide-react';

interface UserMessageProps {
  question: string;
  onDelete: () => void;
}

export function UserMessage({ question, onDelete }: UserMessageProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(question);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  return (
    <div className="flex flex-col items-end gap-1">
      <div className="max-w-[78%] rounded-2xl rounded-tr-md bg-[var(--primary)] px-4 py-3 text-[15px] leading-6 text-white shadow-sm sm:max-w-[72%]">
        {question}
      </div>
      <div className="mr-2 flex items-center gap-2 opacity-60 transition-opacity hover:opacity-100">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 rounded p-1 text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]"
          title="Sao chép câu hỏi"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
          <span>{copied ? 'Đã chép' : 'Sao chép'}</span>
        </button>
        <button
          onClick={onDelete}
          className="flex items-center gap-1.5 rounded p-1 text-[11px] font-medium text-rose-600 hover:bg-rose-50"
          title="Xóa câu hỏi"
        >
          <Trash2 className="h-3.5 w-3.5" />
          <span>Xóa</span>
        </button>
      </div>
    </div>
  );
}

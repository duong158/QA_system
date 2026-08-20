import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { FilePlus2, Send, Square } from 'lucide-react';
import { VoiceButton } from '@/components/input/VoiceButton';
import { mergeQuestionParts } from '@/utils/questionInput';

interface ChatComposerProps {
  value: string;
  transcript: string;
  interimTranscript: string;
  listening: boolean;
  speechSupported: boolean;
  audioActive: boolean;
  submitting: boolean;
  socraticEnabled: boolean;
  onSocraticEnabledChange: (enabled: boolean) => void;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onVoiceToggle: () => void;
  onStopSpeaking: () => void;
}

export function ChatComposer({
  value,
  transcript,
  interimTranscript,
  listening,
  speechSupported,
  audioActive,
  submitting,
  socraticEnabled,
  onSocraticEnabledChange,
  onChange,
  onSubmit,
  onVoiceToggle,
  onStopSpeaking,
}: ChatComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const hasSpeechInput = Boolean(transcript.trim() || interimTranscript.trim());
  const displayValue = hasSpeechInput ? mergeQuestionParts(value, transcript, interimTranscript) : value;
  const canSubmit = Boolean(displayValue.trim()) && !submitting;

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }, [displayValue]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    if (canSubmit) onSubmit();
  };

  return (
    <div className="shrink-0 border-t border-[var(--border)] bg-[var(--surface)] px-3 pb-3 pt-2 sm:px-4 sm:pb-4">
      <div className="mb-2 flex items-center justify-between sm:hidden">
        <span className="text-xs text-[var(--text-secondary)]">Chế độ Gia sư</span>
        <button
          type="button"
          role="switch"
          aria-checked={socraticEnabled}
          onClick={() => onSocraticEnabledChange(!socraticEnabled)}
          className={`relative h-6 w-11 rounded-full p-0.5 ${socraticEnabled ? 'bg-[var(--socratic)]' : 'bg-slate-300'}`}
        >
          <span className={`block h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${socraticEnabled ? 'translate-x-[18px]' : ''}`} />
        </button>
      </div>

      <div className="flex items-end gap-2 rounded-2xl border border-[var(--border-strong)] bg-[var(--surface)] p-2 shadow-sm transition focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-50">
        <Link
          to="/knowledge-blind-spots#document-contribution"
          aria-label="Đóng góp tài liệu"
          title="Đóng góp tài liệu"
          className="soft-icon-button h-10 w-10 shrink-0 border-0"
        >
          <FilePlus2 className="h-5 w-5" />
        </Link>
        <textarea
          ref={textareaRef}
          value={displayValue}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={submitting}
          placeholder="Hỏi Mari về tài liệu..."
          aria-label="Nhập câu hỏi cho Mari"
          className="max-h-40 min-h-10 min-w-0 flex-1 resize-none overflow-y-auto bg-transparent px-1 py-2 text-[15px] leading-6 text-[var(--text-primary)] outline-none placeholder:text-[var(--text-muted)] disabled:cursor-wait"
        />
        {audioActive ? (
          <button type="button" onClick={onStopSpeaking} aria-label="Dừng đọc" title="Dừng đọc" className="soft-icon-button h-10 w-10 shrink-0">
            <Square className="h-4 w-4" />
          </button>
        ) : null}
        <VoiceButton listening={listening} supported={speechSupported} onToggle={onVoiceToggle} />
        <button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit}
          aria-label="Gửi câu hỏi"
          className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-xl bg-[var(--primary)] px-3.5 text-sm font-semibold text-white transition hover:bg-[var(--primary-hover)] disabled:cursor-not-allowed disabled:bg-slate-300 sm:px-4"
        >
          <Send className="h-4 w-4" />
          <span className="hidden sm:inline">{submitting ? 'Đang gửi' : 'Gửi'}</span>
        </button>
      </div>
      <p className="mt-1.5 px-2 text-[11px] text-[var(--text-muted)]">Enter để gửi · Shift + Enter để xuống dòng</p>
    </div>
  );
}

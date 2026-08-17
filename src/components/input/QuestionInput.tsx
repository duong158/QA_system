import { motion } from 'framer-motion';
import { Eraser, Send, Square } from 'lucide-react';
import { VoiceButton } from './VoiceButton';
import { VoiceWaveform } from './VoiceWaveform';

interface QuestionInputProps {
  value: string;
  transcript: string;
  interimTranscript: string;
  listening: boolean;
  speechSupported: boolean;
  audioActive: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onClear: () => void;
  onVoiceToggle: () => void;
  onStopSpeaking: () => void;
}

export function QuestionInput({
  value,
  transcript,
  interimTranscript,
  listening,
  speechSupported,
  audioActive,
  onChange,
  onSubmit,
  onClear,
  onVoiceToggle,
  onStopSpeaking,
}: QuestionInputProps) {
  const displayValue = transcript || interimTranscript ? `${value}${value ? ' ' : ''}${transcript}${interimTranscript}`.trim() : value;

  return (
    <motion.section
      initial={{ y: 16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="viqa-panel sticky bottom-4 z-20 mt-4 p-2.5 lg:p-3"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <VoiceButton listening={listening} supported={speechSupported} onToggle={onVoiceToggle} />
          <div className="min-w-0 flex-1 rounded-xl border border-slate-400/20 bg-slate-700/55 px-3 py-2 shadow-inner shadow-slate-950/15 transition focus-within:border-viqa-cyan/45 focus-within:shadow-[0_0_0_3px_rgba(56,189,248,0.07)]">
            <textarea
              value={displayValue}
              onChange={(event) => onChange(event.target.value)}
              rows={1}
              placeholder="Hỏi Mari về tài liệu..."
              aria-label="Ô nhập câu hỏi"
              className="w-full resize-none bg-transparent text-[15px] leading-6 text-slate-50 outline-none placeholder:text-slate-400"
            />
          </div>
        </div>

        <div className="flex shrink-0 items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClear}
            aria-label="Clear question"
            title="Clear question"
            className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-400/15 bg-slate-700/45 text-slate-200 transition hover:border-slate-300/25"
          >
            <Eraser className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={onStopSpeaking}
            aria-label="Stop speaking"
            title="Stop speaking"
            className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-400/15 bg-slate-700/45 text-slate-200 transition hover:border-viqa-gold/30 hover:text-viqa-gold"
          >
            <Square className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={onSubmit}
            aria-label="Send question"
            className="inline-flex h-10 items-center gap-2 rounded-xl border border-sky-300/20 bg-viqa-cyan px-4 font-semibold text-slate-950 transition hover:bg-sky-300"
          >
            <Send className="h-4 w-4" />
            <span>Gửi</span>
          </button>
        </div>
      </div>

      <div className="mt-2 hidden items-center justify-between gap-3 px-1 text-xs text-slate-400 sm:flex">
        <div className="flex items-center gap-3">
          <span>Speech</span>
          <VoiceWaveform amplitude={audioActive ? 0.7 : 0.15} active={listening || audioActive} />
        </div>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${speechSupported ? 'bg-emerald-300' : 'bg-amber-300'}`} />
          {speechSupported ? 'Speech API ready' : 'Text input only'}
        </div>
      </div>
    </motion.section>
  );
}

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
      className="viqa-panel sticky bottom-4 z-20 mt-4 rounded-[30px] px-4 py-4 lg:px-5"
    >
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <VoiceButton listening={listening} supported={speechSupported} onToggle={onVoiceToggle} />
          <div className="min-w-0 flex-1 rounded-[24px] border border-white/10 bg-black/20 px-4 py-3 shadow-inner shadow-black/30">
            <textarea
              value={displayValue}
              onChange={(event) => onChange(event.target.value)}
              rows={2}
              placeholder="Hỏi VIQA về tập tài liệu tiếng Việt..."
              aria-label="Ô nhập câu hỏi"
              className="w-full resize-none bg-transparent text-[15px] leading-6 text-slate-100 outline-none placeholder:text-slate-500"
            />
          </div>
          <button
            type="button"
            onClick={onClear}
            aria-label="Xóa câu hỏi"
            className="inline-flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-200 transition hover:border-white/20"
          >
            <Eraser className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={onStopSpeaking}
            aria-label="Dừng giọng đọc"
            className="inline-flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-200 transition hover:border-viqa-gold/30 hover:text-viqa-gold"
          >
            <Square className="h-5 w-5" />
          </button>
          <button
            type="button"
            onClick={onSubmit}
            aria-label="Gửi câu hỏi"
            className="inline-flex h-12 items-center gap-2 rounded-2xl border border-viqa-cyan/30 bg-gradient-to-r from-viqa-cyan/20 to-viqa-violet/20 px-4 font-medium text-viqa-cyan transition hover:shadow-[0_0_28px_rgba(88,230,255,0.18)]"
          >
            <Send className="h-4 w-4" />
            <span className="hidden md:inline">Gửi</span>
          </button>
        </div>

        <div className="flex items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-3">
            <span className="tracking-[0.24em] text-slate-500">SPEECH</span>
            <VoiceWaveform amplitude={audioActive ? 0.7 : 0.15} active={listening || audioActive} />
          </div>
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${speechSupported ? 'bg-emerald-300' : 'bg-amber-300'}`} />
            {speechSupported ? 'Speech API ready' : 'Speech API fallback text only'}
          </div>
        </div>
      </div>
    </motion.section>
  );
}
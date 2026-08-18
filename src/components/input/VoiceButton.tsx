import { Mic, MicOff } from 'lucide-react';

interface VoiceButtonProps {
  listening: boolean;
  supported: boolean;
  onToggle: () => void;
}

export function VoiceButton({ listening, supported, onToggle }: VoiceButtonProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={listening ? 'Dừng nhận diện giọng nói' : 'Bật nhận diện giọng nói'}
      disabled={!supported}
      className={`inline-flex h-10 w-10 items-center justify-center rounded-xl border transition ${
        listening
          ? 'border-viqa-cyan/40 bg-viqa-cyan/15 text-viqa-cyan shadow-[0_0_18px_rgba(56,189,248,0.14)]'
          : 'border-slate-400/15 bg-slate-700/45 text-slate-200 hover:border-viqa-cyan/30 hover:text-viqa-cyan'
      } disabled:cursor-not-allowed disabled:opacity-45`}
    >
      {listening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
    </button>
  );
}

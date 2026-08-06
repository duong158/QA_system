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
      className={`inline-flex h-12 w-12 items-center justify-center rounded-2xl border transition ${
        listening
          ? 'border-viqa-cyan/40 bg-viqa-cyan/15 text-viqa-cyan shadow-[0_0_24px_rgba(88,230,255,0.18)]'
          : 'border-white/10 bg-white/5 text-slate-200 hover:border-viqa-cyan/30 hover:text-viqa-cyan'
      } disabled:cursor-not-allowed disabled:opacity-45`}
    >
      {listening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
    </button>
  );
}
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
      className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border transition ${
        listening
          ? 'border-indigo-200 bg-indigo-50 text-indigo-600'
          : 'border-[var(--border)] bg-[var(--surface)] text-[var(--text-secondary)] hover:border-[var(--border-strong)] hover:bg-[var(--surface-muted)] hover:text-[var(--text-primary)]'
      } disabled:cursor-not-allowed disabled:opacity-45`}
    >
      {listening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
    </button>
  );
}

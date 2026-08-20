import { GraduationCap } from 'lucide-react';

interface SocraticToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

export function SocraticToggle({ enabled, onChange }: SocraticToggleProps) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-xl border border-slate-400/15 bg-slate-800/45 px-3 py-2.5">
      <div className="flex min-w-0 items-center gap-2.5">
        <GraduationCap className="h-4 w-4 shrink-0 text-viqa-violet" />
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-100">Chế độ Gia sư</p>
          <p className="truncate text-xs text-slate-400">Gợi ý hướng khám phá có nguồn</p>
        </div>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        aria-label="Bật hoặc tắt Chế độ Gia sư"
        onClick={() => onChange(!enabled)}
        className={`relative h-7 w-12 shrink-0 rounded-full border transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-viqa-cyan/70 ${
          enabled
            ? 'border-viqa-violet/50 bg-viqa-violet/35'
            : 'border-slate-400/25 bg-slate-700'
        }`}
      >
        <span
          className={`absolute left-1 top-1 h-[18px] w-[18px] rounded-full bg-white shadow transition-transform ${
            enabled ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
        <span className="sr-only">{enabled ? 'ON' : 'OFF'}</span>
      </button>
    </div>
  );
}

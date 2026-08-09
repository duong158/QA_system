import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AudioLines, Settings2, BarChart3, Bot } from 'lucide-react';
import { useAppStore } from '@/store/appStore';

interface HeaderProps {
  onToggleAudio: () => void;
  onToggleSettings: () => void;
  audioEnabled: boolean;
}

export function Header({ onToggleAudio, onToggleSettings, audioEnabled }: HeaderProps) {
  const location = useLocation();
  const status = useAppStore((state) => state.statusMessage);

  return (
    <header className="relative z-20 flex items-center justify-between gap-3 rounded-2xl border border-viqa-border bg-slate-800/75 px-4 py-3 shadow-glow backdrop-blur-xl lg:px-5">
      <div className="flex items-center gap-4">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-viqa-cyan/25 bg-viqa-cyan/10">
          <Bot className="h-6 w-6 text-viqa-cyan" />
        </div>
        <div>
          <p className="font-display text-base font-semibold text-slate-100">VIQA Nexus</p>
          <p className="hidden text-xs text-slate-400 sm:block">Vietnamese Intelligent Question Answering</p>
        </div>
      </div>

      <div className="hidden items-center gap-3 md:flex">
        <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-200">
          <motion.span
            className="h-2 w-2 rounded-full bg-emerald-300"
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
          />
          SYSTEM ONLINE
        </div>
        <div className="rounded-full border border-slate-400/15 bg-slate-700/45 px-3 py-1.5 text-xs text-slate-300">{status}</div>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggleAudio}
          aria-label={audioEnabled ? 'Tắt âm thanh' : 'Bật âm thanh'}
          className={`inline-flex h-11 w-11 items-center justify-center rounded-xl border transition ${
            audioEnabled
              ? 'border-viqa-cyan/30 bg-viqa-cyan/15 text-viqa-cyan'
              : 'border-slate-400/15 bg-slate-700/45 text-slate-300 hover:border-slate-300/25'
          }`}
        >
          <AudioLines className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={onToggleSettings}
          aria-label="Mở cài đặt"
          className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-slate-400/15 bg-slate-700/45 text-slate-200 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
        >
          <Settings2 className="h-5 w-5" />
        </button>
        <Link
          to={location.pathname === '/evaluation' ? '/' : '/evaluation'}
          aria-label={location.pathname === '/evaluation' ? 'Về trang hỏi đáp' : 'Mở trang evaluation'}
          className="inline-flex h-11 items-center gap-2 rounded-xl border border-viqa-gold/25 bg-viqa-gold/10 px-3 text-sm text-viqa-gold transition hover:bg-viqa-gold/15"
        >
          <BarChart3 className="h-4 w-4" />
          <span className="hidden sm:inline">Evaluation</span>
        </Link>
      </div>
    </header>
  );
}

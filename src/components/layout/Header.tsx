import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Activity, AudioLines, Settings2, BarChart3, Bot } from 'lucide-react';
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
    <header className="relative z-20 flex items-center justify-between gap-4 rounded-[28px] border border-white/10 bg-white/5 px-5 py-4 shadow-glow backdrop-blur-xl lg:px-6">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-viqa-cyan/30 bg-viqa-cyan/10 shadow-[0_0_40px_rgba(88,230,255,0.18)]">
          <Bot className="h-6 w-6 text-viqa-cyan" />
        </div>
        <div>
          <p className="font-display text-sm tracking-[0.4em] text-slate-100">VIQA NEXUS</p>
          <p className="text-xs text-slate-400">Vietnamese Intelligent Question Answering</p>
        </div>
      </div>

      <div className="hidden items-center gap-3 md:flex">
        <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs tracking-[0.22em] text-emerald-200">
          <motion.span
            className="h-2 w-2 rounded-full bg-emerald-300"
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
          />
          SYSTEM ONLINE
        </div>
        <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300">{status}</div>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggleAudio}
          aria-label={audioEnabled ? 'Tắt âm thanh' : 'Bật âm thanh'}
          className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl border transition ${
            audioEnabled
              ? 'border-viqa-cyan/30 bg-viqa-cyan/15 text-viqa-cyan'
              : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20'
          }`}
        >
          <AudioLines className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={onToggleSettings}
          aria-label="Mở cài đặt"
          className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-200 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
        >
          <Settings2 className="h-5 w-5" />
        </button>
        <Link
          to={location.pathname === '/evaluation' ? '/' : '/evaluation'}
          aria-label={location.pathname === '/evaluation' ? 'Về trang hỏi đáp' : 'Mở trang evaluation'}
          className="inline-flex h-11 items-center gap-2 rounded-2xl border border-viqa-gold/25 bg-viqa-gold/10 px-4 text-sm text-viqa-gold transition hover:bg-viqa-gold/15"
        >
          <BarChart3 className="h-4 w-4" />
          <span className="hidden sm:inline">Evaluation</span>
        </Link>
      </div>
    </header>
  );
}
import { AudioLines, Bot, History, Settings2, Smile } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { SystemStatus } from './SystemStatus';

interface HeaderProps {
  onToggleAudio: () => void;
  onToggleSettings: () => void;
  onToggleHistory: () => void;
  audioEnabled: boolean;
  onToggleMari?: () => void;
}

export function Header({
  onToggleAudio,
  onToggleSettings,
  onToggleHistory,
  audioEnabled,
  onToggleMari,
}: HeaderProps) {
  const location = useLocation();
  const isChat = location.pathname === '/';

  return (
    <header className="surface-card relative z-30 flex h-[60px] shrink-0 items-center justify-between gap-3 rounded-2xl px-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={onToggleHistory}
          aria-label="Mở lịch sử hội thoại"
          className="soft-icon-button h-10 w-10 lg:hidden"
        >
          <History className="h-5 w-5" />
        </button>
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--primary-soft)] text-[var(--primary)]">
          <Bot className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[var(--text-primary)] sm:text-base">VIQA Tutor</p>
          <p className="hidden truncate text-xs text-[var(--text-secondary)] sm:block">Học cùng Mari từ tài liệu của bạn</p>
        </div>
      </div>

      <div className="hidden items-center md:flex">
        <SystemStatus />
      </div>

      <div className="flex shrink-0 items-center gap-2">
        {isChat && onToggleMari ? (
          <button
            type="button"
            onClick={onToggleMari}
            aria-label="Hiện Mari 3D"
            className="soft-icon-button h-10 w-10 text-indigo-600 min-[1200px]:hidden"
          >
            <Smile className="h-5 w-5" />
          </button>
        ) : null}
        <button
          type="button"
          onClick={onToggleAudio}
          aria-label={audioEnabled ? 'Tắt đọc câu trả lời' : 'Bật đọc câu trả lời'}
          aria-pressed={audioEnabled}
          className={`soft-icon-button h-10 w-10 ${audioEnabled ? '!border-indigo-200 !bg-indigo-50 !text-indigo-600' : ''}`}
        >
          <AudioLines className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={onToggleSettings}
          aria-label="Mở cài đặt"
          className="soft-icon-button h-10 w-10"
        >
          <Settings2 className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}

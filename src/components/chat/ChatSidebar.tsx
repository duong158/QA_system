import { BarChart3, BrainCircuit, MessageSquarePlus, Trash2, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { ChatSession } from '@/types/chat';

interface ChatSidebarProps {
  open: boolean;
  items: ChatSession[];
  onClose: () => void;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onClear: () => void;
}

function groupLabel(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startYesterday = startToday - 86_400_000;
  if (date.getTime() >= startToday) return 'Hôm nay';
  if (date.getTime() >= startYesterday) return 'Hôm qua';
  return 'Trước đó';
}

function SidebarContent({ items, onClose, onNewChat, onSelectSession, onClear }: Omit<ChatSidebarProps, 'open'>) {
  let previousGroup = '';
  return (
    <>
      <div className="flex items-center gap-2 p-3">
        <button
          type="button"
          onClick={onNewChat}
          className="flex h-10 flex-1 items-center justify-center gap-2 rounded-xl bg-[var(--primary)] px-3 text-sm font-semibold text-white transition hover:bg-[var(--primary-hover)]"
        >
          <MessageSquarePlus className="h-4 w-4" />
          Cuộc trò chuyện mới
        </button>
        <button type="button" onClick={onClose} aria-label="Đóng lịch sử" className="soft-icon-button h-10 w-10 lg:hidden">
          <X className="h-5 w-5" />
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3" aria-label="Lịch sử hội thoại">
        {items.length ? items.map((item) => {
          const group = groupLabel(item.createdAt);
          const showGroup = group !== previousGroup;
          previousGroup = group;
          return (
            <div key={item.id}>
              {showGroup ? <p className="px-2 pb-1 pt-4 text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">{group}</p> : null}
              <button
                type="button"
                onClick={() => onSelectSession(item.id)}
                className="group w-full rounded-xl px-2.5 py-2.5 text-left transition hover:bg-[var(--surface-muted)]"
              >
                <span className="line-clamp-2 text-sm leading-5 text-[var(--text-primary)]">{item.title}</span>
                <span className="mt-1 block text-[11px] text-[var(--text-muted)]">
                  {new Date(item.createdAt).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}
                </span>
              </button>
            </div>
          );
        }) : (
          <p className="px-3 py-8 text-center text-sm leading-6 text-[var(--text-secondary)]">
            Các câu hỏi gần đây sẽ xuất hiện ở đây.
          </p>
        )}
      </nav>

      <div className="border-t border-[var(--border)] p-3">
        <div className="grid gap-1">
          <Link to="/knowledge-blind-spots" className="flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm text-[var(--text-secondary)] hover:bg-[var(--surface-muted)] hover:text-[var(--text-primary)]">
            <BrainCircuit className="h-4 w-4" /> Điểm mù tri thức
          </Link>
          <Link to="/evaluation" className="flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm text-[var(--text-secondary)] hover:bg-[var(--surface-muted)] hover:text-[var(--text-primary)]">
            <BarChart3 className="h-4 w-4" /> Đánh giá hệ thống
          </Link>
          {items.length ? (
            <button type="button" onClick={onClear} className="flex items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm text-[var(--text-secondary)] hover:bg-rose-50 hover:text-rose-600">
              <Trash2 className="h-4 w-4" /> Xóa lịch sử
            </button>
          ) : null}
        </div>
      </div>
    </>
  );
}

export function ChatSidebar(props: ChatSidebarProps) {
  const contentProps = { ...props };
  return (
    <>
      <aside className="surface-card hidden min-h-0 w-[248px] shrink-0 flex-col overflow-hidden rounded-2xl lg:flex">
        <SidebarContent {...contentProps} />
      </aside>

      {props.open ? (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button type="button" className="absolute inset-0 bg-slate-950/35 backdrop-blur-[2px]" onClick={props.onClose} aria-label="Đóng lịch sử" />
          <aside className="surface-card relative flex h-full w-[min(86vw,300px)] flex-col rounded-none border-y-0 border-l-0 shadow-2xl">
            <SidebarContent {...contentProps} />
          </aside>
        </div>
      ) : null}
    </>
  );
}

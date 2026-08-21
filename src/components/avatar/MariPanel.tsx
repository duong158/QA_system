import { memo } from 'react';
import { ChevronLeft, ChevronRight, GraduationCap, X } from 'lucide-react';
import type { AvatarState } from '@/types/avatar';
import { AvatarScene } from './AvatarScene';

interface MariPanelProps {
  state: AvatarState;
  collapsed: boolean;
  onCollapsedChange: (collapsed: boolean) => void;
}

interface MariSheetProps {
  open: boolean;
  state: AvatarState;
  onClose: () => void;
}

function stateLabel(state: AvatarState): string {
  switch (state) {
    case 'listening': return 'Đang lắng nghe';
    case 'typing': return 'Bạn đang soạn câu hỏi';
    case 'retrieving': return 'Đang tìm trong tài liệu';
    case 'reading': return 'Đang đọc nguồn';
    case 'thinking': return 'Đang suy nghĩ';
    case 'speaking': return 'Đang trò chuyện';
    case 'success': return 'Đã tìm thấy câu trả lời';
    case 'no-answer': return 'Chưa đủ bằng chứng';
    case 'error': return 'Có lỗi kết nối';
    default: return 'Đang sẵn sàng';
  }
}

function MariContent({ state }: Pick<MariPanelProps, 'state'>) {
  return (
    <div className="surface-card flex h-full min-h-0 flex-col overflow-hidden rounded-2xl p-3">
      <div className="mb-3 px-1">
        <p className="text-sm font-semibold text-[var(--text-primary)]">Mari</p>
        <p className="text-xs text-[var(--text-secondary)]">Trợ lý học tập VIQA</p>
      </div>
      <AvatarScene state={state} compact />
      <div className="mt-3 grid gap-2 rounded-xl bg-[var(--surface-subtle)] p-3 text-xs text-[var(--text-secondary)]">
        <p className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${state === 'error' ? 'bg-rose-500' : 'bg-emerald-500'}`} />
          {stateLabel(state)}
        </p>
        <p className="flex items-center gap-2">
          <GraduationCap className="h-4 w-4 text-violet-600" />
          Gia sư luôn bật
        </p>
      </div>
    </div>
  );
}

export const MariPanel = memo(function MariPanel({ state, collapsed, onCollapsedChange }: MariPanelProps) {
  return (
    <aside
      className={collapsed
        ? 'absolute bottom-0 right-0 top-0 z-20 hidden w-12 min-[1200px]:block'
        : 'relative hidden min-h-0 min-[1200px]:block'}
      aria-label="Mari 3D assistant"
    >
      <div className={`absolute right-0 top-0 h-full w-[272px] transition-opacity ${collapsed ? 'pointer-events-none invisible opacity-0' : 'visible opacity-100'}`}>
        <MariContent state={state} />
      </div>
      <button
        type="button"
        onClick={() => onCollapsedChange(!collapsed)}
        aria-label={collapsed ? 'Hiện Mari' : 'Thu gọn Mari'}
        className={`soft-icon-button absolute z-30 h-9 w-9 ${collapsed ? 'right-1 top-3' : 'right-3 top-3'}`}
      >
        {collapsed ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </button>
    </aside>
  );
});

export const MariSheet = memo(function MariSheet({ open, state, onClose }: MariSheetProps) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 min-[1200px]:hidden" role="dialog" aria-modal="true" aria-label="Mari 3D assistant">
      <button type="button" onClick={onClose} className="absolute inset-0 bg-slate-950/35 backdrop-blur-[2px]" aria-label="Đóng Mari" />
      <section className="absolute inset-x-3 bottom-3 max-h-[calc(100dvh-24px)] overflow-y-auto rounded-2xl shadow-2xl sm:left-auto sm:w-[360px]">
        <button type="button" onClick={onClose} aria-label="Đóng Mari" className="soft-icon-button absolute right-5 top-5 z-20 h-9 w-9">
          <X className="h-4 w-4" />
        </button>
        <MariContent state={state} />
      </section>
    </div>
  );
});

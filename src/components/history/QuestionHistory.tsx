import { History, RotateCcw, Trash2 } from 'lucide-react';
import type { QaHistoryItem } from '@/store/appStore';

interface QuestionHistoryProps {
  items: QaHistoryItem[];
  onReuse: (question: string) => void;
  onClear: () => void;
}

function formatHistoryTime(timestamp: number) {
  return new Intl.DateTimeFormat('vi-VN', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(timestamp));
}

export function QuestionHistory({ items, onReuse, onClear }: QuestionHistoryProps) {
  return (
    <section className="viqa-panel p-4 lg:p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-viqa-violet/10 text-viqa-violet">
            <History className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-display text-base font-semibold text-slate-50">Lịch sử câu hỏi</h2>
            <p className="text-xs text-slate-400">{items.length ? `${items.length} lượt gần nhất` : 'Chưa có câu hỏi nào'}</p>
          </div>
        </div>

        <button
          type="button"
          onClick={onClear}
          disabled={!items.length}
          aria-label="Xóa lịch sử"
          title="Xóa lịch sử"
          className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-slate-400/15 bg-slate-700/45 text-slate-300 transition hover:border-viqa-error/35 hover:text-viqa-error disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-4 max-h-72 overflow-y-auto pr-1">
        {items.length ? (
          <div className="divide-y divide-slate-400/10">
            {items.map((item) => (
              <div key={item.id} className="grid gap-2 py-3 first:pt-0 last:pb-0">
                <div className="flex items-start justify-between gap-3">
                  <button
                    type="button"
                    onClick={() => onReuse(item.question)}
                    className="min-w-0 flex-1 text-left text-sm font-medium leading-6 text-slate-100 transition hover:text-viqa-cyan"
                    title="Dùng lại câu hỏi"
                  >
                    {item.question}
                  </button>
                  <button
                    type="button"
                    onClick={() => onReuse(item.question)}
                    aria-label="Dùng lại câu hỏi"
                    title="Dùng lại câu hỏi"
                    className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-400/15 bg-slate-800/70 text-slate-300 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                  <span>{formatHistoryTime(item.createdAt)}</span>
                  <span className={item.hasAnswer ? 'text-emerald-300' : 'text-amber-300'}>
                    {item.hasAnswer ? 'Có đáp án' : 'Không đủ tin cậy'}
                  </span>
                  <span>
                    {typeof item.rankingScore === 'number'
                      ? `Ranking score ${item.rankingScore.toFixed(3)}`
                      : 'Ranking score --'}
                  </span>
                  <span>
                    {typeof item.answerConfidence === 'number'
                      ? `Answer confidence ${(item.answerConfidence * 100).toFixed(1)}%`
                      : 'Answer confidence chưa hiệu chuẩn'}
                  </span>
                </div>

                {item.answer ? <p className="line-clamp-2 text-xs leading-5 text-slate-400">{item.answer}</p> : null}
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-slate-400/15 bg-slate-800/35 px-4 py-6 text-center text-sm text-slate-400">
            Các câu đã hỏi sẽ xuất hiện ở đây.
          </div>
        )}
      </div>
    </section>
  );
}

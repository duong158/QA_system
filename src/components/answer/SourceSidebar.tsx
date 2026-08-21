import { AnimatePresence, motion } from 'framer-motion';
import { Database, X, ChevronDown } from 'lucide-react';
import type { QaResponse, PassageResult } from '@/types/qa';
import { useState } from 'react';
import { highlightAnswer } from '@/utils/highlightAnswer';

interface SourceSidebarProps {
  open: boolean;
  onClose: () => void;
  response: QaResponse | null;
}

function SourceItem({ item, isSelected, answer }: { item: PassageResult; isSelected: boolean; answer: string | null }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`rounded-xl border p-4 ${isSelected ? 'border-viqa-cyan/30 bg-viqa-cyan/5' : 'border-[var(--border)] bg-[var(--surface-subtle)]'}`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <span className="block text-[10px] font-semibold uppercase tracking-wide text-viqa-cyan">
            {isSelected ? 'Nguồn chính' : 'Nguồn liên quan'}
          </span>
          <span className="mt-1 block text-sm font-medium text-[var(--text-primary)]">
            {item.title}{item.page ? ` · Trang ${item.page}` : ''}
          </span>
        </div>
      </div>
      
      <button 
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="mt-3 flex items-center gap-1.5 text-xs text-viqa-cyan hover:text-viqa-cyan/80 transition"
      >
        Xem thêm
        <ChevronDown className={`h-3.5 w-3.5 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      {expanded ? (
        <div className="mt-3 space-y-3 border-t border-[var(--border)] pt-3 text-sm">
          <p className="text-[var(--text-secondary)] leading-6 whitespace-pre-wrap">
            {isSelected ? highlightAnswer(item.text, answer ?? undefined) : item.text}
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs text-[var(--text-muted)] mt-2 border-t border-[var(--border)] pt-3">
            <div><span className="font-medium text-[var(--text-secondary)]">Trạng thái:</span> <br/>
              <span className={item.selection_status === 'SELECTED' ? 'text-viqa-cyan font-medium' : 'text-rose-500 font-medium'}>
                {item.selection_status === 'SELECTED' ? 'Đã chọn' : 'Bị từ chối'}
              </span>
            </div>
            {item.selection_status === 'REJECTED' && item.rejection_detail && (
              <div className="col-span-2 text-rose-500 mt-1 mb-1">
                <span className="font-medium text-[var(--text-secondary)]">Lý do:</span> {item.rejection_detail}
              </div>
            )}
            <div><span className="font-medium text-[var(--text-secondary)]">Document ID:</span> <br/>{item.document_id}</div>
            <div><span className="font-medium text-[var(--text-secondary)]">Passage ID:</span> <br/>{item.passage_id}</div>
            {item.ranking_score !== undefined && (
              <div><span className="font-medium text-[var(--text-secondary)]">Ranking Score:</span> <br/>{item.ranking_score?.toFixed(4)}</div>
            )}
            {item.reader_score !== undefined && (
              <div><span className="font-medium text-[var(--text-secondary)]">Reader Score:</span> <br/>{item.reader_score?.toFixed(4)}</div>
            )}
            {item.retrieval_score_normalized !== undefined && (
              <div><span className="font-medium text-[var(--text-secondary)]">Retrieval Score (Norm):</span> <br/>{item.retrieval_score_normalized?.toFixed(4)}</div>
            )}
            {item.retriever_score !== undefined && (
              <div><span className="font-medium text-[var(--text-secondary)]">Retriever Score:</span> <br/>{item.retriever_score?.toFixed(4)}</div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function SourceSidebar({ open, onClose, response }: SourceSidebarProps) {
  if (!response) return null;

  const passages = response.passages || [];
  const selectedId = response.selected_passage_id ?? response.top_retrieved_passage?.passage_id ?? passages[0]?.passage_id;
  const selectedPassage = passages.find(p => p.passage_id === selectedId);
  const otherPassages = passages.filter(p => p.passage_id !== selectedId);
  const answerText = response.answer || response.answer_span?.text || null;

  return (
    <AnimatePresence>
      {open ? (
        <motion.aside
          initial={{ x: 480, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 480, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 120, damping: 20 }}
          className="fixed right-0 top-0 z-50 h-full w-full max-w-[480px] border-l border-[var(--border)] bg-[var(--surface)]/95 p-4 shadow-2xl backdrop-blur-xl"
        >
          <div className="flex h-full flex-col overflow-hidden rounded-[24px] border border-[var(--border)] bg-[var(--surface)] p-4 lg:p-5 shadow-sm">
            <div className="flex shrink-0 items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-viqa-cyan/10 text-viqa-cyan">
                  <Database className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="font-display text-base font-semibold text-[var(--text-primary)]">Nguồn tham khảo</h2>
                  <p className="text-xs text-[var(--text-muted)]">{passages.length} nguồn được tìm thấy</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button 
                  type="button" 
                  onClick={onClose} 
                  className="rounded-full border border-[var(--border)] bg-[var(--surface-subtle)] p-2 text-[var(--text-secondary)] transition hover:bg-[var(--surface-muted)] hover:text-[var(--text-primary)]"
                  aria-label="Đóng"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="mt-5 min-h-0 flex-1 overflow-y-auto pr-1 space-y-4 chat-scroll">
              {selectedPassage ? (
                <SourceItem item={selectedPassage} isSelected={true} answer={answerText} />
              ) : (
                <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface-muted)] px-4 py-6 text-center text-sm text-[var(--text-secondary)]">
                  Không có nguồn trích dẫn.
                </div>
              )}
              
              {otherPassages.map(passage => (
                <SourceItem key={passage.passage_id} item={passage} isSelected={false} answer={answerText} />
              ))}
            </div>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
}

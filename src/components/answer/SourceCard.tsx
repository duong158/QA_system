import { MapPinned, FileText } from 'lucide-react';
import type { SourceInfo } from '@/types/qa';

interface SourceCardProps {
  source: SourceInfo;
  onScrollToPassage?: (passageId: string) => void;
}

export function SourceCard({ source, onScrollToPassage }: SourceCardProps) {
  return (
    <div className="rounded-lg border border-slate-400/15 bg-[#172033] p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
        <FileText className="h-4 w-4 text-viqa-gold" />
        Source
      </div>
      <div className="mt-3 text-sm text-slate-200">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <button 
            type="button"
            onClick={() => onScrollToPassage?.(source.passage_id)}
            className="group flex items-center gap-2 transition"
            title="Cuộn tới văn bản này trong danh sách nguồn"
          >
            <span className="inline-flex items-center gap-1 rounded-full border border-slate-400/15 bg-slate-700/40 px-2.5 py-1 transition group-hover:border-viqa-cyan/30 group-hover:bg-viqa-cyan/10 group-hover:text-viqa-cyan">
              <MapPinned className="h-3.5 w-3.5" />
              {source.document_id}
            </span>
            <span className="inline-flex items-center gap-1 rounded-full border border-slate-400/15 bg-slate-700/40 px-2.5 py-1 transition group-hover:border-viqa-cyan/30 group-hover:bg-viqa-cyan/10 group-hover:text-viqa-cyan">
              Passage {source.passage_id}
            </span>
          </button>
          {source.page ? <span className="rounded-full border border-slate-400/15 bg-slate-700/40 px-2.5 py-1">Page {source.page}</span> : null}
        </div>
      </div>
    </div>
  );
}

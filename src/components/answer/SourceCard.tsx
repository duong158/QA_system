import { MapPinned, FileText } from 'lucide-react';
import type { SourceInfo } from '@/types/qa';

interface SourceCardProps {
  source: SourceInfo;
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <div className="rounded-lg border border-slate-400/15 bg-[#172033] p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
        <FileText className="h-4 w-4 text-viqa-gold" />
        Source
      </div>
      <div className="mt-3 space-y-2 text-sm text-slate-200">
        <p className="font-medium text-white">{source.title}</p>
        <div className="flex flex-wrap gap-2 text-xs text-slate-400">
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-400/15 bg-slate-700/40 px-2.5 py-1">
            <MapPinned className="h-3.5 w-3.5" />
            {source.document_id}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-400/15 bg-slate-700/40 px-2.5 py-1">Passage {source.passage_id}</span>
          {source.page ? <span className="rounded-full border border-slate-400/15 bg-slate-700/40 px-2.5 py-1">Page {source.page}</span> : null}
        </div>
      </div>
    </div>
  );
}

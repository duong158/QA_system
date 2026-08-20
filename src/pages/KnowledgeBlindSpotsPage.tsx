import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  AlertTriangle,
  ArrowLeft,
  BrainCircuit,
  Check,
  DatabaseZap,
  FilePlus2,
  Loader2,
  RefreshCw,
  ShieldCheck,
  X,
} from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MainLayout } from '@/components/layout/MainLayout';
import { SettingsPanel } from '@/components/settings/SettingsPanel';
import {
  fetchDocumentSubmissions,
  fetchFeedbackAnalytics,
  fetchPendingFeedback,
  reviewDocument,
  reviewFeedback,
  submitDocument,
} from '@/services/feedbackService';
import type { DocumentSubmission, FeedbackAnalytics, FeedbackRecord } from '@/types/feedback';
import { useAppStore } from '@/store/appStore';
import { useSpeechSynthesis } from '@/hooks/useSpeechSynthesis';

const chartTooltipStyle = {
  background: 'rgba(4, 7, 17, 0.97)',
  border: '1px solid rgba(90, 220, 255, 0.24)',
  borderRadius: 12,
  color: '#F8FAFC',
};

const pieColors = ['#38BDF8', '#818CF8', '#FBBF24', '#FB7185'];

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function heatColor(rate: number, total: number): string {
  if (!total) return 'rgba(51, 65, 85, 0.35)';
  if (rate >= 0.75) return 'rgba(251, 113, 133, 0.52)';
  if (rate >= 0.5) return 'rgba(245, 158, 11, 0.42)';
  if (rate >= 0.25) return 'rgba(251, 191, 36, 0.24)';
  return 'rgba(52, 211, 153, 0.18)';
}

export function KnowledgeBlindSpotsPage() {
  const settings = useAppStore((state) => state.settings);
  const toggleSettings = useAppStore((state) => state.toggleSettings);
  const toggleHistory = useAppStore((state) => state.toggleHistory);
  const updateVoiceSettings = useAppStore((state) => state.updateVoiceSettings);
  const synthesis = useSpeechSynthesis();

  const [analytics, setAnalytics] = useState<FeedbackAnalytics | null>(null);
  const [feedback, setFeedback] = useState<FeedbackRecord[]>([]);
  const [documents, setDocuments] = useState<DocumentSubmission[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [sourceView, setSourceView] = useState<FeedbackRecord | null>(null);
  const [documentTitle, setDocumentTitle] = useState('');
  const [documentContent, setDocumentContent] = useState('');
  const [documentMessage, setDocumentMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextAnalytics, nextFeedback, nextDocuments] = await Promise.all([
        fetchFeedbackAnalytics(),
        fetchPendingFeedback(),
        fetchDocumentSubmissions(),
      ]);
      setAnalytics(nextAnalytics);
      setFeedback(nextFeedback);
      setDocuments(nextDocuments);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể tải dashboard.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const review = async (item: FeedbackRecord, decision: 'APPROVED' | 'REJECTED') => {
    setBusyId(item.feedback_id);
    try {
      await reviewFeedback(item.feedback_id, decision);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể cập nhật review.');
    } finally {
      setBusyId(null);
    }
  };

  const reviewSubmission = async (item: DocumentSubmission, decision: 'APPROVED' | 'REJECTED') => {
    setBusyId(item.submission_id);
    try {
      await reviewDocument(item.submission_id, decision);
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể cập nhật tài liệu.');
    } finally {
      setBusyId(null);
    }
  };

  const contributeDocument = async () => {
    if (!documentTitle.trim() || documentContent.trim().length < 20) return;
    setBusyId('document-form');
    setDocumentMessage(null);
    try {
      const result = await submitDocument({
        title: documentTitle.trim(),
        content: documentContent.trim(),
        source_type: 'PLAIN_TEXT',
      });
      setDocumentMessage(result.message);
      setDocumentTitle('');
      setDocumentContent('');
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể gửi tài liệu.');
    } finally {
      setBusyId(null);
    }
  };

  const heatmapLookup = useMemo(() => {
    const lookup = new Map<string, FeedbackAnalytics['heatmap']['cells'][number]>();
    analytics?.heatmap.cells.forEach((cell) => lookup.set(`${cell.question_type}\u241f${cell.relation}`, cell));
    return lookup;
  }, [analytics]);

  const pendingDocuments = documents.filter((item) => item.status === 'PENDING_REVIEW');

  const testVoice = () => synthesis.speak({
    text: 'Dashboard điểm mù tri thức và hàng chờ phản hồi của con người.',
    voiceName: settings.voice.voiceName,
    rate: settings.voice.rate,
    pitch: settings.voice.pitch,
    volume: settings.voice.volume,
  });

  return (
    <MainLayout>
      <Header
        audioEnabled={settings.voice.enabled}
        onToggleAudio={() => updateVoiceSettings({ enabled: !settings.voice.enabled })}
        onToggleSettings={toggleSettings}
        onToggleHistory={toggleHistory}
      />

      <main className="admin-page mt-4 flex-1 overflow-y-auto pr-1">
        <div className="grid gap-4 pb-6">
          <section className="viqa-panel rounded-[28px] p-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className="rounded-2xl border border-viqa-violet/25 bg-viqa-violet/10 p-3 text-indigo-200"><BrainCircuit className="h-6 w-6" /></div>
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Human-in-the-loop Active Learning V1</p>
                  <h1 className="mt-2 font-display text-2xl text-white">Bản đồ Điểm mù Tri thức</h1>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                    Phản hồi được phân tích và đưa qua kiểm duyệt. Model weights và production corpus không tự cập nhật tại runtime.
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <button type="button" onClick={() => void load()} className="inline-flex items-center gap-2 rounded-xl border border-slate-400/15 bg-slate-700/40 px-3 py-2 text-sm text-slate-200"><RefreshCw className="h-4 w-4" /> Làm mới</button>
                <Link to="/" className="inline-flex items-center gap-2 rounded-xl border border-slate-400/15 bg-slate-700/40 px-3 py-2 text-sm text-slate-200"><ArrowLeft className="h-4 w-4" /> Hỏi đáp</Link>
              </div>
            </div>
          </section>

          {loading && !analytics ? (
            <div className="flex items-center justify-center gap-3 py-24 text-slate-400"><Loader2 className="h-6 w-6 animate-spin text-viqa-cyan" /> Đang tải dữ liệu phản hồi...</div>
          ) : error && !analytics ? (
            <div className="rounded-2xl border border-rose-400/25 bg-rose-400/10 p-5 text-rose-200">{error}</div>
          ) : analytics ? (
            <>
              <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                {[
                  ['Total feedback', analytics.summary.total_feedback, 'count'],
                  ['Correct rate', analytics.summary.correct_rate, 'rate'],
                  ['Incorrect rate', analytics.summary.incorrect_rate, 'rate'],
                  ['No-answer complaints', analytics.summary.no_answer_complaint_rate, 'rate'],
                  ['Pending review', analytics.summary.pending_review, 'count'],
                ].map(([label, value, kind]) => (
                  <article key={String(label)} className="viqa-panel rounded-2xl p-4">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
                    <p className="mt-3 font-display text-3xl text-white">{kind === 'rate' ? percent(Number(value)) : Number(value).toLocaleString()}</p>
                  </article>
                ))}
              </section>

              <section className="grid gap-4 xl:grid-cols-2">
                <article className="viqa-panel rounded-[24px] p-5">
                  <h2 className="font-display text-lg text-white">Failure by relation</h2>
                  <div className="mt-4 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.relations.slice(0, 10)} layout="vertical" margin={{ left: 24 }}>
                        <CartesianGrid stroke="rgba(148,163,184,.1)" horizontal={false} />
                        <XAxis type="number" domain={[0, 1]} tickFormatter={percent} stroke="#94A3B8" />
                        <YAxis type="category" dataKey="semantic_relation" width={110} stroke="#94A3B8" />
                        <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => percent(Number(value))} />
                        <Bar dataKey="failure_rate" name="Failure rate" fill="#FB7185" radius={[0, 6, 6, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </article>

                <article className="viqa-panel rounded-[24px] p-5">
                  <h2 className="font-display text-lg text-white">Failure by question type</h2>
                  <div className="mt-4 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.question_types}>
                        <CartesianGrid stroke="rgba(148,163,184,.1)" vertical={false} />
                        <XAxis dataKey="question_type" stroke="#94A3B8" />
                        <YAxis domain={[0, 1]} tickFormatter={percent} stroke="#94A3B8" />
                        <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => percent(Number(value))} />
                        <Bar dataKey="failure_rate" name="Failure rate" fill="#818CF8" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </article>

                <article className="viqa-panel rounded-[24px] p-5">
                  <h2 className="font-display text-lg text-white">Gap classification</h2>
                  <div className="mt-4 h-72">
                    {analytics.gap_types.length ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={analytics.gap_types} dataKey="count" nameKey="gap_type" innerRadius={58} outerRadius={94} paddingAngle={3}>
                            {analytics.gap_types.map((item, index) => <Cell key={item.gap_type} fill={pieColors[index % pieColors.length]} />)}
                          </Pie>
                          <Tooltip contentStyle={chartTooltipStyle} />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    ) : <p className="flex h-full items-center justify-center text-sm text-slate-500">Chưa có failure để phân loại.</p>}
                  </div>
                </article>

                <article className="viqa-panel rounded-[24px] p-5">
                  <h2 className="font-display text-lg text-white">Top rejection reasons</h2>
                  <div className="mt-4 h-72">
                    {analytics.top_rejection_reasons.length ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={analytics.top_rejection_reasons.slice(0, 8)} layout="vertical" margin={{ left: 30 }}>
                          <CartesianGrid stroke="rgba(148,163,184,.1)" horizontal={false} />
                          <XAxis type="number" allowDecimals={false} stroke="#94A3B8" />
                          <YAxis type="category" dataKey="reason" width={150} stroke="#94A3B8" />
                          <Tooltip contentStyle={chartTooltipStyle} />
                          <Bar dataKey="count" fill="#FBBF24" radius={[0, 6, 6, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : <p className="flex h-full items-center justify-center text-sm text-slate-500">Chưa có rejection reason.</p>}
                  </div>
                </article>
              </section>

              <section className="viqa-panel rounded-[24px] p-5">
                <h2 className="font-display text-lg text-white">Feedback trend</h2>
                <div className="mt-4 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={analytics.trend}>
                      <CartesianGrid stroke="rgba(148,163,184,.1)" vertical={false} />
                      <XAxis dataKey="date" stroke="#94A3B8" />
                      <YAxis allowDecimals={false} stroke="#94A3B8" />
                      <Tooltip contentStyle={chartTooltipStyle} />
                      <Legend />
                      <Line type="monotone" dataKey="correct" stroke="#34D399" strokeWidth={2} />
                      <Line type="monotone" dataKey="incorrect" stroke="#FB7185" strokeWidth={2} />
                      <Line type="monotone" dataKey="no_answer" stroke="#FBBF24" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="viqa-panel rounded-[24px] p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="font-display text-lg text-white">Heatmap: Question Type × Relation</h2>
                    <p className="mt-1 text-xs text-slate-400">Màu thể hiện failure rate; score xếp hạng còn tính cả số lượng mẫu.</p>
                  </div>
                  <span className="text-xs text-slate-500">Synthetic: {analytics.summary.synthetic_feedback} · Real: {analytics.summary.real_feedback}</span>
                </div>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[720px] border-separate border-spacing-1 text-xs">
                    <thead><tr><th className="p-2 text-left text-slate-500">Question type</th>{analytics.heatmap.relations.map((relation) => <th key={relation} className="p-2 text-slate-400">{relation}</th>)}</tr></thead>
                    <tbody>
                      {analytics.heatmap.question_types.map((questionType) => (
                        <tr key={questionType}>
                          <th className="p-2 text-left font-medium text-slate-300">{questionType}</th>
                          {analytics.heatmap.relations.map((relation) => {
                            const cell = heatmapLookup.get(`${questionType}\u241f${relation}`);
                            return (
                              <td key={relation} className="rounded-lg p-3 text-center text-slate-100" style={{ backgroundColor: heatColor(cell?.failure_rate ?? 0, cell?.total ?? 0) }} title={cell ? `score ${cell.blind_spot_score.toFixed(3)} · n=${cell.total}` : 'No samples'}>
                                {cell ? `${percent(cell.failure_rate)} · n=${cell.total}` : '—'}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : null}

          {error && analytics ? <div className="rounded-xl border border-rose-400/20 bg-rose-400/10 p-3 text-sm text-rose-200">{error}</div> : null}

          <section className="viqa-panel rounded-[24px] p-5">
            <div className="flex items-center gap-2"><ShieldCheck className="h-5 w-5 text-viqa-cyan" /><h2 className="font-display text-lg text-white">Pending Feedback</h2><span className="rounded-full bg-slate-700 px-2 py-0.5 text-xs text-slate-300">{feedback.length}</span></div>
            <p className="mt-2 text-xs text-amber-200/75">Local/demo review endpoints chưa có production authorization.</p>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[1050px] text-left text-xs">
                <thead className="border-b border-slate-400/15 uppercase tracking-[0.14em] text-slate-500"><tr><th className="p-3">Question</th><th>Prediction</th><th>Correction</th><th>Relation</th><th>Gap</th><th>Status</th><th>Timestamp</th><th>Actions</th></tr></thead>
                <tbody className="divide-y divide-slate-400/10 text-slate-300">
                  {feedback.map((item) => (
                    <tr key={item.feedback_id} className={item.conflict ? 'bg-rose-400/[0.05]' : ''}>
                      <td className="max-w-[220px] p-3"><p className="line-clamp-2">{item.question}</p>{item.conflict ? <span className="mt-1 inline-flex items-center gap-1 text-rose-300"><AlertTriangle className="h-3 w-3" /> CONFLICT</span> : null}</td>
                      <td className="max-w-[180px]"><p className="line-clamp-2">{item.predicted_answer || 'No answer'}</p></td>
                      <td className="max-w-[180px]"><p className="line-clamp-2 text-emerald-200">{item.corrected_answer || item.user_note || '—'}</p></td>
                      <td>{item.semantic_relation || 'UNKNOWN'}</td><td>{item.gap_type || '—'}</td><td>{item.status}</td><td>{new Date(item.timestamp).toLocaleString('vi-VN')}</td>
                      <td><div className="flex gap-1">
                        {item.source_passage ? <button type="button" onClick={() => setSourceView(item)} className="rounded-lg border border-slate-400/15 px-2 py-1">Source</button> : null}
                        <button type="button" disabled={busyId === item.feedback_id} onClick={() => void review(item, 'APPROVED')} className="rounded-lg border border-emerald-400/20 px-2 py-1 text-emerald-200"><Check className="h-3.5 w-3.5" /></button>
                        <button type="button" disabled={busyId === item.feedback_id} onClick={() => void review(item, 'REJECTED')} className="rounded-lg border border-rose-400/20 px-2 py-1 text-rose-200"><X className="h-3.5 w-3.5" /></button>
                      </div></td>
                    </tr>
                  ))}
                  {!feedback.length ? <tr><td colSpan={8} className="p-8 text-center text-slate-500">Không có feedback đang chờ duyệt.</td></tr> : null}
                </tbody>
              </table>
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
            <article id="document-contribution" className="viqa-panel scroll-mt-4 rounded-[24px] p-5">
              <div className="flex items-center gap-2"><FilePlus2 className="h-5 w-5 text-viqa-gold" /><h2 className="font-display text-lg text-white">Đóng góp tài liệu</h2></div>
              <p className="mt-2 text-xs leading-5 text-slate-400">V1 nhận plain text. Nội dung chỉ vào hàng chờ, không chunk/index production tự động.</p>
              <input value={documentTitle} onChange={(event) => setDocumentTitle(event.target.value)} placeholder="Tiêu đề tài liệu" className="mt-4 w-full rounded-xl border border-slate-400/15 bg-slate-900/70 px-3 py-2 text-sm outline-none focus:border-viqa-cyan/30" />
              <textarea value={documentContent} onChange={(event) => setDocumentContent(event.target.value)} rows={7} placeholder="Nội dung tài liệu..." className="mt-3 w-full rounded-xl border border-slate-400/15 bg-slate-900/70 px-3 py-2 text-sm leading-6 outline-none focus:border-viqa-cyan/30" />
              <button type="button" onClick={() => void contributeDocument()} disabled={!documentTitle.trim() || documentContent.trim().length < 20 || busyId === 'document-form'} className="mt-3 rounded-xl bg-viqa-gold px-4 py-2 text-sm font-semibold text-slate-950 disabled:opacity-40">Gửi vào hàng chờ</button>
              {documentMessage ? <p className="mt-3 text-sm text-emerald-200">{documentMessage}</p> : null}
            </article>

            <article className="viqa-panel rounded-[24px] p-5">
              <div className="flex items-center gap-2"><DatabaseZap className="h-5 w-5 text-viqa-violet" /><h2 className="font-display text-lg text-white">Document Review Queue</h2><span className="rounded-full bg-slate-700 px-2 py-0.5 text-xs">{pendingDocuments.length}</span></div>
              <div className="mt-4 grid gap-3">
                {pendingDocuments.map((item) => (
                  <div key={item.submission_id} className="rounded-xl border border-slate-400/15 bg-slate-900/35 p-4">
                    <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="font-medium text-white">{item.title}</p><p className="mt-1 text-xs text-slate-500">{item.source_type} · {new Date(item.timestamp).toLocaleString('vi-VN')}</p></div><div className="flex gap-2"><button type="button" onClick={() => void reviewSubmission(item, 'APPROVED')} disabled={busyId === item.submission_id} className="rounded-lg border border-emerald-400/20 px-2 py-1 text-xs text-emerald-200">Approve candidate</button><button type="button" onClick={() => void reviewSubmission(item, 'REJECTED')} disabled={busyId === item.submission_id} className="rounded-lg border border-rose-400/20 px-2 py-1 text-xs text-rose-200">Reject</button></div></div>
                    <p className="mt-3 line-clamp-3 text-sm leading-6 text-slate-300">{item.content}</p>
                  </div>
                ))}
                {!pendingDocuments.length ? <p className="py-10 text-center text-sm text-slate-500">Không có tài liệu chờ duyệt.</p> : null}
              </div>
            </article>
          </section>
        </div>
      </main>

      {sourceView?.source_passage ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4" role="dialog" aria-modal="true">
          <section className="viqa-panel max-h-[80vh] w-full max-w-3xl overflow-y-auto p-5">
            <div className="flex items-center justify-between gap-3"><h2 className="font-display text-lg text-white">{sourceView.source_passage.passage_id}</h2><button type="button" onClick={() => setSourceView(null)} aria-label="Đóng source"><X className="h-5 w-5" /></button></div>
            <p className="mt-4 whitespace-pre-wrap rounded-xl border border-slate-400/15 bg-slate-900/60 p-4 text-sm leading-7 text-slate-200">{sourceView.source_passage.text}</p>
          </section>
        </div>
      ) : null}

      <SettingsPanel voices={synthesis.voices} onTestVoice={testVoice} />
    </MainLayout>
  );
}

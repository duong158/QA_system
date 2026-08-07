import { Link } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ArrowLeft, Clock3, FileWarning, Table2 } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { MainLayout } from '@/components/layout/MainLayout';
import { SettingsPanel } from '@/components/settings/SettingsPanel';
import {
  errorAnalysis,
  evaluationMetrics,
  readerComparison,
  recallCurveData,
  retrieverChartData,
} from '@/data/evaluationMockData';
import { useSpeechSynthesis } from '@/hooks/useSpeechSynthesis';
import { useAppStore } from '@/store/appStore';
import { formatLatency, formatScore } from '@/utils/formatScore';

const chartTooltipStyle = {
  background: 'rgba(4, 7, 17, 0.96)',
  border: '1px solid rgba(90, 220, 255, 0.24)',
  borderRadius: 16,
  color: '#F8FAFC',
};

export function EvaluationPage() {
  const settings = useAppStore((state) => state.settings);
  const toggleSettings = useAppStore((state) => state.toggleSettings);
  const updateVoiceSettings = useAppStore((state) => state.updateVoiceSettings);
  const synthesis = useSpeechSynthesis();

  const testVoice = () => {
    synthesis.speak({
      text: 'Xin chào, tôi là Mari. Tôi có thể giúp bạn tìm câu trả lời trong tập tài liệu.',
      voiceName: settings.voice.voiceName,
      rate: settings.voice.rate,
      pitch: settings.voice.pitch,
      volume: settings.voice.volume,
    });
  };

  return (
    <MainLayout>
      <Header
        audioEnabled={settings.voice.enabled}
        onToggleAudio={() => updateVoiceSettings({ enabled: !settings.voice.enabled })}
        onToggleSettings={toggleSettings}
      />

      <main className="mt-4 grid gap-4 pb-4">
        <section className="viqa-panel rounded-[30px] p-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Evaluation Console</p>
              <h1 className="mt-2 font-display text-2xl tracking-[0.16em] text-white">QA SYSTEM METRICS</h1>
            </div>
            <Link
              to="/"
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 transition hover:border-viqa-cyan/25 hover:text-viqa-cyan"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to VIQA
            </Link>
          </div>
        </section>

        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {evaluationMetrics.map((metric) => (
            <article key={metric.label} className="viqa-panel rounded-[24px] p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs uppercase tracking-[0.28em] text-slate-500">{metric.label}</p>
                <Clock3 className="h-4 w-4 text-viqa-cyan" />
              </div>
              <p className="mt-4 font-display text-3xl text-white">
                {metric.label.includes('Response') ? formatLatency(metric.value * 1000) : formatScore(metric.value)}
              </p>
            </article>
          ))}
        </section>

        <section className="grid gap-4 xl:grid-cols-2">
          <article className="viqa-panel rounded-[28px] p-5">
            <h2 className="font-display text-lg tracking-[0.16em] text-white">RETRIEVER COMPARISON</h2>
            <div className="mt-5 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={retrieverChartData}>
                  <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                  <XAxis dataKey="name" stroke="#94A3B8" />
                  <YAxis stroke="#94A3B8" tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatScore(Number(value))} />
                  <Bar dataKey="recall1" name="Recall@1" fill="#58E6FF" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="recall3" name="Recall@3" fill="#9F7AEA" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="recall5" name="Recall@5" fill="#FFD76A" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </article>

          <article className="viqa-panel rounded-[28px] p-5">
            <h2 className="font-display text-lg tracking-[0.16em] text-white">RECALL@K CURVE</h2>
            <div className="mt-5 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={recallCurveData}>
                  <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
                  <XAxis dataKey="k" stroke="#94A3B8" />
                  <YAxis stroke="#94A3B8" tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatScore(Number(value))} />
                  <Line type="monotone" dataKey="tfidf" name="TF-IDF" stroke="#94A3B8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="bm25" name="BM25" stroke="#58E6FF" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="dense" name="Dense" stroke="#FFD76A" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="pyserini" name="Pyserini" stroke="#9F7AEA" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </article>
        </section>

        <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
          <article className="viqa-panel rounded-[28px] p-5">
            <div className="mb-4 flex items-center gap-2">
              <Table2 className="h-5 w-5 text-viqa-violet" />
              <h2 className="font-display text-lg tracking-[0.16em] text-white">READER TABLE</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-sm">
                <thead className="text-xs uppercase tracking-[0.24em] text-slate-500">
                  <tr>
                    <th className="py-3">Reader</th>
                    <th>Exact Match</th>
                    <th>F1</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10 text-slate-200">
                  {readerComparison.map((row) => (
                    <tr key={row.reader}>
                      <td className="py-3 font-medium uppercase text-white">{row.reader}</td>
                      <td>{formatScore(row.exactMatch)}</td>
                      <td>{formatScore(row.f1)}</td>
                      <td>{formatLatency(row.avgLatencyMs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="viqa-panel rounded-[28px] p-5">
            <div className="mb-4 flex items-center gap-2">
              <FileWarning className="h-5 w-5 text-viqa-gold" />
              <h2 className="font-display text-lg tracking-[0.16em] text-white">ERROR ANALYSIS</h2>
            </div>
            <div className="grid gap-3">
              {errorAnalysis.map((item) => (
                <div key={item.issue} className="rounded-2xl border border-white/10 bg-black/20 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-medium text-white">{item.issue}</p>
                    <span className="rounded-full border border-viqa-gold/20 bg-viqa-gold/10 px-3 py-1 text-xs text-viqa-gold">
                      {item.count} cases
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-400">{item.note}</p>
                </div>
              ))}
            </div>
          </article>
        </section>
      </main>

      <SettingsPanel voices={synthesis.voices} onTestVoice={testVoice} />
    </MainLayout>
  );
}

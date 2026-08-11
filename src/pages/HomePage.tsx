import { useMemo, useState } from 'react';
import { FileText, GitCompareArrows, RotateCcw, X } from 'lucide-react';
import { AvatarScene } from '@/components/avatar/AvatarScene';
import { AnswerPanel } from '@/components/answer/AnswerPanel';
import { RetrieverComparison } from '@/components/compare/RetrieverComparison';
import { QuestionInput } from '@/components/input/QuestionInput';
import { Header } from '@/components/layout/Header';
import { MainLayout } from '@/components/layout/MainLayout';
import { PipelineFlow } from '@/components/pipeline/PipelineFlow';
import { SettingsPanel } from '@/components/settings/SettingsPanel';
import { useQaPipeline } from '@/hooks/useQaPipeline';
import { useSpeechRecognition } from '@/hooks/useSpeechRecognition';
import { useSpeechSynthesis } from '@/hooks/useSpeechSynthesis';
import { useAppStore } from '@/store/appStore';
import type { PassageResult, RetrieverComparisonRow } from '@/types/qa';

export function HomePage() {
  const [comparisonRows, setComparisonRows] = useState<RetrieverComparisonRow[]>([]);
  const [selectedSource, setSelectedSource] = useState<PassageResult | null>(null);
  const draft = useAppStore((state) => state.draft);
  const answer = useAppStore((state) => state.answer);
  const pipelineState = useAppStore((state) => state.pipelineState);
  const avatarState = useAppStore((state) => state.avatarState);
  const errorMessage = useAppStore((state) => state.errorMessage);
  const settings = useAppStore((state) => state.settings);
  const comparisonEnabled = useAppStore((state) => state.isComparisonOpen);
  const setDraft = useAppStore((state) => state.setDraft);
  const setAnswer = useAppStore((state) => state.setAnswer);
  const setComparisonOpen = useAppStore((state) => state.setComparisonOpen);
  const toggleSettings = useAppStore((state) => state.toggleSettings);
  const updateVoiceSettings = useAppStore((state) => state.updateVoiceSettings);
  const resetTransientState = useAppStore((state) => state.resetTransientState);
  const setStatusMessage = useAppStore((state) => state.setStatusMessage);
  const { submitQuestion, stopAll, stop } = useQaPipeline();
  const speech = useSpeechRecognition();
  const synthesis = useSpeechSynthesis();

  const composedQuestion = useMemo(
    () => `${draft} ${speech.transcript} ${speech.interimTranscript}`.replace(/\s+/g, ' ').trim(),
    [draft, speech.interimTranscript, speech.transcript],
  );
  const displayedAvatarState = speech.isListening
    ? 'listening'
    : composedQuestion && avatarState === 'idle'
      ? 'typing'
      : avatarState;

  const submit = async () => {
    const result = await submitQuestion(composedQuestion);
    if (result.compare) {
      setComparisonRows(result.compare);
    }
    setDraft('');
    speech.resetTranscript();
  };

  const clear = () => {
    setDraft('');
    speech.resetTranscript();
    setAnswer(null);
    setComparisonRows([]);
    setSelectedSource(null);
    resetTransientState();
  };

  const toggleVoiceInput = () => {
    if (speech.isListening) {
      speech.stopListening();
      return;
    }
    speech.startListening();
  };

  const reset = () => {
    stopAll();
    setComparisonRows([]);
    setSelectedSource(null);
    resetTransientState();
  };

  const testVoice = () => {
    synthesis.speak({
      text: 'Xin chào, tôi là Mari. Tôi có thể giúp bạn tìm câu trả lời trong tập tài liệu.',
      voiceName: settings.voice.voiceName,
      rate: settings.voice.rate,
      pitch: settings.voice.pitch,
      volume: settings.voice.volume,
    });
  };

  const viewSource = (passage: PassageResult) => {
    setSelectedSource(passage);
    setStatusMessage(`SOURCE ${passage.passage_id} | PAGE ${passage.page ?? '--'}`);
  };

  return (
    <MainLayout>
      <Header
        audioEnabled={settings.voice.enabled}
        onToggleAudio={() => updateVoiceSettings({ enabled: !settings.voice.enabled })}
        onToggleSettings={toggleSettings}
      />

      <main className="mt-4 grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,0.92fr)_minmax(500px,1.08fr)]">
        <div className="flex min-w-0 flex-col gap-4 overflow-y-auto pb-4 pr-1">
          <AvatarScene state={displayedAvatarState} />
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => setComparisonOpen(!comparisonEnabled)}
              className={`inline-flex items-center gap-2 rounded-2xl border px-4 py-3 text-sm font-medium transition ${
                comparisonEnabled
                  ? 'border-viqa-cyan/35 bg-viqa-cyan/15 text-viqa-cyan'
                  : 'border-white/10 bg-white/5 text-slate-200 hover:border-viqa-cyan/25'
              }`}
            >
              <GitCompareArrows className="h-4 w-4" />
              Compare retrievers
            </button>
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 transition hover:border-white/20"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </button>
            {errorMessage ? <span className="text-sm text-viqa-error">{errorMessage}</span> : null}
            {speech.error ? <span className="text-sm text-viqa-warning">{speech.error}</span> : null}
          </div>
          {comparisonEnabled && comparisonRows.length ? <RetrieverComparison rows={comparisonRows} /> : null}
        </div>

        <div className="flex min-w-0 flex-col gap-4 overflow-y-auto pb-4 pr-1">
          <div className="flex shrink-0 flex-col gap-4">
            <AnswerPanel response={answer} state={pipelineState} compareMode={comparisonEnabled} onViewSource={viewSource} />
            <PipelineFlow state={speech.isListening ? 'listening' : pipelineState} />
          </div>
        </div>
      </main>

      <QuestionInput
        value={draft}
        transcript={speech.transcript}
        interimTranscript={speech.interimTranscript}
        listening={speech.isListening}
        speechSupported={speech.isSupported}
        audioActive={pipelineState === 'speaking' || synthesis.speaking}
        onChange={setDraft}
        onSubmit={submit}
        onClear={clear}
        onVoiceToggle={toggleVoiceInput}
        onStopSpeaking={stop}
      />
      <SettingsPanel voices={synthesis.voices} onTestVoice={testVoice} />
      {selectedSource ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 px-4 backdrop-blur-sm" role="dialog" aria-modal="true">
          <section className="viqa-panel max-h-[82vh] w-full max-w-2xl overflow-y-auto p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium text-slate-400">
                  <FileText className="h-4 w-4 text-viqa-gold" />
                  Source passage
                </div>
                <h2 className="mt-2 text-xl font-semibold text-white">{selectedSource.title}</h2>
              </div>
              <button
                type="button"
                onClick={() => setSelectedSource(null)}
                aria-label="Close source passage"
                className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-200 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-300">
              <span className="rounded-full border border-slate-400/15 bg-slate-700/40 px-3 py-1.5">{selectedSource.document_id}</span>
              <span className="rounded-full border border-slate-400/15 bg-slate-700/40 px-3 py-1.5">{selectedSource.passage_id}</span>
              <span className="rounded-full border border-viqa-cyan/20 bg-viqa-cyan/10 px-3 py-1.5 text-viqa-cyan">
                Retrieval {(selectedSource.retrieval_score * 100).toFixed(1)}%
              </span>
              {typeof selectedSource.reader_score === 'number' ? (
                <span className="rounded-full border border-viqa-violet/20 bg-viqa-violet/10 px-3 py-1.5 text-viqa-violet">
                  Reader {(selectedSource.reader_score * 100).toFixed(1)}%
                </span>
              ) : null}
            </div>

            <p className="mt-5 rounded-lg border border-slate-400/15 bg-[#172033] p-4 text-sm leading-7 text-slate-200">
              {selectedSource.text}
            </p>
          </section>
        </div>
      ) : null}
    </MainLayout>
  );
}

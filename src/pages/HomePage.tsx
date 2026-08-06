import { useMemo, useState } from 'react';
import { GitCompareArrows, RotateCcw } from 'lucide-react';
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
    resetTransientState();
  };

  const viewSource = (passage: PassageResult) => {
    setStatusMessage(`SOURCE ${passage.passage_id} | PAGE ${passage.page ?? '--'}`);
  };

  return (
    <MainLayout>
      <Header
        audioEnabled={settings.voice.enabled}
        onToggleAudio={() => updateVoiceSettings({ enabled: !settings.voice.enabled })}
        onToggleSettings={toggleSettings}
      />

      <main className="mt-4 grid flex-1 gap-4 xl:grid-cols-[minmax(0,1.18fr)_minmax(420px,0.82fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <AvatarScene state={speech.isListening ? 'listening' : avatarState} />
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
              COMPARE RETRIEVERS
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

        <div className="flex min-w-0 flex-col gap-4">
          <PipelineFlow state={speech.isListening ? 'listening' : pipelineState} />
          <AnswerPanel response={answer} state={pipelineState} compareMode={comparisonEnabled} onViewSource={viewSource} />
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
      <SettingsPanel voices={synthesis.voices} />
    </MainLayout>
  );
}

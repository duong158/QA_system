import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { AssistantMessage, ThinkingMessage } from '@/components/chat/AssistantMessage';
import { ChatComposer } from '@/components/chat/ChatComposer';
import { ChatSidebar } from '@/components/chat/ChatSidebar';
import { SourceSidebar } from '@/components/answer/SourceSidebar';
import { Header } from '@/components/layout/Header';
import { MainLayout } from '@/components/layout/MainLayout';
import { SettingsPanel } from '@/components/settings/SettingsPanel';
import { useQaPipeline } from '@/hooks/useQaPipeline';
import { useSocraticFollowups } from '@/hooks/useSocraticFollowups';
import { useSpeechRecognition } from '@/hooks/useSpeechRecognition';
import { useSpeechSynthesis } from '@/hooks/useSpeechSynthesis';
import { useAppStore } from '@/store/appStore';
import type { ChatTurn } from '@/types/chat';
import type { FollowUpCandidate, QaResponse } from '@/types/qa';
import { collapseRepeatedQuestion, mergeQuestionParts } from '@/utils/questionInput';

const showDebugScores = import.meta.env.DEV || String(import.meta.env.VITE_QA_DEBUG ?? 'false').toLowerCase() === 'true';
const MariPanel = lazy(() => import('@/components/avatar/MariPanel').then((module) => ({ default: module.MariPanel })));
const MariSheet = lazy(() => import('@/components/avatar/MariPanel').then((module) => ({ default: module.MariSheet })));

function turnId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function HomePage() {
  const [avatarCollapsed, setAvatarCollapsed] = useState(false);
  const [mobileMariOpen, setMobileMariOpen] = useState(false);
  const [desktopAvatarVisible, setDesktopAvatarVisible] = useState(false);
  const [activeSourceResponse, setActiveSourceResponse] = useState<QaResponse | null>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const requestInFlightRef = useRef(false);

  const draft = useAppStore((state) => state.draft);
  const answer = useAppStore((state) => state.answer);
  const sessions = useAppStore((state) => state.sessions);
  const currentSessionId = useAppStore((state) => state.currentSessionId);
  const turns = useMemo(() => sessions.find((s) => s.id === currentSessionId)?.turns || [], [sessions, currentSessionId]);
  const pipelineState = useAppStore((state) => state.pipelineState);
  const avatarState = useAppStore((state) => state.avatarState);
  const errorMessage = useAppStore((state) => state.errorMessage);
  const settings = useAppStore((state) => state.settings);
  const isHistoryOpen = useAppStore((state) => state.isHistoryOpen);
  const setDraft = useAppStore((state) => state.setDraft);
  const setQuestion = useAppStore((state) => state.setQuestion);
  const setAnswer = useAppStore((state) => state.setAnswer);
  const clearSessions = useAppStore((state) => state.clearSessions);
  const addTurn = useAppStore((state) => state.addTurn);
  const updateTurn = useAppStore((state) => state.updateTurn);
  const createNewSession = useAppStore((state) => state.createNewSession);
  const switchSession = useAppStore((state) => state.switchSession);
  const toggleSettings = useAppStore((state) => state.toggleSettings);
  const setHistoryOpen = useAppStore((state) => state.setHistoryOpen);
  const toggleHistory = useAppStore((state) => state.toggleHistory);
  const updateVoiceSettings = useAppStore((state) => state.updateVoiceSettings);
  const resetTransientState = useAppStore((state) => state.resetTransientState);

  const { submitQuestion, stopAll, stop } = useQaPipeline();
  const speech = useSpeechRecognition();
  const synthesis = useSpeechSynthesis();
  const question = useAppStore((state) => state.question);
  const socratic = useSocraticFollowups(answer);

  const composedQuestion = useMemo(
    () => mergeQuestionParts(draft, speech.transcript, speech.interimTranscript),
    [draft, speech.interimTranscript, speech.transcript],
  );
  const isProcessing = ['retrieving', 'reading', 'extracting'].includes(pipelineState);
  const displayedAvatarState = speech.isListening
    ? 'listening'
    : synthesis.speaking
      ? 'speaking'
    : composedQuestion && avatarState === 'idle'
      ? 'typing'
      : avatarState;

  useEffect(() => {
    const media = window.matchMedia('(min-width: 1200px)');
    const syncAvatarVisibility = () => {
      setDesktopAvatarVisible(media.matches);
      if (media.matches) setMobileMariOpen(false);
    };
    syncAvatarVisibility();
    media.addEventListener('change', syncAvatarVisibility);
    return () => media.removeEventListener('change', syncAvatarVisibility);
  }, []);

  useEffect(() => {
    const viewport = conversationRef.current;
    if (!viewport || !shouldAutoScrollRef.current) return;
    const frame = window.requestAnimationFrame(() => viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' }));
    return () => window.cancelAnimationFrame(frame);
  }, [turns, socratic.followUps, socratic.loadState]);

  const submitConversationQuestion = async (
    rawQuestion: string,
    preferredPassageId?: string | null,
  ) => {
    const normalizedQuestion = collapseRepeatedQuestion(rawQuestion);
    if (!normalizedQuestion || isProcessing || requestInFlightRef.current) return;

    const id = turnId();
    requestInFlightRef.current = true;
    shouldAutoScrollRef.current = true;
    addTurn({ id, question: normalizedQuestion, createdAt: Date.now(), status: 'pending' });
    try {
      const result = await submitQuestion(normalizedQuestion, { preferredPassageId });
      updateTurn(id, result.response
        ? { status: 'complete', response: result.response }
        : { status: 'error', error: 'Không thể nhận câu trả lời. Vui lòng kiểm tra kết nối và thử lại.' }
      );
    } finally {
      requestInFlightRef.current = false;
    }
  };

  const submit = async () => {
    const nextQuestion = composedQuestion;
    if (!nextQuestion.trim() || isProcessing) return;
    setDraft('');
    speech.stopListening();
    speech.resetTranscript();
    await submitConversationQuestion(nextQuestion);
  };

  const changeDraft = (nextDraft: string) => {
    if (speech.transcript || speech.interimTranscript) {
      speech.stopListening();
      speech.resetTranscript();
    }
    setDraft(nextDraft);
  };

  const startNewChat = () => {
    stopAll();
    setDraft('');
    setQuestion('');
    setAnswer(null);
    createNewSession();
    speech.resetTranscript();
    socratic.resetSession();
    resetTransientState();
    setHistoryOpen(false);
  };

  const toggleVoiceInput = () => {
    if (speech.isListening) speech.stopListening();
    else speech.startListening();
  };

  const askFollowUp = async (followUp: FollowUpCandidate) => {
    socratic.markSelected(followUp);
    speech.stopListening();
    speech.resetTranscript();
    setDraft('');
    await submitConversationQuestion(followUp.question, followUp.source_passage_id);
  };

  const speakText = (text: string) => {
    synthesis.speak({
      text,
      voiceName: settings.voice.voiceName,
      rate: settings.voice.rate,
      pitch: settings.voice.pitch,
      volume: settings.voice.volume,
    });
  };

  const speakFollowUp = (followUp: FollowUpCandidate) => speakText(followUp.question);

  const testVoice = () => speakText('Xin chào, tôi là Mari. Tôi có thể giúp bạn học từ tập tài liệu này.');

  const onConversationScroll = () => {
    const viewport = conversationRef.current;
    if (!viewport) return;
    shouldAutoScrollRef.current = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 140;
  };

  return (
    <MainLayout>
      <Header
        audioEnabled={settings.voice.enabled}
        onToggleAudio={() => updateVoiceSettings({ enabled: !settings.voice.enabled })}
        onToggleSettings={toggleSettings}
        onToggleHistory={toggleHistory}
        onToggleMari={() => setMobileMariOpen(true)}
      />

      <main className={`mt-3 grid min-h-0 flex-1 gap-3 lg:grid-cols-[248px_minmax(0,1fr)] ${
        avatarCollapsed
          ? 'min-[1200px]:grid-cols-[248px_minmax(0,1fr)]'
          : 'min-[1200px]:grid-cols-[248px_minmax(0,1fr)_272px]'
      }`}>
        <ChatSidebar
          open={isHistoryOpen}
          items={sessions}
          onClose={() => setHistoryOpen(false)}
          onNewChat={startNewChat}
          onSelectSession={switchSession}
          onClear={clearSessions}
        />

        <section className="surface-card flex min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl" aria-label="Cuộc trò chuyện với Mari">
          <div
            ref={conversationRef}
            onScroll={onConversationScroll}
            className="chat-scroll min-h-0 flex-1 overflow-y-auto px-3 py-5 sm:px-6 sm:py-7"
            role="log"
            aria-live="polite"
          >
            <div className="mx-auto flex w-full max-w-[860px] flex-col gap-5">
              {!turns.length ? (
                <div className="mx-auto flex max-w-lg flex-col items-center px-4 py-10 text-center sm:py-16">
                  <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--primary-soft)] text-[var(--primary)]">
                    <Sparkles className="h-6 w-6" />
                  </span>
                  <h1 className="mt-5 text-xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-2xl">Bạn muốn khám phá điều gì?</h1>
                  <p className="mt-2 text-sm leading-6 text-[var(--text-secondary)]">
                    Hỏi một câu về tài liệu. Mari sẽ trả lời, dẫn nguồn và luôn gợi mở từng bước.
                  </p>
                </div>
              ) : null}

              {turns.map((turn) => {
                const isCurrentResponse = Boolean(turn.response && answer === turn.response);
                return (
                  <div key={turn.id} className="grid gap-3">
                    <div className="flex justify-end">
                      <div className="max-w-[78%] rounded-2xl rounded-tr-md bg-[var(--primary)] px-4 py-3 text-[15px] leading-6 text-white shadow-sm sm:max-w-[72%]">
                        {turn.question}
                      </div>
                    </div>
                    {turn.status === 'pending' ? <ThinkingMessage /> : null}
                    {turn.status === 'error' ? (
                      <div className="ml-10 max-w-[82%] rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700" role="alert">{turn.error}</div>
                    ) : null}
                    {turn.response ? (
                      <AssistantMessage
                        response={turn.response}
                        followUps={isCurrentResponse ? socratic.followUps : []}
                        followUpsState={isCurrentResponse ? socratic.loadState : 'idle'}
                        followUpsLatencyMs={isCurrentResponse ? socratic.latencyMs : null}
                        onFollowUpSelect={askFollowUp}
                        onFollowUpSpeak={isCurrentResponse && synthesis.isSupported ? speakFollowUp : undefined}
                        onSpeakAnswer={speakText}
                        onSourceClick={setActiveSourceResponse}
                        showDebug={showDebugScores}
                      />
                    ) : null}
                  </div>
                );
              })}

              {errorMessage && !turns.some((turn) => turn.status === 'error') ? (
                <p className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700" role="alert">{errorMessage}</p>
              ) : null}
              {speech.error ? <p className="text-center text-xs text-amber-700">{speech.error}</p> : null}
            </div>
          </div>

          <ChatComposer
            value={draft}
            transcript={speech.transcript}
            interimTranscript={speech.interimTranscript}
            listening={speech.isListening}
            speechSupported={speech.isSupported}
            audioActive={pipelineState === 'speaking' || synthesis.speaking}
            submitting={isProcessing}
            onChange={changeDraft}
            onSubmit={submit}
            onVoiceToggle={toggleVoiceInput}
            onStopSpeaking={stop}
          />
        </section>

        {desktopAvatarVisible ? (
          <Suspense fallback={<aside className="hidden min-h-0 animate-pulse rounded-2xl bg-[var(--surface-muted)] min-[1200px]:block" aria-label="Đang tải Mari" />}>
            <MariPanel
              state={displayedAvatarState}
              collapsed={avatarCollapsed}
              onCollapsedChange={setAvatarCollapsed}
            />
          </Suspense>
        ) : null}
      </main>

      {!desktopAvatarVisible && mobileMariOpen ? (
        <Suspense fallback={<div className="fixed inset-x-3 bottom-3 z-50 h-56 animate-pulse rounded-2xl bg-[var(--surface-muted)]" aria-label="Đang tải Mari" />}>
          <MariSheet
            open={mobileMariOpen}
            state={displayedAvatarState}
            onClose={() => setMobileMariOpen(false)}
          />
        </Suspense>
      ) : null}

      <SettingsPanel voices={synthesis.voices} onTestVoice={testVoice} />
      
      <SourceSidebar 
        open={!!activeSourceResponse} 
        onClose={() => setActiveSourceResponse(null)} 
        response={activeSourceResponse} 
      />
    </MainLayout>
  );
}

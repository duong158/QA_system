import { useMemo, useRef } from 'react';
import { askQuestion, compareRetrievers } from '@/services/qaService';
import { useAppStore } from '@/store/appStore';
import type { AskQuestionRequest, QaResponse } from '@/types/qa';
import type { PipelineState } from '@/types/pipeline';
import { useSpeechSynthesis } from '@/hooks/useSpeechSynthesis';
import { collapseRepeatedQuestion } from '@/utils/questionInput';

export interface SubmitQuestionResult {
  response?: QaResponse;
  compare?: Awaited<ReturnType<typeof compareRetrievers>>;
}

export function useQaPipeline() {
  const setQuestion = useAppStore((state) => state.setQuestion);
  const setAnswer = useAppStore((state) => state.setAnswer);
  const setPipelineState = useAppStore((state) => state.setPipelineState);
  const setAvatarState = useAppStore((state) => state.setAvatarState);
  const setErrorMessage = useAppStore((state) => state.setErrorMessage);
  const setStatusMessage = useAppStore((state) => state.setStatusMessage);
  const setSpeaking = useAppStore((state) => state.setSpeaking);
  const settings = useAppStore((state) => state.settings);
  const comparison = useAppStore((state) => state.comparison);
  const isComparisonOpen = useAppStore((state) => state.isComparisonOpen);
  const setComparisonQuestion = useAppStore((state) => state.setComparisonQuestion);
  const { speak, stop, isSupported } = useSpeechSynthesis();
  const timersRef = useRef<number[]>([]);
  const currentRunRef = useRef(0);
  const requestControllerRef = useRef<AbortController | null>(null);

  const clearTimers = () => {
    timersRef.current.forEach((timer) => window.clearTimeout(timer));
    timersRef.current = [];
  };

  const stopAll = () => {
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    clearTimers();
    stop();
    setSpeaking(false);
    setPipelineState('idle');
    setAvatarState('idle');
    setStatusMessage('SYSTEM ONLINE');
  };

  const submitQuestion = async (question: string): Promise<SubmitQuestionResult> => {
    const normalizedQuestion = collapseRepeatedQuestion(question);
    if (!normalizedQuestion) {
      return {};
    }

    clearTimers();
    requestControllerRef.current?.abort();
    const requestController = new AbortController();
    requestControllerRef.current = requestController;
    currentRunRef.current += 1;
    const runId = currentRunRef.current;
    setErrorMessage(null);
    setQuestion(normalizedQuestion);
    setAnswer(null);
    setSpeaking(false);
    setPipelineState('retrieving');
    setAvatarState('retrieving');
    setStatusMessage('SEARCHING KNOWLEDGE BASE');
    setComparisonQuestion(isComparisonOpen ? normalizedQuestion : '');

    timersRef.current.push(
      window.setTimeout(() => {
        if (currentRunRef.current !== runId) {
          return;
        }
        setPipelineState('reading');
        setAvatarState('reading');
        setStatusMessage('ANALYZING PASSAGES');
      }, 400),
    );

    timersRef.current.push(
      window.setTimeout(() => {
        if (currentRunRef.current !== runId) {
          return;
        }
        setPipelineState('extracting');
        setAvatarState('thinking');
        setStatusMessage('EXTRACTING ANSWER');
      }, 900),
    );

    const request: AskQuestionRequest = {
      question: normalizedQuestion,
      retriever: settings.retriever,
      reader: settings.reader,
      top_k: settings.topK,
    };

    try {
      const [response, compare] = await Promise.all([
        askQuestion(request, requestController.signal),
        isComparisonOpen ? compareRetrievers(normalizedQuestion) : Promise.resolve(undefined),
      ]);

      if (currentRunRef.current !== runId) {
        return {};
      }

      requestControllerRef.current = null;

      setAnswer(response);
      setStatusMessage('ANSWER READY');

      const responseHasAnswer = response.has_answer ?? Boolean(response.answer);
      const finalState: PipelineState = responseHasAnswer ? 'completed' : 'no-answer';
      setPipelineState(finalState);
      setAvatarState(responseHasAnswer ? 'success' : 'no-answer');

      timersRef.current.push(
        window.setTimeout(() => {
          if (currentRunRef.current !== runId) {
            return;
          }
          if (responseHasAnswer && response.answer) {
            setPipelineState('speaking');
            setAvatarState('speaking');
            setSpeaking(true);
            if (settings.voice.enabled) {
              speak({
                text: response.answer,
                voiceName: settings.voice.voiceName,
                rate: settings.voice.rate,
                pitch: settings.voice.pitch,
                volume: settings.voice.volume,
              });
            }
            timersRef.current.push(
              window.setTimeout(() => {
                if (currentRunRef.current !== runId) {
                  return;
                }
                setPipelineState('idle');
                setAvatarState('idle');
                setSpeaking(false);
                setStatusMessage('SYSTEM ONLINE');
              }, Math.max(1800, response.answer.length * 28)),
            );
          } else {
            setStatusMessage('NO ANSWER FOUND');
            timersRef.current.push(
              window.setTimeout(() => {
                if (currentRunRef.current !== runId) {
                  return;
                }
                setPipelineState('idle');
                setAvatarState('idle');
                setStatusMessage('SYSTEM ONLINE');
              }, 1600),
            );
          }
        }, 0),
      );

      return { response, compare };
    } catch (error) {
      if (currentRunRef.current !== runId) {
        return {};
      }
      requestControllerRef.current = null;
      const message = error instanceof Error ? error.message : 'Unknown error';
      setErrorMessage(message);
      setStatusMessage('ERROR');
      setPipelineState('error');
      setAvatarState('error');
      timersRef.current.push(
        window.setTimeout(() => {
          if (currentRunRef.current !== runId) {
            return;
          }
          setPipelineState('idle');
          setAvatarState('idle');
          setStatusMessage('SYSTEM ONLINE');
        }, 1800),
      );
      return {};
    }
  };

  return useMemo(
    () => ({
      submitQuestion,
      stopAll,
      speak,
      stop,
      isSpeechSupported: isSupported,
      clearTimers,
    }),
    [isSupported, speak],
  );
}

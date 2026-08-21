import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchSocraticFollowups } from '@/services/qaService';
import type { FollowUpCandidate, QaResponse } from '@/types/qa';

export type SocraticLoadState = 'idle' | 'loading' | 'ready' | 'error';

interface SocraticSession {
  subjectKey: string;
  askedQuestions: Set<string>;
}

function normalizedKey(value: string | null | undefined): string {
  return String(value ?? '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/gi, 'd')
    .toLocaleLowerCase('vi')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

export function useSocraticFollowups(response: QaResponse | null) {
  const [followUps, setFollowUps] = useState<FollowUpCandidate[]>([]);
  const [loadState, setLoadState] = useState<SocraticLoadState>('idle');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const sessionRef = useRef<SocraticSession>({
    subjectKey: '',
    askedQuestions: new Set(),
  });

  useEffect(() => {
    const controller = new AbortController();
    if (!response || !response.question || !response.answer) {
      setFollowUps([]);
      setLoadState('idle');
      setLatencyMs(null);
      return () => controller.abort();
    }

    const question = response.question;
    const subjectKey = normalizedKey(question);
    if (sessionRef.current.subjectKey !== subjectKey) {
      sessionRef.current = {
        subjectKey,
        askedQuestions: new Set(),
      };
    }

    sessionRef.current.askedQuestions.add(question);

    setFollowUps([]);
    setLoadState('loading');
    setLatencyMs(null);
    void fetchSocraticFollowups(
      {
        question: question,
        answer: response.answer,
        selected_passage_id: response.selected_passage_id ?? null,
        retrieved_passage_ids: response.passages?.map(p => p.passage_id) || [],
        question_type: response.question_type,
        relation: response.relation_type ?? null,
        subject: response.question_subject ?? null,
        target: response.question_target ?? null,
        predicate: response.question_predicate ?? null,
        modifier: response.question_modifier ?? null,
        visited_relations: [],
        asked_questions: Array.from(sessionRef.current.askedQuestions),
        limit: 3,
      },
      controller.signal,
    )
      .then((result) => {
        if (controller.signal.aborted) {
          return;
        }
        setFollowUps(result.followups);
        setLatencyMs(result.processing_time_ms);
        setLoadState('ready');
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        // Socratic suggestions are optional; never promote this to the main QA error state.
        console.debug('Socratic follow-up generation unavailable', error);
        setFollowUps([]);
        setLatencyMs(null);
        setLoadState('error');
      });

    return () => controller.abort();
  }, [response]);

  const markSelected = useCallback((followUp: FollowUpCandidate) => {
    if (followUp.subject) {
      sessionRef.current.subjectKey = normalizedKey(followUp.subject);
    }
    sessionRef.current.askedQuestions.add(followUp.question);
  }, []);

  const resetSession = useCallback(() => {
    sessionRef.current = {
      subjectKey: '',
      askedQuestions: new Set(),
    };
    setFollowUps([]);
    setLoadState('idle');
    setLatencyMs(null);
  }, []);

  return {
    followUps,
    loadState,
    latencyMs,
    markSelected,
    resetSession,
  };
}

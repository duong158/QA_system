import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchSocraticFollowups } from '@/services/qaService';
import type { FollowUpCandidate, QaResponse } from '@/types/qa';

export type SocraticLoadState = 'idle' | 'loading' | 'ready' | 'error';

interface SocraticSession {
  subjectKey: string;
  visitedRelations: Set<string>;
  askedQuestions: Set<string>;
  contextPassageIds: Set<string>;
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

function responseRelation(response: QaResponse): string | null {
  return response.semantic_relation || response.question_relation || response.relation_type || null;
}

export function useSocraticFollowups(response: QaResponse | null) {
  const [followUps, setFollowUps] = useState<FollowUpCandidate[]>([]);
  const [loadState, setLoadState] = useState<SocraticLoadState>('idle');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const sessionRef = useRef<SocraticSession>({
    subjectKey: '',
    visitedRelations: new Set(),
    askedQuestions: new Set(),
    contextPassageIds: new Set(),
  });

  useEffect(() => {
    const controller = new AbortController();
    const hasAnswer = response?.has_answer ?? Boolean(response?.answer);
    if (!response || !hasAnswer || !response.answer) {
      setFollowUps([]);
      setLoadState('idle');
      setLatencyMs(null);
      return () => controller.abort();
    }

    const subjectKey = normalizedKey(response.question_subject || response.question);
    if (sessionRef.current.subjectKey !== subjectKey) {
      sessionRef.current = {
        subjectKey,
        visitedRelations: new Set(),
        askedQuestions: new Set(),
        contextPassageIds: new Set(),
      };
    }

    if (response.selected_passage_id) {
      sessionRef.current.contextPassageIds.add(response.selected_passage_id);
    }
    for (const passage of response.passages ?? []) {
      if (passage.passage_id) {
        sessionRef.current.contextPassageIds.add(passage.passage_id);
      }
    }

    const relation = responseRelation(response);
    if (relation) {
      sessionRef.current.visitedRelations.add(relation.toUpperCase());
    }
    sessionRef.current.askedQuestions.add(response.question);

    setFollowUps([]);
    setLoadState('loading');
    setLatencyMs(null);
    void fetchSocraticFollowups(
      {
        question: response.question,
        answer: response.answer,
        selected_passage_id: response.selected_passage_id ?? null,
        retrieved_passage_ids: Array.from(sessionRef.current.contextPassageIds),
        question_type: response.question_type,
        relation,
        subject: response.question_subject,
        target: response.question_target,
        predicate: response.question_predicate,
        modifier: response.question_modifier,
        visited_relations: Array.from(sessionRef.current.visitedRelations),
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
    if (followUp.relation) {
      sessionRef.current.visitedRelations.add(followUp.relation.toUpperCase());
    }
    sessionRef.current.askedQuestions.add(followUp.question);
  }, []);

  const resetSession = useCallback(() => {
    sessionRef.current = {
      subjectKey: '',
      visitedRelations: new Set(),
      askedQuestions: new Set(),
      contextPassageIds: new Set(),
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

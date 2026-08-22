import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AvatarState } from '@/types/avatar';
import type { PipelineState } from '@/types/pipeline';
import type { QaResponse, ReaderType, RetrieverType } from '@/types/qa';
import type { ChatSession, ChatTurn } from '@/types/chat';

export interface VoiceSettings {
  enabled: boolean;
  voiceName?: string;
  rate: number;
  pitch: number;
  volume: number;
}

export interface DisplaySettings {
  particlesEnabled: boolean;
  hologramEnabled: boolean;
  advancedMotion: boolean;
  lowPerformanceMode: boolean;
}

export interface QaSettings {
  retriever: RetrieverType;
  reader: ReaderType;
  topK: number;
  voice: VoiceSettings;
  display: DisplaySettings;
}

export interface RetrieverComparisonState {
  enabled: boolean;
  question: string;
}

export interface AppState {
  question: string;
  draft: string;
  answer: QaResponse | null;
  sessions: ChatSession[];
  currentSessionId: string | null;
  pipelineState: PipelineState;
  avatarState: AvatarState;
  isSettingsOpen: boolean;
  isComparisonOpen: boolean;
  isHistoryOpen: boolean;
  isSpeaking: boolean;
  isListening: boolean;
  recognitionTranscript: string;
  errorMessage: string | null;
  statusMessage: string;
  settings: QaSettings;
  comparison: RetrieverComparisonState;
  setQuestion: (question: string) => void;
  setDraft: (draft: string) => void;
  setAnswer: (answer: QaResponse | null) => void;
  
  // Session Actions
  createNewSession: () => void;
  switchSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  clearSessions: () => void;
  addTurn: (turn: ChatTurn) => void;
  updateTurn: (turnId: string, updates: Partial<ChatTurn>) => void;
  removeTurn: (turnId: string) => void;

  setPipelineState: (state: PipelineState) => void;
  setAvatarState: (state: AvatarState) => void;
  setSettingsOpen: (open: boolean) => void;
  toggleSettings: () => void;
  setHistoryOpen: (open: boolean) => void;
  toggleHistory: () => void;
  setComparisonOpen: (open: boolean) => void;
  setSpeaking: (speaking: boolean) => void;
  setListening: (listening: boolean) => void;
  setRecognitionTranscript: (transcript: string) => void;
  setErrorMessage: (message: string | null) => void;
  setStatusMessage: (message: string) => void;
  updateSettings: (next: Partial<QaSettings>) => void;
  updateVoiceSettings: (next: Partial<VoiceSettings>) => void;
  updateDisplaySettings: (next: Partial<DisplaySettings>) => void;
  setComparisonQuestion: (question: string) => void;
  resetTransientState: () => void;
}

const defaultSettings: QaSettings = {
  retriever: 'bm25',
  reader: 'phobert',
  topK: 10,
  voice: {
    enabled: true,
    rate: 0.92,
    pitch: 1.08,
    volume: 1,
  },
  display: {
    particlesEnabled: true,
    hologramEnabled: true,
    advancedMotion: true,
    lowPerformanceMode: false,
  },
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      question: '',
      draft: '',
      answer: null,
      sessions: [],
      currentSessionId: null,
      pipelineState: 'idle',
      avatarState: 'idle',
      isSettingsOpen: false,
      isComparisonOpen: false,
      isHistoryOpen: false,
      isSpeaking: false,
      isListening: false,
      recognitionTranscript: '',
      errorMessage: null,
      statusMessage: 'SYSTEM ONLINE',
      settings: defaultSettings,
      comparison: {
        enabled: false,
        question: '',
      },
      setQuestion: (question) => set({ question }),
      setDraft: (draft) => set({ draft }),
      setAnswer: (answer) => set({ answer }),
      
      createNewSession: () => set(() => ({ currentSessionId: null })),
      switchSession: (sessionId) => set({ currentSessionId: sessionId, isHistoryOpen: false }),
      deleteSession: (sessionId) => set((state) => {
        const nextSessions = state.sessions.filter(s => s.id !== sessionId);
        return {
          sessions: nextSessions,
          currentSessionId: state.currentSessionId === sessionId ? null : state.currentSessionId,
        };
      }),
      clearSessions: () => set({ sessions: [], currentSessionId: null }),
      addTurn: (turn) => set((state) => {
        if (!state.currentSessionId) {
          const newSession: ChatSession = {
            id: `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            title: turn.question,
            turns: [turn],
            createdAt: Date.now(),
            updatedAt: Date.now()
          };
          return {
            sessions: [newSession, ...state.sessions].slice(0, 50),
            currentSessionId: newSession.id
          };
        }

        const nextSessions = state.sessions.map(s => {
          if (s.id === state.currentSessionId) {
            return {
              ...s,
              turns: [...s.turns, turn],
              updatedAt: Date.now()
            };
          }
          return s;
        });
        
        const activeIndex = nextSessions.findIndex(s => s.id === state.currentSessionId);
        if (activeIndex > 0) {
          const activeSession = nextSessions.splice(activeIndex, 1)[0];
          nextSessions.unshift(activeSession);
        }

        return { sessions: nextSessions };
      }),
      updateTurn: (turnId, updates) => set((state) => {
        if (!state.currentSessionId) return state;
        return {
          sessions: state.sessions.map(s => {
            if (s.id === state.currentSessionId) {
              return {
                ...s,
                turns: s.turns.map(t => t.id === turnId ? { ...t, ...updates } : t),
                updatedAt: Date.now()
              };
            }
            return s;
          })
        };
      }),
      removeTurn: (turnId) => set((state) => {
        if (!state.currentSessionId) return state;
        return {
          sessions: state.sessions.map(s => {
            if (s.id === state.currentSessionId) {
              return {
                ...s,
                turns: s.turns.filter(t => t.id !== turnId),
                updatedAt: Date.now()
              };
            }
            return s;
          })
        };
      }),

      setPipelineState: (pipelineState) => set({ pipelineState }),
      setAvatarState: (avatarState) => set({ avatarState }),
      setSettingsOpen: (isSettingsOpen) => set({ isSettingsOpen }),
      toggleSettings: () => set((state) => ({ isSettingsOpen: !state.isSettingsOpen })),
      setHistoryOpen: (isHistoryOpen) => set({ isHistoryOpen }),
      toggleHistory: () => set((state) => ({ isHistoryOpen: !state.isHistoryOpen })),
      setComparisonOpen: (isComparisonOpen) => set({ isComparisonOpen }),
      setSpeaking: (isSpeaking) => set({ isSpeaking }),
      setListening: (isListening) => set({ isListening }),
      setRecognitionTranscript: (recognitionTranscript) => set({ recognitionTranscript }),
      setErrorMessage: (errorMessage) => set({ errorMessage }),
      setStatusMessage: (statusMessage) => set({ statusMessage }),
      updateSettings: (next) => set((state) => ({ settings: { ...state.settings, ...next } })),
      updateVoiceSettings: (next) =>
        set((state) => ({
          settings: {
            ...state.settings,
            voice: { ...state.settings.voice, ...next },
          },
        })),
      updateDisplaySettings: (next) =>
        set((state) => ({
          settings: {
            ...state.settings,
            display: { ...state.settings.display, ...next },
          },
        })),
      setComparisonQuestion: (question) =>
        set((state) => ({
          comparison: {
            ...state.comparison,
            enabled: Boolean(question),
            question,
          },
        })),
      resetTransientState: () =>
        set({
          pipelineState: 'idle',
          avatarState: 'idle',
          isSpeaking: false,
          isListening: false,
          recognitionTranscript: '',
          errorMessage: null,
          statusMessage: 'SYSTEM ONLINE',
        }),
    }),
    {
      name: 'viqa-nexus-settings',
      version: 8,
      partialize: (state) => ({ 
        settings: state.settings,
        sessions: state.sessions,
        currentSessionId: state.currentSessionId
      }),
      migrate: (persistedState: any) => {
        const state = persistedState as Partial<AppState> | undefined;
        const settings = state?.settings;
        
        if (!settings) {
          return {
            ...persistedState,
            sessions: state?.sessions || [],
            currentSessionId: state?.currentSessionId || null
          } as Partial<AppState>;
        }

        const migratedSettings = {
          ...(settings as QaSettings & { socraticEnabled?: boolean }),
        };
        delete migratedSettings.socraticEnabled;

        return {
          ...state,
          sessions: state?.sessions || [],
          currentSessionId: state?.currentSessionId || null,
          settings: {
            ...defaultSettings,
            ...migratedSettings,
            retriever: settings.retriever === 'tfidf' || settings.retriever === 'bm25' || settings.retriever === 'hybrid' || settings.retriever === 'dense' ? settings.retriever : defaultSettings.retriever,
            reader: settings.reader === 'phobert' ? settings.reader : defaultSettings.reader,
            voice: {
              ...defaultSettings.voice,
              ...settings.voice,
              rate: settings.voice?.rate === 1 ? defaultSettings.voice.rate : settings.voice?.rate ?? defaultSettings.voice.rate,
              pitch: settings.voice?.pitch === 1 ? defaultSettings.voice.pitch : settings.voice?.pitch ?? defaultSettings.voice.pitch,
            },
            display: {
              ...defaultSettings.display,
              ...settings.display,
            },
          },
        };
      },
    },
  ),
);

export { defaultSettings };

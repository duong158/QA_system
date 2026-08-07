import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AvatarState } from '@/types/avatar';
import type { PipelineState } from '@/types/pipeline';
import type { QaResponse, ReaderType, RetrieverType } from '@/types/qa';

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
  pipelineState: PipelineState;
  avatarState: AvatarState;
  isSettingsOpen: boolean;
  isComparisonOpen: boolean;
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
  setPipelineState: (state: PipelineState) => void;
  setAvatarState: (state: AvatarState) => void;
  setSettingsOpen: (open: boolean) => void;
  toggleSettings: () => void;
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
  retriever: 'dense',
  reader: 'phobert',
  topK: 5,
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
      pipelineState: 'idle',
      avatarState: 'idle',
      isSettingsOpen: false,
      isComparisonOpen: false,
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
      setPipelineState: (pipelineState) => set({ pipelineState }),
      setAvatarState: (avatarState) => set({ avatarState }),
      setSettingsOpen: (isSettingsOpen) => set({ isSettingsOpen }),
      toggleSettings: () => set((state) => ({ isSettingsOpen: !state.isSettingsOpen })),
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
      version: 2,
      partialize: (state) => ({ settings: state.settings }),
      migrate: (persistedState) => {
        const state = persistedState as Partial<AppState> | undefined;
        const settings = state?.settings;
        if (!settings) {
          return persistedState;
        }

        return {
          ...state,
          settings: {
            ...defaultSettings,
            ...settings,
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

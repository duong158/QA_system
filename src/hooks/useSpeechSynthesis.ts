import { useEffect, useMemo, useState } from 'react';
import { getAvailableVoices, pickVietnameseVoice, supportsSpeechSynthesis } from '@/utils/speechUtils';

interface SpeakOptions {
  text: string;
  voiceName?: string;
  rate?: number;
  pitch?: number;
  volume?: number;
  lang?: string;
}

interface UseSpeechSynthesisResult {
  isSupported: boolean;
  voices: SpeechSynthesisVoice[];
  speaking: boolean;
  speak: (options: SpeakOptions) => void;
  stop: () => void;
}

export function useSpeechSynthesis(): UseSpeechSynthesisResult {
  const isSupported = useMemo(() => supportsSpeechSynthesis(), []);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    if (!isSupported) {
      return;
    }

    const updateVoices = () => setVoices(getAvailableVoices());
    updateVoices();
    window.speechSynthesis.addEventListener('voiceschanged', updateVoices);

    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', updateVoices);
    };
  }, [isSupported]);

  const speak = ({ text, voiceName, rate = 1, pitch = 1, volume = 1, lang = 'vi-VN' }: SpeakOptions) => {
    if (!isSupported || !text.trim()) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = pickVietnameseVoice(voices, voiceName);
    if (voice) {
      utterance.voice = voice;
    }
    utterance.lang = voice?.lang || lang;
    utterance.rate = rate;
    utterance.pitch = pitch;
    utterance.volume = volume;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  const stop = () => {
    if (!isSupported) {
      return;
    }

    window.speechSynthesis.cancel();
    setSpeaking(false);
  };

  return {
    isSupported,
    voices,
    speaking,
    speak,
    stop,
  };
}
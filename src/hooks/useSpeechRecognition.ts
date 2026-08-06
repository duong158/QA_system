import { useEffect, useMemo, useRef, useState } from 'react';
import { supportsSpeechRecognition } from '@/utils/speechUtils';

type SpeechRecognitionCtor = typeof window.SpeechRecognition;

type SpeechRecognitionEvent = Event & {
  results: SpeechRecognitionResultList;
  resultIndex: number;
};

interface UseSpeechRecognitionResult {
  isSupported: boolean;
  isListening: boolean;
  transcript: string;
  interimTranscript: string;
  error: string | null;
  startListening: () => void;
  stopListening: () => void;
  resetTranscript: () => void;
}

export function useSpeechRecognition(): UseSpeechRecognitionResult {
  const isSupported = useMemo(() => supportsSpeechRecognition(), []);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isSupported || recognitionRef.current) {
      return;
    }

    const SpeechRecognitionImpl = (window.SpeechRecognition || window.webkitSpeechRecognition) as SpeechRecognitionCtor | undefined;
    if (!SpeechRecognitionImpl) {
      return;
    }

    const recognition = new SpeechRecognitionImpl();
    recognition.lang = 'vi-VN';
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognition.onresult = (event: Event) => {
      const speechEvent = event as SpeechRecognitionEvent;
      let finalTranscript = '';
      let interim = '';

      for (let index = speechEvent.resultIndex; index < speechEvent.results.length; index += 1) {
        const result = speechEvent.results[index];
        const text = result[0]?.transcript ?? '';
        if (result.isFinal) {
          finalTranscript += text;
        } else {
          interim += text;
        }
      }

      if (finalTranscript) {
        setTranscript((current) => `${current} ${finalTranscript}`.trim());
      }
      setInterimTranscript(interim);
    };

    recognition.onerror = () => {
      setError('Speech Recognition chưa sẵn sàng trên trình duyệt này.');
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimTranscript('');
    };

    recognitionRef.current = recognition;
  }, [isSupported]);

  useEffect(() => () => {
    recognitionRef.current?.abort();
  }, []);

  const startListening = () => {
    if (!recognitionRef.current || isListening) {
      return;
    }

    setTranscript('');
    setInterimTranscript('');
    recognitionRef.current.start();
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
  };

  const resetTranscript = () => {
    setTranscript('');
    setInterimTranscript('');
    setError(null);
  };

  return {
    isSupported,
    isListening,
    transcript,
    interimTranscript,
    error,
    startListening,
    stopListening,
    resetTranscript,
  };
}

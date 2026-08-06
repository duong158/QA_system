export function supportsSpeechRecognition(): boolean {
  return typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);
}

export function supportsSpeechSynthesis(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

export function getAvailableVoices(): SpeechSynthesisVoice[] {
  if (!supportsSpeechSynthesis()) {
    return [];
  }

  return window.speechSynthesis.getVoices();
}

export function pickVietnameseVoice(voices: SpeechSynthesisVoice[], preferredName?: string): SpeechSynthesisVoice | undefined {
  const preferred = preferredName
    ? voices.find((voice) => voice.name === preferredName)
    : undefined;

  if (preferred) {
    return preferred;
  }

  return (
    voices.find((voice) => voice.lang.toLowerCase().startsWith('vi')) ||
    voices.find((voice) => voice.lang.toLowerCase().startsWith('en')) ||
    voices[0]
  );
}
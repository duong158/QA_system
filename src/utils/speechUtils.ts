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

const vietnameseHints = ['vi', 'vietnam', 'vietnamese', 'tieng viet'];
const preferredFemaleHints = [
  'hoaimy',
  'hoai my',
  'female',
  'vietnamese female',
  'linh',
  'mai',
  'trang',
  'chi',
  'yen',
  'my',
  'an',
  'zira',
  'aria',
  'jenny',
  'sara',
  'samantha',
  'susan',
  'victoria',
  'hazel',
  'helena',
];
const maleHints = [' male', '-male', ' nam ', ' nam)', '(nam', 'namminh', 'nam minh', 'duy'];

function normalizedText(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

function isVietnameseVoice(voice: SpeechSynthesisVoice): boolean {
  const name = normalizedText(voice.name);
  const lang = normalizedText(voice.lang);
  return lang.startsWith('vi') || vietnameseHints.some((hint) => name.includes(hint) || lang.includes(hint));
}

function isLikelyFemaleVoice(voice: SpeechSynthesisVoice): boolean {
  const name = ` ${normalizedText(voice.name)} `;
  return preferredFemaleHints.some((hint) => name.includes(hint));
}

function isLikelyMaleVoice(voice: SpeechSynthesisVoice): boolean {
  const name = ` ${normalizedText(voice.name)} `;
  return maleHints.some((hint) => name.includes(hint));
}

function scoreVoice(voice: SpeechSynthesisVoice): number {
  const lang = normalizedText(voice.lang);
  let score = 0;

  if (isVietnameseVoice(voice)) {
    score += 80;
  }
  if (isLikelyFemaleVoice(voice)) {
    score += 70;
  }
  if (isLikelyMaleVoice(voice)) {
    score -= 160;
  }
  if (voice.localService) {
    score += 3;
  }
  if (lang.startsWith('vi-vn')) {
    score += 8;
  }

  return score;
}

export function pickVietnameseVoice(voices: SpeechSynthesisVoice[], preferredName?: string): SpeechSynthesisVoice | undefined {
  const preferred = preferredName ? voices.find((voice) => voice.name === preferredName) : undefined;

  if (preferred && !isLikelyMaleVoice(preferred)) {
    return preferred;
  }

  const rankedVoices = [...voices].sort((a, b) => scoreVoice(b) - scoreVoice(a));
  return (
    rankedVoices.find((voice) => isVietnameseVoice(voice) && isLikelyFemaleVoice(voice)) ||
    rankedVoices.find((voice) => isLikelyFemaleVoice(voice) && !isLikelyMaleVoice(voice)) ||
    rankedVoices.find((voice) => isVietnameseVoice(voice) && !isLikelyMaleVoice(voice)) ||
    rankedVoices.find((voice) => !isLikelyMaleVoice(voice)) ||
    voices[0]
  );
}

export function sortVoicesForVietnameseAssistant(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice[] {
  return [...voices].sort((a, b) => scoreVoice(b) - scoreVoice(a) || a.name.localeCompare(b.name));
}

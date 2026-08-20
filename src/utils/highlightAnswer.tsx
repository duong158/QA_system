import type { ReactNode } from 'react';

export function highlightAnswer(text: string, answer?: string): ReactNode[] {
  if (!answer || !text) {
    return [text];
  }

  const index = text.toLowerCase().indexOf(answer.toLowerCase());
  if (index === -1) {
    return [text];
  }

  return [
    text.slice(0, index),
    <mark key="highlight" className="rounded bg-amber-400/30 px-1 text-[var(--text-primary)] ring-1 ring-amber-400/50">
      {text.slice(index, index + answer.length)}
    </mark>,
    text.slice(index + answer.length),
  ];
}

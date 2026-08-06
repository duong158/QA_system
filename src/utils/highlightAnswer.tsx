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
    <mark key="highlight" className="rounded-md bg-viqa-gold/20 px-1 text-viqa-gold ring-1 ring-viqa-gold/40">
      {text.slice(index, index + answer.length)}
    </mark>,
    text.slice(index + answer.length),
  ];
}

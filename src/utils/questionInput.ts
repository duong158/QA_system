function normalizedQuestion(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/gi, 'd')
    .toLocaleLowerCase('vi')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

export function collapseRepeatedQuestion(value: string): string {
  const words = value.replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  if (words.length >= 4 && words.length % 2 === 0) {
    const midpoint = words.length / 2;
    const first = words.slice(0, midpoint).join(' ');
    const second = words.slice(midpoint).join(' ');
    if (normalizedQuestion(first) === normalizedQuestion(second)) {
      return first;
    }
  }
  return words.join(' ');
}

export function mergeQuestionParts(...parts: Array<string | null | undefined>): string {
  const merged: string[] = [];
  for (const rawPart of parts) {
    const part = String(rawPart ?? '').replace(/\s+/g, ' ').trim();
    if (!part) {
      continue;
    }
    const key = normalizedQuestion(part);
    if (merged.some((existing) => normalizedQuestion(existing) === key)) {
      continue;
    }
    merged.push(part);
  }
  return collapseRepeatedQuestion(merged.join(' '));
}

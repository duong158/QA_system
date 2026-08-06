export function formatScore(score: number, fractionDigits = 1): string {
  return `${(score * 100).toFixed(fractionDigits)}%`;
}

export function formatLatency(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }

  return `${(ms / 1000).toFixed(2)} s`;
}
interface ConfidenceBadgeProps {
  confidence: number;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const isHigh = confidence >= 0.8;
  const isMedium = confidence >= 0.5;

  const className = isHigh
    ? 'border-emerald-400/25 bg-emerald-400/10 text-emerald-200'
    : isMedium
      ? 'border-amber-400/25 bg-amber-400/10 text-amber-200'
      : 'border-rose-400/25 bg-rose-400/10 text-rose-200';

  return <span className={`rounded-full border px-3 py-1 text-xs font-medium ${className}`}>{(confidence * 100).toFixed(1)}%</span>;
}
export function BackgroundEffects() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.10),transparent_35%),radial-gradient(circle_at_80%_30%,rgba(129,140,248,0.10),transparent_40%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(148,163,184,0.025)_1px,transparent_1px),linear-gradient(to_bottom,rgba(148,163,184,0.025)_1px,transparent_1px)] bg-[size:96px_96px] opacity-40" />
      <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-viqa-bg via-viqa-bg/60 to-transparent" />
    </div>
  );
}

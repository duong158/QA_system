export function BackgroundEffects() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <div className="absolute -right-24 -top-32 h-96 w-96 rounded-full bg-indigo-100/45 blur-3xl" />
      <div className="absolute -bottom-32 -left-24 h-80 w-80 rounded-full bg-violet-100/35 blur-3xl" />
    </div>
  );
}

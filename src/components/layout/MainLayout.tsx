import type { ReactNode } from 'react';
import { BackgroundEffects } from './BackgroundEffects';

interface MainLayoutProps { children: ReactNode; }

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="relative h-[100dvh] overflow-hidden text-[var(--text-primary)]">
      <BackgroundEffects />
      <div className="relative z-10 mx-auto flex h-full w-full max-w-[1536px] flex-col px-3 pb-3 pt-3 sm:px-4 sm:pb-4 sm:pt-4">
        {children}
      </div>
    </div>
  );
}

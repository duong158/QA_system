import type { ReactNode } from 'react';
import { BackgroundEffects } from './BackgroundEffects';

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="relative h-screen overflow-hidden bg-viqa-bg text-slate-50">
      <BackgroundEffects />
      <div className="relative z-10 mx-auto flex h-full w-full max-w-[1600px] flex-col px-4 pb-4 pt-4 lg:px-6">
        {children}
      </div>
    </div>
  );
}
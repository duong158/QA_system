import { Component, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  override componentDidCatch(error: Error): void {
    console.error('VIQA Nexus runtime error:', error);
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-viqa-bg px-6 text-center text-slate-100">
          <div className="max-w-xl rounded-3xl border border-rose-400/20 bg-rose-500/10 p-8 shadow-2xl shadow-rose-950/20 backdrop-blur-xl">
            <p className="font-display text-xs uppercase tracking-[0.35em] text-rose-200">System error</p>
            <h1 className="mt-4 text-3xl font-semibold">VIQA Nexus gặp lỗi khi tải giao diện.</h1>
            <p className="mt-3 text-sm text-slate-300">
              Hãy tải lại trang. Nếu lỗi vẫn còn, kiểm tra lại các file cấu hình hoặc console của trình duyệt.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
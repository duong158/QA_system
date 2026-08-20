import { Navigate, Route, Routes } from 'react-router-dom';
import { EvaluationPage } from '@/pages/EvaluationPage';
import { HomePage } from '@/pages/HomePage';
import { KnowledgeBlindSpotsPage } from '@/pages/KnowledgeBlindSpotsPage';
import { ErrorBoundary } from '@/components/layout/ErrorBoundary';

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/knowledge-blind-spots" element={<KnowledgeBlindSpotsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ErrorBoundary>
  );
}

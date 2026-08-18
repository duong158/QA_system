import { AnimatePresence, motion } from 'framer-motion';
import { Palette, Volume2, Cpu, DatabaseZap, Brain, Play } from 'lucide-react';
import { useAppStore } from '@/store/appStore';
import type { ReaderType, RetrieverType } from '@/types/qa';
import { sortVoicesForVietnameseAssistant } from '@/utils/speechUtils';

interface SettingsPanelProps {
  voices: SpeechSynthesisVoice[];
  onTestVoice?: () => void;
}

const retrieverOptions: Array<{ value: RetrieverType; label: string; description?: string; implemented: boolean }> = [
  { value: 'tfidf', label: 'TF-IDF', implemented: true },
  { value: 'bm25', label: 'BM25', implemented: true },
  { value: 'hybrid', label: 'Hybrid (BM25 + Dense)', description: 'BM25 + Dense kết hợp bằng Reciprocal Rank Fusion', implemented: true },
  { value: 'dense', label: 'Dense Retrieval', implemented: true },
  { value: 'pyserini', label: 'Pyserini BM25', implemented: false },
];

const readerOptions: Array<{ value: ReaderType; label: string; implemented: boolean }> = [
  { value: 'phobert', label: 'PhoBERT QA', implemented: true },
  { value: 'vibert', label: 'viBERT QA', implemented: false },
  { value: 'xlmr', label: 'XLM-R QA', implemented: true },
];

const topKOptions = [1, 3, 5, 10];
const avatarModelName = import.meta.env.VITE_AVATAR_MODEL_NAME || 'Mari 3D VRoid Model';
const avatarCreatorName = import.meta.env.VITE_AVATAR_CREATOR_NAME || 'Creator name not configured';
const avatarLicense = import.meta.env.VITE_AVATAR_LICENSE || 'Free to use with credit';

export function SettingsPanel({ voices, onTestVoice }: SettingsPanelProps) {
  const open = useAppStore((state) => state.isSettingsOpen);
  const settings = useAppStore((state) => state.settings);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const updateVoiceSettings = useAppStore((state) => state.updateVoiceSettings);
  const updateDisplaySettings = useAppStore((state) => state.updateDisplaySettings);
  const setSettingsOpen = useAppStore((state) => state.setSettingsOpen);
  const sortedVoices = sortVoicesForVietnameseAssistant(voices);

  return (
    <AnimatePresence>
      {open ? (
        <motion.aside
          initial={{ x: 420, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 420, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 120, damping: 20 }}
          className="fixed right-0 top-0 z-40 h-full w-full max-w-[420px] border-l border-white/10 bg-[rgba(4,7,17,0.96)] p-4 shadow-2xl shadow-black/40 backdrop-blur-2xl"
        >
          <div className="flex h-full flex-col gap-4 overflow-y-auto rounded-[28px] border border-white/10 bg-white/5 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Settings</p>
                <h2 className="mt-1 font-display text-lg tracking-[0.18em] text-white">SYSTEM CONTROL</h2>
              </div>
              <button type="button" onClick={() => setSettingsOpen(false)} className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-xs text-slate-300">
                Close
              </button>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm text-white">
                  <DatabaseZap className="h-4 w-4 text-viqa-cyan" /> Retriever
                </div>
                <div className="grid gap-2">
                  {retrieverOptions.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      disabled={!option.implemented}
                      onClick={() => option.implemented && updateSettings({ retriever: option.value })}
                      className={`rounded-xl border px-3 py-2 text-left text-sm transition ${
                        settings.retriever === option.value
                          ? 'border-viqa-cyan/30 bg-viqa-cyan/10 text-viqa-cyan'
                          : option.implemented
                            ? 'border-white/10 bg-white/5 text-slate-200 hover:border-viqa-cyan/25'
                            : 'cursor-not-allowed border-white/5 bg-white/[0.03] text-slate-500'
                      }`}
                    >
                      <span className="flex items-center justify-between gap-3">
                        <span>
                          <span>{option.label}</span>
                          {option.description ? <span className="mt-0.5 block text-[10px] leading-tight text-slate-400">{option.description}</span> : null}
                        </span>
                        {!option.implemented ? <span className="rounded-full border border-slate-500/20 px-2 py-0.5 text-[10px] uppercase tracking-[0.16em]">Coming soon</span> : null}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm text-white">
                  <Brain className="h-4 w-4 text-viqa-violet" /> Reader
                </div>
                <div className="grid gap-2">
                  {readerOptions.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      disabled={!option.implemented}
                      onClick={() => option.implemented && updateSettings({ reader: option.value })}
                      className={`rounded-xl border px-3 py-2 text-left text-sm transition ${
                        settings.reader === option.value
                          ? 'border-viqa-violet/30 bg-viqa-violet/10 text-viqa-violet'
                          : option.implemented
                            ? 'border-white/10 bg-white/5 text-slate-200 hover:border-viqa-violet/25'
                            : 'cursor-not-allowed border-white/5 bg-white/[0.03] text-slate-500'
                      }`}
                    >
                      <span className="flex items-center justify-between gap-3">
                        <span>{option.label}</span>
                        {!option.implemented ? <span className="rounded-full border border-slate-500/20 px-2 py-0.5 text-[10px] uppercase tracking-[0.16em]">Coming soon</span> : null}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm text-white">
                  <Cpu className="h-4 w-4 text-viqa-gold" /> Top-k
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {topKOptions.map((value) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => updateSettings({ topK: value })}
                      className={`rounded-xl border px-3 py-2 text-sm transition ${
                        settings.topK === value ? 'border-viqa-gold/30 bg-viqa-gold/10 text-viqa-gold' : 'border-white/10 bg-white/5 text-slate-200'
                      }`}
                    >
                      {value}
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm text-white">
                  <Volume2 className="h-4 w-4 text-emerald-300" /> Voice
                </div>
                <label className="flex items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200">
                  <span>Bật đọc câu trả lời</span>
                  <input
                    type="checkbox"
                    checked={settings.voice.enabled}
                    onChange={(event) => updateVoiceSettings({ enabled: event.target.checked })}
                  />
                </label>
                <div className="mt-3 grid gap-3 text-sm text-slate-300">
                  <label className="grid gap-2">
                    <span>Voice</span>
                    <select
                      value={settings.voice.voiceName ?? ''}
                      onChange={(event) => updateVoiceSettings({ voiceName: event.target.value || undefined })}
                      className="rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-slate-100 outline-none"
                    >
                      <option value="">Auto-select Vietnamese female</option>
                      {sortedVoices.map((voice) => (
                        <option key={voice.name} value={voice.name}>
                          {voice.name} ({voice.lang})
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="grid gap-2">
                    <span>Rate: {settings.voice.rate.toFixed(1)}</span>
                    <input
                      type="range"
                      min="0.6"
                      max="1.4"
                      step="0.1"
                      value={settings.voice.rate}
                      onChange={(event) => updateVoiceSettings({ rate: Number(event.target.value) })}
                    />
                  </label>
                  <label className="grid gap-2">
                    <span>Pitch: {settings.voice.pitch.toFixed(1)}</span>
                    <input
                      type="range"
                      min="0.6"
                      max="1.4"
                      step="0.1"
                      value={settings.voice.pitch}
                      onChange={(event) => updateVoiceSettings({ pitch: Number(event.target.value) })}
                    />
                  </label>
                  <label className="grid gap-2">
                    <span>Volume: {settings.voice.volume.toFixed(1)}</span>
                    <input
                      type="range"
                      min="0.2"
                      max="1"
                      step="0.1"
                      value={settings.voice.volume}
                      onChange={(event) => updateVoiceSettings({ volume: Number(event.target.value) })}
                    />
                  </label>
                  <button
                    type="button"
                    onClick={onTestVoice}
                    disabled={!onTestVoice}
                    className="inline-flex items-center justify-center gap-2 rounded-xl border border-emerald-300/20 bg-emerald-300/10 px-3 py-2 text-sm font-medium text-emerald-200 transition hover:border-emerald-300/40 hover:bg-emerald-300/15 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Play className="h-4 w-4" />
                    Test voice
                  </button>
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm text-white">
                  <Palette className="h-4 w-4 text-viqa-cyan" /> Display
                </div>
                <div className="grid gap-2 text-sm text-slate-200">
                  {[
                    ['particlesEnabled', 'Particles'],
                    ['hologramEnabled', 'Hologram'],
                    ['advancedMotion', 'Advanced motion'],
                    ['lowPerformanceMode', 'Low performance'],
                  ].map(([key, label]) => (
                    <label key={key} className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                      <span>{label}</span>
                      <input
                        type="checkbox"
                        checked={settings.display[key as keyof typeof settings.display]}
                        onChange={(event) =>
                          updateDisplaySettings({ [key]: event.target.checked } as Partial<typeof settings.display>)
                        }
                      />
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 p-4 text-xs leading-6 text-slate-400">
              <p className="text-slate-200">Avatar Model: {avatarModelName}</p>
              <p>Creator: {avatarCreatorName}</p>
              <p>License: {avatarLicense}</p>
              <p>Model redistribution: Prohibited</p>
            </div>

            <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-4 text-xs leading-6 text-slate-400">
              Cấu hình được lưu trong <span className="text-slate-200">localStorage</span> để giữ trạng thái khi demo.
            </div>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
}

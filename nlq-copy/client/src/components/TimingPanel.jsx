import { X } from 'lucide-react';
import { humanStageLabel, formatDuration } from '../utils/helpers';

export default function TimingPanel({ open, onClose, timingState }) {
  if (!open) return null;

  const entries = timingState?.entries || [];
  let sum = 0;
  const runningMs = timingState?.currentStage && !timingState?.done
    ? Math.max(0, performance.now() - (timingState.currentStart || 0))
    : 0;

  return (
    <aside className="timing-panel open">
      <div className="timing-header">
        <span>Query Timing</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text3)' }}>
            {!timingState ? 'Idle' : timingState.done
              ? `${formatDuration(sum)} (${timingState.status || 'done'})`
              : `${formatDuration(sum + runningMs)}+`}
          </span>
          <button className="timing-close" onClick={onClose}>
            <X size={12} />
          </button>
        </div>
      </div>
      <div className="timing-list">
        {entries.length === 0 && !timingState?.currentStage ? (
          <div className="timing-row">
            <span className="label">No active query</span>
            <span className="value">-</span>
          </div>
        ) : (
          <>
            {entries.map((entry, i) => {
              sum += entry.ms;
              return (
                <div key={i} className="timing-row">
                  <span className="label">{humanStageLabel(entry.stage)}</span>
                  <span className="value">{formatDuration(entry.ms)}</span>
                </div>
              );
            })}
            {timingState?.currentStage && !timingState.done && (
              <div className="timing-row running">
                <span className="label">{humanStageLabel(timingState.currentStage)}</span>
                <span className="value">{formatDuration(runningMs)}</span>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}

import { HelpCircle, CornerDownRight } from 'lucide-react';
import { escHtml } from '../utils/helpers';

export default function ClarificationBar({ data, onApply }) {
  if (!data) return null;

  const message = String(data?.narrative || '').trim() || 'Please confirm your request.';
  const suggestions = Array.isArray(data?.followUps) ? data.followUps.filter(Boolean).slice(0, 3) : [];

  return (
    <div className="clarification-bar">
      <div className="clarification-inner">
        <div className="clarification-label">
          <HelpCircle size={12} />
          Query rewrite
        </div>
        <div className="clarification-text">{escHtml(message)}</div>
        {suggestions.length > 0 && (
          <div className="clarification-actions">
            {suggestions.map((q, i) => (
              <button key={i} className="clarification-chip" onClick={() => onApply(q)}>
                <CornerDownRight size={13} />
                <span>{escHtml(q)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

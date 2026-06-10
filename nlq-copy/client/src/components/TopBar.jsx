import { Brain } from 'lucide-react';

export default function TopBar({
  onClearChat, timingPanelOpen, onToggleTiming,
  contextInfo, selectedModelLabel,
  responseTimer,
}) {
  const { limit, used, percentLeft } = contextInfo || {};
  const colorStyle = percentLeft > 50 ? 'var(--green)' : percentLeft > 20 ? 'var(--amber)' : 'var(--red)';

  return (
    <div className="topbar">
      <button className="topbar-btn" onClick={onClearChat}>Clear Chat</button>
      <button className="topbar-btn" onClick={onToggleTiming}>
        {timingPanelOpen ? 'Hide Timing' : 'Show Timing'}
      </button>
      <div className="context-badge">
        <Brain size={13} style={{ color: 'var(--accent2)', flexShrink: 0 }} />
        <span style={{ color: 'var(--text3)', fontWeight: 500, fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>Context:</span>
        <span style={{ fontFamily: 'var(--font-mono)', color: colorStyle, fontWeight: 600, marginLeft: 4 }}>
          {percentLeft != null ? `${percentLeft}% left` : '100% left'}
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text3)', fontSize: 11 }}>
          ({used != null ? used.toLocaleString() : '0'} used / {limit ? limit >= 1000 ? (limit / 1000) + 'K' : limit : '258K'} · {selectedModelLabel || 'Selected model'})
        </span>
      </div>
      <div className="topbar-title">Natural Language Query</div>
      <div className="topbar-status">
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text2)', marginRight: 8 }}>{responseTimer}</span>
        <div className="status-dot" />
        <span id="status-text">Connected</span>
      </div>
    </div>
  );
}

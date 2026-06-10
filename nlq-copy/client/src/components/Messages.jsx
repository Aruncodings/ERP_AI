import { Search, Bot, MessageSquare, Zap } from 'lucide-react';
import { escHtml, parseNarrativeMarkdown } from '../utils/helpers';

export default function WelcomeBlock() {
  return (
    <div className="welcome-block welcome-enter">
      <div className="welcome-icon welcome-icon-pulse">
        <Search size={26} />
      </div>
      <div className="welcome-title">Ask your ERP anything</div>
      <div className="welcome-sub">
        Type in plain English. Get data, charts, insights — instantly generated from your live MongoDB collections.
      </div>
    </div>
  );
}

export function UserMessage({ content, index }) {
  return (
    <div className="msg-row user msg-enter" style={{ animationDelay: `${(index || 0) * 0.08}s` }}>
      <div className="msg-content">
        <div className="user-bubble user-bubble-dynamic">{escHtml(content)}</div>
      </div>
      <div className="msg-avatar user-av">
        <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>
      </div>
    </div>
  );
}

export function ThinkingMessage({ step, reasoning, answer }) {
  const hasStreamContent = reasoning || answer;

  return (
    <div className="msg-row msg-enter">
      <div className="msg-avatar">
        <Bot size={16} />
      </div>
      <div className="msg-content" style={{ maxWidth: 440 }}>
        <div className="thinking-block thinking-block-dynamic">
          <div className="claude-loader">
            <span className="claude-dot" />
            <span className="claude-dot" />
            <span className="claude-dot" />
          </div>
          <div className="thinking-step-text">{step || 'Processing...'}</div>
        </div>
        {hasStreamContent && (
          <div className="thinking-stream-panel thinking-stream-enter">
            {reasoning && (
              <div className="thinking-section">
                <div className="thinking-section-label">
                  <Zap size={10} /> Reasoning
                </div>
                <div className="thinking-section-text thinking-section-reasoning" dangerouslySetInnerHTML={{ __html: parseNarrativeMarkdown(reasoning) }} />
              </div>
            )}
            {answer && (
              <div className="thinking-section">
                <div className="thinking-section-label thinking-section-label-answer">
                  <MessageSquare size={10} /> Answer Draft
                </div>
                <div className="thinking-section-text thinking-section-answer" dangerouslySetInnerHTML={{ __html: parseNarrativeMarkdown(answer) }} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function ErrorMessage({ content }) {
  return (
    <div className="msg-row msg-enter">
      <div className="msg-avatar">
        <MessageSquare size={16} />
      </div>
      <div className="msg-content">
        <div className="error-block error-block-dynamic">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0 }}>
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>{escHtml(content)}</span>
        </div>
      </div>
    </div>
  );
}

import { useState, useRef, useCallback } from 'react';
import { Send } from 'lucide-react';

export default function InputArea({ value, onChange, onSend, isLoading }) {
  const textareaRef = useRef(null);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }, []);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  }, [onSend]);

  return (
    <div className="input-area">
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          className="input-box"
          placeholder="Ask anything... e.g. Show me all approved purchase orders as a bar chart"
          rows={1}
          value={value}
          onChange={(e) => { onChange(e.target.value); autoResize(); }}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        <button className="send-btn" onClick={onSend} disabled={isLoading || !value.trim()}>
          <Send size={16} />
        </button>
      </div>
      <div className="input-hint">Enter to send · Shift+Enter for new line</div>
    </div>
  );
}

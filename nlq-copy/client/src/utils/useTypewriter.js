import { useState, useEffect, useRef, useCallback } from 'react';

export default function useTypewriter(text, { speed = 20, enabled = true } = {}) {
  const [displayed, setDisplayed] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const controllerRef = useRef(null);
  const textRef = useRef(text);

  useEffect(() => {
    textRef.current = text;
  }, [text]);

  const startTyping = useCallback(() => {
    if (!text || !enabled) {
      setDisplayed(text || 'No data returned.');
      setIsComplete(true);
      return;
    }

    setIsComplete(false);
    setDisplayed('');

    const chars = [...text];
    let index = 0;
    let paused = false;
    let timeoutId = null;

    function getDelay() {
      const char = chars[index - 1];
      if (!char) return speed;
      if (char === '.' || char === '!' || char === '?') return speed * 6;
      if (char === ',' || char === ';' || char === ':') return speed * 3;
      if (char === '\n') return speed * 4;
      if (char === ' ') return speed * 0.5;
      return speed;
    }

    function tick() {
      if (paused) return;
      if (index < chars.length) {
        setDisplayed(chars.slice(0, index + 1).join(''));
        index++;
        timeoutId = setTimeout(tick, getDelay());
      } else {
        setIsComplete(true);
      }
    }

    tick();

    controllerRef.current = {
      pause: () => { paused = true; if (timeoutId) clearTimeout(timeoutId); },
      resume: () => { paused = false; tick(); },
      cancel: () => { paused = true; if (timeoutId) clearTimeout(timeoutId); },
      isComplete: () => index >= chars.length,
    };

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [text, speed, enabled]);

  useEffect(() => {
    const cleanup = startTyping();
    return cleanup;
  }, [startTyping]);

  return { displayed, isComplete, controller: controllerRef.current };
}

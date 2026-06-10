export function typeText(text, onChar, onComplete, speed = 20) {
  let index = 0;
  const chars = [...text];
  let paused = false;
  let timeoutId = null;

  function tick() {
    if (paused) return;
    if (index < chars.length) {
      onChar(chars.slice(0, index + 1).join(''));
      index++;
      const variableSpeed = getVariableDelay(chars, index, speed);
      timeoutId = setTimeout(tick, variableSpeed);
    } else {
      if (onComplete) onComplete();
    }
  }

  tick();

  return {
    pause: () => { paused = true; if (timeoutId) clearTimeout(timeoutId); },
    resume: () => { paused = false; tick(); },
    cancel: () => { paused = true; if (timeoutId) clearTimeout(timeoutId); },
    isComplete: () => index >= chars.length,
    getProgress: () => index / chars.length,
  };
}

function getVariableDelay(chars, index, baseSpeed) {
  const char = chars[index - 1];
  if (!char) return baseSpeed;
  if (char === '.' || char === '!' || char === '?') return baseSpeed * 6;
  if (char === ',' || char === ';' || char === ':') return baseSpeed * 3;
  if (char === '\n') return baseSpeed * 4;
  if (char === ' ') return baseSpeed * 0.5;
  return baseSpeed;
}

export function staggerArray(items, onItem, onComplete, delay = 80) {
  let index = 0;
  const interval = setInterval(() => {
    if (index < items.length) {
      onItem(items[index], index);
      index++;
    } else {
      clearInterval(interval);
      if (onComplete) onComplete();
    }
  }, delay);
  return () => clearInterval(interval);
}

export function animateValue(start, end, onValue, onComplete, duration = 600) {
  const startTime = performance.now();

  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * eased);
    onValue(current);
    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      if (onComplete) onComplete();
    }
  }

  requestAnimationFrame(tick);
}

export function useStaggeredVisibility(count, delay = 80) {
  const visible = new Array(count).fill(false);

  function start(onComplete) {
    let i = 0;
    const interval = setInterval(() => {
      if (i < count) {
        visible[i] = true;
        i++;
      } else {
        clearInterval(interval);
        if (onComplete) onComplete();
      }
    }, delay);
    return () => clearInterval(interval);
  }

  return { visible, start };
}

export const transitions = {
  fadeUp: {
    initial: { opacity: 0, transform: 'translateY(12px)' },
    animate: { opacity: 1, transform: 'translateY(0)' },
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
  },
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    transition: { duration: 0.3 },
  },
  scaleIn: {
    initial: { opacity: 0, transform: 'scale(0.95)' },
    animate: { opacity: 1, transform: 'scale(1)' },
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
  slideUp: {
    initial: { opacity: 0, transform: 'translateY(20px)' },
    animate: { opacity: 1, transform: 'translateY(0)' },
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] },
  },
};

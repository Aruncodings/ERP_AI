import { useMemo } from 'react';

const STAR_COUNT = 60;

export default function StarsBackground() {
  const stars = useMemo(() => {
    return Array.from({ length: STAR_COUNT }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      top: Math.random() * 100,
      size: 1 + Math.random() * 2,
      dur: 2 + Math.random() * 4,
      delay: Math.random() * 5,
    }));
  }, []);

  return (
    <div className="stars-bg">
      {stars.map(s => (
        <div
          key={s.id}
          className="star"
          style={{
            left: `${s.left}%`,
            top: `${s.top}%`,
            width: s.size,
            height: s.size,
            opacity: 0.5,
            '--dur': `${s.dur}s`,
            '--delay': `${s.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

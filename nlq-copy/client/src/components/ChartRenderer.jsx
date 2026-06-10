import { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import { CHART_COLORS, CHART_BORDERS, escHtml } from '../utils/helpers';

Chart.register(...registerables);

Chart.defaults.color = '#a0a0b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
Chart.defaults.font.family = "'Inter', sans-serif";

export default function ChartRenderer({ config }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !config) return;
    if (chartRef.current) chartRef.current.destroy();

    const isPolar = ['pie', 'doughnut'].includes(config.type);
    const datasets = (config.datasets || []).map((ds, i) => ({
      label: ds.label || 'Value',
      data: ds.data || [],
      backgroundColor: isPolar
        ? (ds.data || []).map((_, j) => CHART_COLORS[j % CHART_COLORS.length])
        : CHART_COLORS[i % CHART_COLORS.length],
      borderColor: isPolar
        ? (ds.data || []).map((_, j) => CHART_BORDERS[j % CHART_BORDERS.length])
        : CHART_BORDERS[i % CHART_BORDERS.length],
      borderWidth: isPolar ? 1 : 2,
      borderRadius: config.type === 'bar' ? 4 : 0,
      tension: 0.4,
      fill: config.type === 'line' ? { target: 'origin', above: 'rgba(124,106,247,0.08)' } : false,
      pointBackgroundColor: '#7c6af7',
      pointRadius: 3,
      pointHoverRadius: 5,
    }));

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current = new Chart(ctx, {
      type: config.type || 'bar',
      data: {
        labels: config.labels || [],
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: datasets.length > 1 || isPolar,
            labels: { color: '#a0a0b8', font: { size: 11 }, boxWidth: 12, padding: 16 },
          },
          tooltip: {
            backgroundColor: '#1e1e28',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            titleColor: '#f0f0f8',
            bodyColor: '#a0a0b8',
            padding: 10,
            cornerRadius: 8,
          },
        },
        scales: isPolar ? {} : {
          x: { ticks: { color: '#606070', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#606070', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        },
        animation: { duration: 600, easing: 'easeOutQuart' },
      },
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [config]);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div className="chart-title">{escHtml(config?.title || 'Chart')}</div>
        <div className="chart-type-badge">{escHtml(config?.type || 'bar')}</div>
      </div>
      <div className="chart-container">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}

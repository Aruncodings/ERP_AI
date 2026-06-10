export function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function estimateTokens(text) {
  if (!text) return 0;
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  return Math.ceil(str.length / 3.7);
}

export function formatDuration(ms) {
  const value = Number(ms || 0);
  if (!Number.isFinite(value) || value <= 0) return '0 s';
  if (value < 1000) return `${(value / 1000).toFixed(2)} s`;
  const totalSeconds = Math.round(value / 1000);
  if (totalSeconds < 60) return `${totalSeconds} s`;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (totalMinutes < 60) return `${totalMinutes}m ${seconds}s`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `${hours}h ${minutes}m`;
}

export function formatTokenLimit(limit) {
  if (limit >= 1000) return (limit / 1000) + 'K';
  return limit.toString();
}

export function formatModelLabel(value) {
  const text = String(value || '');
  if (!text) return '';
  const normalized = text.replaceAll('\\', '/');
  const hfSnapshotMatch = normalized.match(/models--([^/]+)--([^/]+)\/snapshots\/([^/]+)/i);
  if (hfSnapshotMatch) {
    return `${hfSnapshotMatch[1]}/${hfSnapshotMatch[2]} @ ${hfSnapshotMatch[3].slice(0, 8)}`;
  }
  const hfRepoMatch = normalized.match(/models--([^/]+)--([^/]+)/i);
  if (hfRepoMatch) return `${hfRepoMatch[1]}/${hfRepoMatch[2]}`;
  const parts = normalized.split('/').filter(Boolean);
  const last = parts.length ? parts[parts.length - 1] : text;
  const isHashLike = /^[0-9a-f]{7,}$/i.test(last) || /^\d+$/.test(last);
  if (isHashLike && parts.length > 1) return parts[parts.length - 2];
  return last || text;
}

export function humanStageLabel(stage) {
  const labels = {
    ui_start: 'UI Loading',
    planning_started: 'Backend Bootstrap',
    llm_loading: 'LLM: Model Loading',
    single_pass_routing: 'LLM: Single-Pass Routing',
    prompt_rewriting: 'LLM: Prompt Rewrite',
    prompt_rewritten: 'LLM: Rewrite Ready',
    prompt_validated: 'LLM: Routing and Planning',
    backend_validation: 'Backend Validation',
    query_generated: 'Backend Plan Ready',
    query_executing: 'Backend Query Execute',
    exact_lookup: 'Backend Exact Value Cache',
    query_execute_core: 'Backend Execute Core Query',
    lookup_resolving: 'Backend Resolve Lookup Values',
    repair_empty_result: 'Backend Empty-Result Repair',
    retry_alt_collection: 'Backend Retry Alternate Collection',
    retry_fallback_aggregate: 'Backend Retry Aggregate Fallback',
    repair_join_nulls: 'Backend Join-Null Repair',
    result_verifying: 'LLM: Result Verification',
    repair_verifier_mismatch: 'LLM: Verifier Mismatch Repair',
    rows_fetched: 'Backend Rows Fetched',
    narrating: 'LLM: Answer Streaming',
    ui_rendering: 'UI Rendering',
  };
  return labels[stage] || String(stage || 'processing').replace(/_/g, ' ');
}

export function parseNarrativeMarkdown(text) {
  let escaped = escHtml(text);
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  escaped = escaped.replace(/\[(.*?)\]\(suggest:(.*?)\)/g, (match, label, query) => {
    const cleanQuery = query.replace(/'/g, "\\'").replace(/"/g, "&quot;");
    return `<button class="suggest-btn" data-query="${cleanQuery}"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>${escHtml(label)}</button>`;
  });
  return escaped;
}

export function generateConversationId() {
  return `nlq-web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export const CHART_COLORS = [
  'rgba(124,106,247,0.85)', 'rgba(34,211,238,0.85)', 'rgba(52,211,153,0.85)',
  'rgba(251,191,36,0.85)', 'rgba(248,113,113,0.85)', 'rgba(167,139,250,0.85)',
  'rgba(56,189,248,0.85)', 'rgba(74,222,128,0.85)', 'rgba(253,186,116,0.85)',
  'rgba(249,168,212,0.85)',
];

export const CHART_BORDERS = CHART_COLORS.map(c => c.replace('0.85', '1'));

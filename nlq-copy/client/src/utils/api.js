function extractTableDataFromRaw(rawData, tableColumns) {
  if (!Array.isArray(rawData) || rawData.length === 0) {
    return { columns: [], rows: [] };
  }

  const isBlankCell = (value) => {
    if (value === null || value === undefined) return true;
    const text = String(value).trim();
    return !text || text === '—' || text === '-';
  };

  let columns = Array.isArray(tableColumns) ? tableColumns.map(c => String(c || '').trim()).filter(Boolean) : [];
  if (!columns.length) {
    const keys = new Set();
    rawData.forEach(item => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return;
      Object.keys(item).forEach(key => {
        if (key !== '_id') keys.add(key);
      });
    });
    columns = Array.from(keys);
  }

  const rows = rawData.map(item => {
    if (Array.isArray(item)) {
      return item.map(value => Array.isArray(value) ? value.join(', ') : value);
    }
    return columns.map(column => {
      const value = item?.[column];
      if (Array.isArray(value)) return value.join(', ');
      if (value && typeof value === 'object') return JSON.stringify(value);
      return value;
    });
  }).filter(row => Array.isArray(row) && row.some(cell => !isBlankCell(cell)));

  return { columns, rows };
}

export function mapFastApiResult(payload) {
  const rawDocs = Array.isArray(payload.docs)
    ? payload.docs
    : (Array.isArray(payload.rawData) ? payload.rawData : []);
  const preferredSourceColumns = Array.isArray(payload.table_columns)
    ? payload.table_columns
    : (Array.isArray(payload.tableColumns) ? payload.tableColumns : []);
  const preferredColumns = preferredSourceColumns.map(c => String(c || '').trim()).filter(Boolean);
  const { columns: derivedColumns, rows: derivedRows } = extractTableDataFromRaw(rawDocs, preferredColumns);

  const flatRows = Array.isArray(payload.table_rows)
    ? payload.table_rows
    : (Array.isArray(payload.tableRows) ? payload.tableRows : []);

  const rows = flatRows.length
    ? flatRows
    : derivedRows;
  const columns = preferredColumns.length ? preferredColumns : derivedColumns;
  return {
    narrative: payload.response || 'No response.',
    summary: payload.summary || '',
    insights: Array.isArray(payload.insights) ? payload.insights : [],
    chartConfig: payload.chart_config || payload.chartConfig || null,
    dbPerformance: payload.db_performance || payload.dbPerformance || null,
    tableColumns: columns,
    tableRows: rows,
    followUps: Array.isArray(payload.follow_ups) ? payload.follow_ups : [],
    needsClarification: Boolean(payload.needs_clarification),
    pipeline: payload.plan || {},
    tableChoice: payload.table_choice || {},
    collectionName: payload.collection || '',
    totalRecords: Number(payload.total || rawDocs.length),
    rawData: rawDocs,
  };
}

function getApiBase() {
  try {
    return (localStorage.getItem('nlq-api-base') || 'http://127.0.0.1:8000').replace(/\/+$/, '');
  } catch {
    return 'http://127.0.0.1:8000';
  }
}

export function setApiBase(url) {
  localStorage.setItem('nlq-api-base', url);
}

export function apiUrl(path) {
  return `${getApiBase()}${path}`;
}

export async function fetchJSON(path, options = {}) {
  const res = await fetch(apiUrl(path), {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(data.error || data.detail || `Server returned ${res.status}`);
  }
  return res.json();
}

export function fetchNDJSON(url, options, callbacks) {
  const { onProgress, onComplete, onError, onToken, onState } = callbacks || {};
  (async () => {
    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        const data = await res.json().catch(() => ({ error: res.statusText }));
        onError?.(data.error || `Server returned ${res.status}`);
        return;
      }
      const contentType = res.headers.get('content-type') || '';
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      if (contentType.includes('text/event-stream')) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split('\n\n');
          buffer = blocks.pop() || '';
          for (const block of blocks) {
            const eventLine = block.split('\n').find(x => x.startsWith('event: '));
            const dataLine = block.split('\n').find(x => x.startsWith('data: '));
            if (!eventLine || !dataLine) continue;
            const eventName = eventLine.slice(7).trim();
            let payload = {};
            try { payload = JSON.parse(dataLine.slice(6)); } catch { continue; }
            if (eventName === 'status') onProgress?.(payload.stage || 'processing', payload.message || '');
            else if (eventName === 'llm_state') onState?.(payload.state || 'ANSWER');
            else if (eventName === 'llm_token') onToken?.(payload.state || 'ANSWER', payload.token || '');
            else if (eventName === 'error') { onError?.(payload.detail || 'Server error'); return; }
            else if (eventName === 'done') { onComplete?.(mapFastApiResult(payload)); return; }
          }
        }
        return;
      }
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const chunk = JSON.parse(line);
          if (chunk.status === 'error') return onError?.(chunk.message || chunk.error || 'Server error');
          if (chunk.status === 'complete') return onComplete?.(mapFastApiResult(chunk));
          if (onProgress) onProgress(chunk.status, chunk.message);
        }
      }
    } catch (e) {
      onError?.('Network error: ' + e.message);
    }
  })();
}

export async function getTrainingStatus() {
  return fetchJSON('/train');
}

export async function triggerTraining() {
  return fetchJSON('/train', { method: 'POST' });
}

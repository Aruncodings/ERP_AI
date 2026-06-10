import { useEffect, useMemo, useState, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Download, Database, Filter } from 'lucide-react';
import { escHtml } from '../utils/helpers';

const RECORDS_LIMIT = 10;

export default function DataTable({ containerId, columns, rows, total, currentQuestion, collectionName, rawData }) {
  const [page, setPage] = useState(1);
  const isBlankCell = (value) => {
    if (value === null || value === undefined) return true;
    const text = String(value).trim();
    return !text || text === '—' || text === '-';
  };
  const normalizedColumns = useMemo(() => (
    Array.isArray(columns) ? columns.filter(Boolean) : []
  ), [columns]);
  const normalizedRows = useMemo(() => (
    Array.isArray(rows) ? rows : []
  ), [rows]);
  const inferredColumns = useMemo(() => (
    normalizedColumns.length > 0
      ? normalizedColumns
      : (Array.isArray(rawData) && rawData.length > 0 && rawData[0] && typeof rawData[0] === 'object'
          ? Object.keys(rawData[0]).filter(key => key !== '_id')
          : [])
  ), [normalizedColumns, rawData]);
  const inferredRows = useMemo(() => (
    normalizedRows.length > 0
      ? normalizedRows
      : (Array.isArray(rawData) && rawData.length > 0 && inferredColumns.length > 0
          ? rawData.map(row => {
              if (Array.isArray(row)) {
                return row.map(cell => Array.isArray(cell) ? cell.join(', ') : cell);
              }
              return inferredColumns.map(col => {
                const value = row?.[col];
                if (Array.isArray(value)) return value.join(', ');
                if (value && typeof value === 'object') return JSON.stringify(value);
                return value;
              });
            })
          : [])
  ), [normalizedRows, rawData, inferredColumns]);
  const totalRowCount = Math.max(Number(total || 0), inferredRows.length);

  useEffect(() => {
    setPage(1);
  }, [containerId, currentQuestion, collectionName, inferredColumns, inferredRows]);

  const getFilterableCols = useCallback(() => {
    if (!inferredRows?.length) return [];
    const filterable = [];
    inferredColumns.forEach((col, colIdx) => {
      const uniqueVals = new Set();
      inferredRows.forEach(row => {
        const val = row[colIdx];
        if (val !== undefined && val !== null && String(val).trim() !== '') {
          uniqueVals.add(String(val).trim());
        }
      });
      if (uniqueVals.size >= 2 && uniqueVals.size <= 8) {
        filterable.push({ col, colIdx, values: Array.from(uniqueVals).sort() });
      }
    });
    return filterable;
  }, [inferredColumns, inferredRows]);

  const filterableCols = getFilterableCols();

  const [filters, setFilters] = useState({});

  const filteredRows = useMemo(() => inferredRows.filter(row => {
    for (const [colIdx, filterVal] of Object.entries(filters)) {
      if (filterVal && String(row[Number(colIdx)] ?? '').trim() !== filterVal) return false;
    }
    return true;
  }), [filters, inferredRows]);

  const filteredTotal = filteredRows.length;
  const effectiveTotal = filters && Object.keys(filters).some(key => filters[key]) ? filteredTotal : totalRowCount;
  const effectiveTotalPages = Math.max(1, Math.ceil(Math.max(filteredTotal, 1) / RECORDS_LIMIT));

  useEffect(() => {
    if (page > effectiveTotalPages) {
      setPage(1);
    }
  }, [page, effectiveTotalPages]);

  const currentPageRows = useMemo(() => {
    const start = (page - 1) * RECORDS_LIMIT;
    return filteredRows.slice(start, start + RECORDS_LIMIT);
  }, [filteredRows, page]);

  const startIdx = filteredRows.length > 0 ? (page - 1) * RECORDS_LIMIT + 1 : 0;
  const endIdx = Math.min(page * RECORDS_LIMIT, filteredRows.length);

  const changePage = useCallback((offset) => {
    const targetPage = page + offset;
    if (targetPage < 1 || targetPage > effectiveTotalPages) return;
    setPage(targetPage);
  }, [page, effectiveTotalPages]);

  const handleFilterChange = (colIdx, value) => {
    setFilters(prev => ({ ...prev, [colIdx]: value }));
  };

  const exportCSV = () => {
    const header = inferredColumns.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',');
    const bodyRows = filteredRows.map(r => r.map(val => `"${String(val ?? '').replace(/"/g, '""')}"`).join(','));
    const csv = [header, ...bodyRows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const filename = (currentQuestion || 'erp_export').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 50) + '.csv';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="table-card">
      <div className="table-header-row">
        <span className="table-title">Query Results</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginLeft: 'auto' }}>
          <button className="topbar-btn" onClick={exportCSV} style={{ padding: '4px 8px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Download size={12} /> Export CSV
          </button>
          <span className="table-count" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Database size={12} style={{ opacity: 0.6 }} />
            Showing {startIdx}-{endIdx} of {effectiveTotal} records
          </span>
        </div>
      </div>

      {filterableCols.length > 0 && (
        <div className="filter-bar">
          <div className="filter-label">
            <Filter size={12} /> Quick Filters:
          </div>
          {filterableCols.map(f => (
            <div key={f.colIdx} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
              <label style={{ color: 'var(--text3)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{f.col}:</label>
              <select className="filter-select" value={filters[f.colIdx] || ''} onChange={e => handleFilterChange(f.colIdx, e.target.value)}>
                <option value="">All {f.col}</option>
                {f.values.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          ))}
        </div>
      )}

      <>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                {inferredColumns.map((c, i) => <th key={i}>{escHtml(String(c))}</th>)}
              </tr>
            </thead>
            <tbody>
              {currentPageRows.map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td key={ci} title={escHtml(isBlankCell(cell) ? '' : String(cell))}>{escHtml(isBlankCell(cell) ? '' : String(cell))}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="pagination-bar">
          <span>Page <strong>{page}</strong> of <strong>{effectiveTotalPages}</strong></span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="topbar-btn" disabled={page <= 1} onClick={() => changePage(-1)} style={{ padding: '4px 8px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
              <ChevronLeft size={12} /> Prev
            </button>
            <button className="topbar-btn" disabled={page >= effectiveTotalPages} onClick={() => changePage(1)} style={{ padding: '4px 8px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
              Next <ChevronRight size={12} />
            </button>
          </div>
        </div>
      </>
    </div>
  );
}

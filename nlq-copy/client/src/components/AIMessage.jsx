import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Bot, ThumbsUp, ThumbsDown, Zap, MessageSquare, HelpCircle, CornerDownRight, AlignLeft, Database, DatabaseZap, Filter, Check, X } from 'lucide-react';
import { escHtml, parseNarrativeMarkdown, formatDuration } from '../utils/helpers';
import useTypewriter from '../utils/useTypewriter';
import ChartRenderer from './ChartRenderer';
import DataTable from './DataTable';

const STAGGER_DELAY = 120;

export default function AIMessage({ message, currentQuestion, onFeedback, onSuggestQuery, index }) {
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const [visibleCards, setVisibleCards] = useState(new Set());
  const [feedbackState, setFeedbackState] = useState(null);
  const [hoveredSection, setHoveredSection] = useState(null);
  const cardRefs = useRef({});
  const isBlankCell = (value) => {
    if (value === null || value === undefined) return true;
    const text = String(value).trim();
    return !text || text === '—' || text === '-';
  };

  const { narrative, summary, insights, chartConfig, tableColumns, tableRows, totalRecords, pipeline, followUps, collectionName, needsClarification, rawData, dbPerformance } = message;
  const derivedColumns = useMemo(() => {
    if (Array.isArray(tableColumns) && tableColumns.length > 0) {
      return tableColumns;
    }
    if (Array.isArray(tableRows) && tableRows.length > 0 && Array.isArray(tableRows[0])) {
      return tableRows[0].map((_, idx) => `Column ${idx + 1}`);
    }
    if (Array.isArray(rawData) && rawData.length > 0 && rawData[0] && typeof rawData[0] === 'object' && !Array.isArray(rawData[0])) {
      return Object.keys(rawData[0]).filter(key => key !== '_id');
    }
    return [];
  }, [tableColumns, tableRows, rawData]);
  const derivedRows = useMemo(() => {
    if (Array.isArray(tableRows) && tableRows.length > 0) {
      return tableRows.filter(row => Array.isArray(row) && row.some(cell => !isBlankCell(cell)));
    }
    if (Array.isArray(rawData) && rawData.length > 0) {
      const columnSet = derivedColumns.length > 0 ? derivedColumns : Object.keys(rawData[0] || {});
      return rawData.map(row => {
        if (Array.isArray(row)) {
          return row.map(cell => Array.isArray(cell) ? cell.join(', ') : cell);
        }
        return columnSet.map(col => {
          const value = row?.[col];
          if (Array.isArray(value)) return value.join(', ');
          if (value && typeof value === 'object') return JSON.stringify(value);
          return value;
        });
      }).filter(row => Array.isArray(row) && row.some(cell => !isBlankCell(cell)));
    }
    return [];
  }, [tableRows, rawData, derivedColumns]);
  const hasRows = Number(totalRecords || 0) > 0 && derivedRows.length > 0;
  const isUnrelated = message.isUnrelated;

  const { displayed: typedNarrative, isComplete: typingDone } = useTypewriter(narrative || 'No data returned.', { speed: 18 });

  const cardTypes = useMemo(() => {
    const types = [];
    if (dbPerformance) types.push('performance');
    if (insights?.length) types.push('insights');
    if (totalRecords > 0 && !derivedColumns.length) types.push('stats');
    if (chartConfig) types.push('chart');
    if (derivedColumns.length && derivedRows.length) types.push('table');
    if (pipeline && Object.keys(pipeline).length > 0) types.push('pipeline');
    return types;
  }, [dbPerformance, insights, totalRecords, derivedColumns, derivedRows, chartConfig, pipeline]);

  useEffect(() => {
    if (typingDone && cardTypes.length > 0) {
      let i = 0;
      const interval = setInterval(() => {
        if (i < cardTypes.length) {
          setVisibleCards(prev => new Set(prev).add(cardTypes[i]));
          i++;
        } else {
          clearInterval(interval);
        }
      }, STAGGER_DELAY);
      return () => clearInterval(interval);
    }
  }, [typingDone, cardTypes]);

  const handleFeedback = useCallback((verdict) => {
    setFeedbackState(verdict);
    onFeedback(verdict, message);
  }, [onFeedback, message]);

  const handleSuggestQuery = useCallback((query) => {
    onSuggestQuery(query);
  }, [onSuggestQuery]);

  const pipelineId = 'pl-' + (message.id || Date.now());
  const tableContainerId = 'tbl-' + (message.id || Date.now());

  const showFollowUps = typingDone && !needsClarification && followUps?.length > 0;

  return (
    <div className="msg-row ai-msg" style={{ animationDelay: `${(index || 0) * 0.1}s` }}>
      <div className="msg-avatar">
        <Bot size={16} />
      </div>
      <div className="msg-content ai-block">
        <div className="narrative-card narrative-card-dynamic">
          <div className="narr-label narr-label-animated">
            <svg width="12" height="12" viewBox="0 0 24 24" stroke="currentColor" fill="none" strokeWidth="2">
              <path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
            </svg>
            Analysis
            {!typingDone && <span className="typing-cursor" />}
          </div>
          <div className="narrative-text-stream" dangerouslySetInnerHTML={{ __html: parseNarrativeMarkdown(typedNarrative) }} />

          {hasRows && (
            <div className={`feedback-bar feedback-dynamic ${feedbackState ? 'feedback-done' : ''}`}>
              <span className="feedback-label">Result correct?</span>
              <button
                className={`feedback-btn ${feedbackState === 'up' ? 'feedback-active feedback-active-up' : ''}`}
                disabled={!!feedbackState}
                onClick={() => handleFeedback('up')}
              >
                {feedbackState === 'up' ? <Check size={13} /> : <ThumbsUp size={13} />}
                Yes
              </button>
              <button
                className={`feedback-btn ${feedbackState === 'down' ? 'feedback-active feedback-active-down' : ''}`}
                disabled={!!feedbackState}
                onClick={() => handleFeedback('down')}
              >
                {feedbackState === 'down' ? <X size={13} /> : <ThumbsDown size={13} />}
                No
              </button>
            </div>
          )}

          {showFollowUps && (
            <div className="followups-container followups-enter">
              <div className="followups-header">
                <MessageSquare size={12} /> Suggested Follow-ups:
              </div>
              <div className="followups-list">
                {followUps.map((q, i) => (
                  <div
                    key={i}
                    className="followup-item"
                    onClick={() => handleSuggestQuery(q)}
                    style={{ animationDelay: `${i * 0.08}s` }}
                  >
                    <CornerDownRight size={14} className="followup-icon" />
                    <span>{escHtml(q)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className={`ai-secondary-card-group ${typingDone ? 'ready' : ''}`}>
          {visibleCards.has('performance') && dbPerformance && (
            <div className="card-enter card-enter-performance">
              <DbPerformanceCard perf={dbPerformance} />
            </div>
          )}

          {visibleCards.has('insights') && insights?.length > 0 && (
            <div className="card-enter card-enter-insights">
              <div className="insights-row insights-dynamic">
                {insights.map((i, idx) => (
                  <div key={idx} className="insight-chip insight-chip-dynamic" style={{ animationDelay: `${idx * 0.06}s` }}>
                    <Zap size={12} className="insight-icon" />
                    <span>{escHtml(i)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {visibleCards.has('stats') && totalRecords > 0 && !derivedColumns.length && (
            <div className="card-enter card-enter-stats">
              <div className="stats-row">
                <div className="stat-card stat-card-animated">
                  <div className="stat-num stat-num-count">{totalRecords.toLocaleString()}</div>
                  <div className="stat-label">Total Records</div>
                </div>
              </div>
            </div>
          )}

          {visibleCards.has('chart') && chartConfig && (
            <div className="card-enter card-enter-chart">
              <ChartRenderer config={chartConfig} />
            </div>
          )}

          {derivedColumns.length > 0 && derivedRows.length > 0 && (
            <div className="card-enter card-enter-table">
              <DataTable
                containerId={tableContainerId}
                columns={derivedColumns}
                rows={derivedRows}
                total={totalRecords}
                currentQuestion={currentQuestion}
                collectionName={collectionName}
                rawData={rawData}
              />

              {summary && (
                <div className="table-summary-card table-summary-enter">
                  <div className="table-summary-label">
                    <AlignLeft size={12} /> Table Summary
                  </div>
                  <div className="table-summary-text">{escHtml(summary)}</div>
                </div>
              )}
            </div>
          )}

          {visibleCards.has('pipeline') && pipeline && Object.keys(pipeline).length > 0 && (
            <div className="card-enter card-enter-pipeline">
              <div className={`pipeline-toggle ${pipelineOpen ? 'open' : ''}`} onClick={() => setPipelineOpen(!pipelineOpen)}>
                <span className="pipeline-arrow">▶</span>
                View generated aggregation pipeline
              </div>
              <pre className={`pipeline-code ${pipelineOpen ? 'open' : ''}`}>
                <code>{syntaxHighlight(JSON.stringify(pipeline, null, 2))}</code>
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function syntaxHighlight(json) {
  return json
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
    .replace(/: "([^"]+)"/g, ': <span class="json-string">"$1"</span>')
    .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
    .replace(/: (true|false)/g, ': <span class="json-boolean">$1</span>')
    .replace(/: (null)/g, ': <span class="json-null">$1</span>');
}

function DbPerformanceCard({ perf }) {
  const isCollScan = perf.isCollScan && perf.matchFields?.length;
  const isGreen = perf.indexUsed;

  if (isCollScan) {
    return (
      <div className="dba-insight-card dba-warning">
        <DatabaseZap size={18} className="dba-icon dba-icon-warning" />
        <div>
          <div className="dba-title dba-title-warning">DBA Performance Insight</div>
          <div className="dba-text">
            This query performed a <code className="dba-code">COLLSCAN</code> (Full Collection Scan) in{' '}
            <strong>{formatDuration(perf.executionTimeMs)}</strong>.
            Creating a single-field index on the filter field(s){' '}
            <code className="dba-code-highlight">{escHtml(perf.matchFields.join(', '))}</code>{' '}
            will prevent scanning all documents, speeding up execution by up to <strong className="dba-strong">15x</strong>.
          </div>
        </div>
      </div>
    );
  }

  if (isGreen) {
    return (
      <div className="dba-insight-card dba-success">
        <Database size={18} className="dba-icon dba-icon-success" />
        <div>
          <div className="dba-title dba-title-success">DBA Optimization Active</div>
          <div className="dba-text">
            Query executed efficiently in <strong>{formatDuration(perf.executionTimeMs)}</strong>{' '}
            using active index <code className="dba-code-highlight dba-code-success">{escHtml(perf.indexUsed)}</code>.
            No performance optimizations required.
          </div>
        </div>
      </div>
    );
  }

  return null;
}

import { Zap, HelpCircle, BarChart3, ClipboardList, Calculator, AlertTriangle, DollarSign, RefreshCw, Play, BrainCircuit } from 'lucide-react';

const iconMap = {
  'help-circle': HelpCircle,
  'bar-chart-3': BarChart3,
  'clipboard-list': ClipboardList,
  'calculator': Calculator,
  'alert-triangle': AlertTriangle,
  'dollar-sign': DollarSign,
};

function getIcon(text) {
  const lower = text.toLowerCase();
  if (lower.includes('chart') || lower.includes('pie') || lower.includes('bar') || lower.includes('line') || lower.includes('doughnut') || lower.includes('scatter')) return 'bar-chart-3';
  if (lower.includes('list') || lower.includes('show') || lower.includes('find')) return 'clipboard-list';
  if (lower.includes('how many') || lower.includes('count') || lower.includes('total') || lower.includes('sum') || lower.includes('average')) return 'calculator';
  if (lower.includes('draft') || lower.includes('pending') || lower.includes('status') || lower.includes('error') || lower.includes('warning') || lower.includes('critical')) return 'alert-triangle';
  if (lower.includes('revenue') || lower.includes('cost') || lower.includes('price') || lower.includes('sale') || lower.includes('budget') || lower.includes('financial')) return 'dollar-sign';
  return 'help-circle';
}

export default function Sidebar({
  suggestions, suggestionsLoading, onReloadSuggestions, onSendSuggestion,
  templates, dbList, userList, modelOptions,
  dbName, onDbNameChange, userId, onUserIdChange,
  apiBase, onApiBaseChange,
  runtime, onRuntimeChange,
  ggufModel, onGgufModelChange, safeModel, onSafeModelChange, ollamaModel, onOllamaModelChange,
  reasoningEnabled, onReasoningEnabledChange, reasoningRuntime, onReasoningRuntimeChange,
  reasoningGgufModel, onReasoningGgufModelChange, reasoningSafeModel, onReasoningSafeModelChange,
  reasoningOllamaModel, onReasoningOllamaModelChange,
  validationEnabled, onValidationEnabledChange, computeMode, onComputeModeChange,
  hybridGpuLayers, onHybridGpuLayersChange, hybridGpuMemory, onHybridGpuMemoryChange,
  trainingInfo, trainingRunning, onTrain,
}) {
  const ggufModels = modelOptions?.gguf_models || [];
  const safeModels = modelOptions?.safetensors_models || [];
  const ollamaModels = modelOptions?.ollama_models || [];

  const hasTrainingData = (trainingInfo?.feedback_count || 0) > 0
    || (trainingInfo?.collection_corrections || 0) > 0
    || (trainingInfo?.field_corrections || 0) > 0;

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-mark">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
          </div>
          <div>
            <div className="logo-text">ERP Intelligence</div>
            <div className="logo-sub">AI Query Engine</div>
          </div>
        </div>
      </div>

      <div className="sidebar-main">
        <div className="sidebar-section">
          <div className="sidebar-label-row">
            <div className="sidebar-label">Suggested Queries</div>
            <button className="sidebar-refresh-btn" onClick={onReloadSuggestions} disabled={suggestionsLoading} title="Reload suggestions">
              <RefreshCw size={13} className={suggestionsLoading ? 'animate-spin' : ''} />
            </button>
          </div>
          <ul className="suggestion-list">
            {suggestionsLoading ? (
              <li className="suggestion-item" style={{ cursor: 'default', opacity: 0.7, display: 'flex', alignItems: 'center', gap: 8 }}>
                <RefreshCw size={14} className="animate-spin" style={{ color: 'var(--accent2)', flexShrink: 0 }} />
                <span>Loading suggestions...</span>
              </li>
            ) : suggestions.length === 0 ? (
              <li className="suggestion-item" style={{ cursor: 'default', opacity: 0.7, display: 'flex', alignItems: 'center', gap: 8 }}>
                <HelpCircle size={14} style={{ opacity: 0.7, flexShrink: 0 }} />
                <span>No suggestions available</span>
              </li>
            ) : suggestions.map((s, i) => {
              const Icon = iconMap[getIcon(s)] || HelpCircle;
              return (
                <li key={i} className="suggestion-item" onClick={() => onSendSuggestion(s)} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Icon size={14} style={{ opacity: 0.7, flexShrink: 0 }} />
                  <span>{s}</span>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="sidebar-section" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 180 }}>
          <div className="sidebar-label">Active Templates</div>
          <div className="sidebar-templates">
            {templates.length === 0 ? (
              <div style={{ color: 'var(--text3)', fontSize: 12 }}>Loading...</div>
            ) : templates.map((t, i) => (
              <div key={i} className="template-badge">
                <div className="dot" />
                <span>{t}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label-row">
            <div className="sidebar-label">
              <BrainCircuit size={13} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              Model Training
            </div>
            <button
              className="sidebar-train-btn"
              onClick={onTrain}
              disabled={trainingRunning || !hasTrainingData}
              title={trainingRunning ? 'Training in progress...' : 'Start training on accumulated feedback'}
            >
              <Play size={12} style={{ marginRight: 3 }} />
              {trainingRunning ? 'Training...' : 'Train'}
            </button>
          </div>
          <div className="training-stats">
            <div className="training-stat-item">
              <span className="training-stat-label">Feedback</span>
              <span className="training-stat-value">{trainingInfo?.feedback_count ?? 0}</span>
            </div>
            <div className="training-stat-item">
              <span className="training-stat-label">Collection corrections</span>
              <span className="training-stat-value">{trainingInfo?.collection_corrections ?? 0}</span>
            </div>
            <div className="training-stat-item">
              <span className="training-stat-label">Field corrections</span>
              <span className="training-stat-value">{trainingInfo?.field_corrections ?? 0}</span>
            </div>
            <div className="training-stat-item">
              <span className="training-stat-label">Retrain dataset</span>
              <span className="training-stat-value">{trainingInfo?.retrain_dataset_size ?? 0} rows</span>
            </div>
            {trainingRunning && (
              <div className="training-progress">Training in progress... <span className="animate-pulse">⏳</span></div>
            )}
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Runtime Controls</div>
          <div className="controls-grid">
            <input className="control-input" value={apiBase} onChange={e => onApiBaseChange(e.target.value)} placeholder="API Base URL" />
            <select className="control-select" value={dbName} onChange={e => onDbNameChange(e.target.value)}>
              {dbList.map(db => <option key={db} value={db}>{db}</option>)}
            </select>
            <select className="control-select" value={userId} onChange={e => onUserIdChange(e.target.value)}>
              {userList.map(u => {
                const id = String(u.user_id || u);
                const label = u.name ? `${u.name} (${id})` : id;
                return <option key={id} value={id}>{label}</option>;
              })}
            </select>
            <select className="control-select" value={runtime} onChange={e => onRuntimeChange(e.target.value)}>
              <option value="gguf">Runtime: GGUF ({ggufModels.length})</option>
              <option value="safetensors">Runtime: Safetensors ({safeModels.length})</option>
              <option value="ollama">Runtime: Ollama ({ollamaModels.length})</option>
            </select>
            <label className="control-checkbox-label">
              <span>Special Reasoning Model</span>
              <input type="checkbox" checked={reasoningEnabled} onChange={e => onReasoningEnabledChange(e.target.checked)} />
            </label>
            <select className="control-select" value={validationEnabled ? 'on' : 'off'} onChange={e => onValidationEnabledChange(e.target.value === 'on')}>
              <option value="on">Validation: On</option>
              <option value="off">Validation: Off</option>
            </select>
            {reasoningEnabled && (
              <select className="control-select" value={reasoningRuntime} onChange={e => onReasoningRuntimeChange(e.target.value)}>
                <option value="gguf">Reasoning: GGUF ({ggufModels.length})</option>
                <option value="safetensors">Reasoning: Safetensors ({safeModels.length})</option>
                <option value="ollama">Reasoning: Ollama ({ollamaModels.length})</option>
              </select>
            )}
            <select className="control-select" value={computeMode} onChange={e => onComputeModeChange(e.target.value)}>
              <option value="gpu">Compute: GPU</option>
              <option value="cpu">Compute: CPU</option>
              <option value="hybrid">Compute: Hybrid (GPU+CPU)</option>
              <option value="auto">Compute: Auto</option>
            </select>
            {computeMode === 'hybrid' && runtime === 'gguf' && (
              <input className="control-input" type="number" min={1} max={80} value={hybridGpuLayers} onChange={e => onHybridGpuLayersChange(Number(e.target.value))} placeholder="Hybrid GGUF GPU layers" />
            )}
            {computeMode === 'hybrid' && runtime === 'safetensors' && (
              <input className="control-input" type="number" min={512} value={hybridGpuMemory} onChange={e => onHybridGpuMemoryChange(Number(e.target.value))} placeholder="Hybrid safetensors GPU MB" />
            )}
            {runtime === 'gguf' && ggufModels.length > 0 && (
              <select className="control-select" value={ggufModel} onChange={e => onGgufModelChange(e.target.value)}>
                {ggufModels.map(m => <option key={m} value={m}>{m.split('/').pop() || m}</option>)}
              </select>
            )}
            {runtime === 'safetensors' && safeModels.length > 0 && (
              <select className="control-select" value={safeModel} onChange={e => onSafeModelChange(e.target.value)}>
                {safeModels.map(m => <option key={m} value={m}>{m.split('/').pop() || m}</option>)}
              </select>
            )}
            {runtime === 'ollama' && ollamaModels.length > 0 && (
              <select className="control-select" value={ollamaModel} onChange={e => onOllamaModelChange(e.target.value)}>
                {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            )}
            {reasoningEnabled && reasoningRuntime === 'gguf' && ggufModels.length > 0 && (
              <select className="control-select" value={reasoningGgufModel} onChange={e => onReasoningGgufModelChange(e.target.value)}>
                {ggufModels.map(m => <option key={m} value={m}>{m.split('/').pop() || m}</option>)}
              </select>
            )}
            {reasoningEnabled && reasoningRuntime === 'safetensors' && safeModels.length > 0 && (
              <select className="control-select" value={reasoningSafeModel} onChange={e => onReasoningSafeModelChange(e.target.value)}>
                {safeModels.map(m => <option key={m} value={m}>{m.split('/').pop() || m}</option>)}
              </select>
            )}
            {reasoningEnabled && reasoningRuntime === 'ollama' && ollamaModels.length > 0 && (
              <select className="control-select" value={reasoningOllamaModel} onChange={e => onReasoningOllamaModelChange(e.target.value)}>
                {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            )}
          </div>
        </div>
      </div>

      <div className="sidebar-footer">
        Connected to <span style={{ color: 'var(--accent2)' }}>{dbName || 'MongoDB'}</span>
      </div>
    </aside>
  );
}

import { useState, useRef, useCallback } from 'react';
import { fetchNDJSON, apiUrl, fetchJSON, setApiBase, getTrainingStatus, triggerTraining } from '../utils/api';
import { estimateTokens, generateConversationId, formatDuration } from '../utils/helpers';

export default function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState('');
  const [thinkingStep, setThinkingStep] = useState('');
  const [thinkingReasoning, setThinkingReasoning] = useState('');
  const [thinkingAnswer, setThinkingAnswer] = useState('');
  const [responseTimer, setResponseTimer] = useState('0 s');
  const [contextLimit, setContextLimit] = useState(10);
  const [contextTokenLimit, setContextTokenLimit] = useState(8000);
  const [baseTokens, setBaseTokens] = useState(1200);
  const [currentQueryTokens, setCurrentQueryTokens] = useState(0);
  const [currentResponseTokens, setCurrentResponseTokens] = useState(0);
  const [activeDataTokens, setActiveDataTokens] = useState(0);
  const [timingPanelOpen, setTimingPanelOpen] = useState(false);
  const [timingState, setTimingState] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [templates, setTemplates] = useState([]);
  const [dbList, setDbList] = useState([]);
  const [userList, setUserList] = useState([]);
  const [modelOptions, setModelOptions] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [conversationId, setConversationId] = useState(generateConversationId());
  const [apiBase, setApiBaseState] = useState(() => {
    try { return localStorage.getItem('nlq-api-base') || 'http://127.0.0.1:8000'; } catch { return 'http://127.0.0.1:8000'; }
  });
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [clarificationData, setClarificationData] = useState(null);
  const [dbName, setDbName] = useState('ECMS_MAY03_COPY');
  const [userId, setUserId] = useState('admin');
  const [runtime, setRuntime] = useState('gguf');
  const [ggufModel, setGgufModel] = useState('');
  const [safeModel, setSafeModel] = useState('');
  const [ollamaModel, setOllamaModel] = useState('');
  const [reasoningEnabled, setReasoningEnabled] = useState(true);
  const [reasoningRuntime, setReasoningRuntime] = useState('gguf');
  const [reasoningGgufModel, setReasoningGgufModel] = useState('');
  const [reasoningSafeModel, setReasoningSafeModel] = useState('');
  const [reasoningOllamaModel, setReasoningOllamaModel] = useState('');
  const [validationEnabled, setValidationEnabled] = useState(true);
  const [computeMode, setComputeMode] = useState('gpu');
  const [hybridGpuLayers, setHybridGpuLayers] = useState(20);
  const [hybridGpuMemory, setHybridGpuMemory] = useState(3072);
  const [trainingInfo, setTrainingInfo] = useState(null);
  const [trainingRunning, setTrainingRunning] = useState(false);

  const timerRef = useRef(null);
  const requestStartedRef = useRef(0);
  const dbRef = useRef(dbName);
  const userRef = useRef(userId);

  dbRef.current = dbName;
  userRef.current = userId;

  const updateApiBase = useCallback((url) => {
    setApiBaseState(url);
    setApiBase(url);
  }, []);

  const startTimer = useCallback(() => {
    requestStartedRef.current = Date.now();
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setResponseTimer(formatDuration(Date.now() - requestStartedRef.current));
      setTimingState(prev => {
        if (!prev || prev.done) return prev;
        return { ...prev };
      });
    }, 100);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (requestStartedRef.current) {
      setResponseTimer(formatDuration(Date.now() - requestStartedRef.current));
    }
  }, []);

  const updateContextBadge = useCallback(() => {
    const limit = Math.max(1, Number(contextTokenLimit) || 8000);
    const used = Math.max(0, Math.floor(baseTokens + currentQueryTokens + currentResponseTokens + activeDataTokens));
    return { limit, used, percentLeft: Math.max(0, Math.floor(((limit - used) / limit) * 100)) };
  }, [contextTokenLimit, baseTokens, currentQueryTokens, currentResponseTokens, activeDataTokens]);

  const resetTiming = useCallback((question) => {
    setTimingState({
      question: String(question || ''),
      start: performance.now(),
      currentStage: 'ui_start',
      currentStart: performance.now(),
      entries: [],
      done: false,
      status: null,
    });
  }, []);

  const addTimingStage = useCallback((stage, ms) => {
    setTimingState(prev => {
      if (!prev || prev.done) return prev;
      return { ...prev, entries: [...prev.entries, { stage, ms: Math.max(0, Number(ms || 0)) }] };
    });
  }, []);

  const beginTimingStage = useCallback((stage) => {
    setTimingState(prev => {
      if (!prev || prev.done) return prev;
      const now = performance.now();
      const ms = prev.currentStage ? Math.max(0, now - prev.currentStart) : 0;
      const newEntries = prev.currentStage
        ? [...prev.entries, { stage: prev.currentStage, ms }]
        : prev.entries;
      return { ...prev, entries: newEntries, currentStage: stage, currentStart: now };
    });
  }, []);

  const finishTiming = useCallback((status) => {
    setTimingState(prev => {
      if (!prev) return prev;
      const now = performance.now();
      const ms = prev.currentStage ? Math.max(0, now - prev.currentStart) : 0;
      const newEntries = prev.currentStage
        ? [...prev.entries, { stage: prev.currentStage, ms }]
        : prev.entries;
      return { ...prev, entries: newEntries, currentStage: null, currentStart: 0, done: true, status: status || 'done' };
    });
  }, []);

  const loadSuggestions = useCallback(async (overrides = {}) => {
    setSuggestionsLoading(true);
    try {
      const db = overrides.dbName || dbRef.current;
      const user = overrides.userId || userRef.current;
      const data = await fetchJSON(`/suggestions?db_name=${encodeURIComponent(db)}&user_id=${encodeURIComponent(user)}`);
      setSuggestions(data.suggestions || []);
    } catch {
      setSuggestions([]);
    } finally {
      setSuggestionsLoading(false);
    }
  }, []);

  const loadTemplates = useCallback(async (overrides = {}) => {
    try {
      const db = overrides.dbName || dbRef.current;
      const user = overrides.userId || userRef.current;
      const data = await fetchJSON(`/bootstrap?db_name=${encodeURIComponent(db)}&user_id=${encodeURIComponent(user)}&refresh_rbac=true`);
      setDbList(data.databases || []);
      setUserList(data.users || []);
      setTemplates(Object.keys(data.table_metadata || {}));
    } catch {
      setTemplates([]);
    }
  }, []);

  const loadModelOptions = useCallback(async () => {
    try {
      const data = await fetchJSON('/models/options');
      setModelOptions(data);
      return data;
    } catch {
      return null;
    }
  }, []);

  const loadTrainingStatus = useCallback(async () => {
    try {
      const data = await getTrainingStatus();
      setTrainingInfo(data);
      return data;
    } catch {
      return null;
    }
  }, []);

  const handleTriggerTraining = useCallback(async () => {
    if (trainingRunning) return;
    setTrainingRunning(true);
    try {
      const result = await triggerTraining();
      setTrainingInfo(prev => ({ ...prev, ...result.info_after }));
      return result;
    } catch {
      return null;
    } finally {
      setTrainingRunning(false);
    }
  }, [trainingRunning]);

  const clearChat = useCallback(() => {
    setMessages([]);
    setClarificationData(null);
    setChatHistory([]);
    setConversationId(generateConversationId());
    setActiveDataTokens(0);
    setCurrentQueryTokens(0);
    setCurrentResponseTokens(0);
  }, []);

  const appendUser = useCallback((text) => {
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setChatHistory(prev => [...prev, { role: 'user', content: text }]);
  }, []);

  const appendAI = useCallback((result) => {
    setMessages(prev => {
      const filtered = prev.filter(m => m.role !== 'thinking');
      return [...filtered, { role: 'assistant', ...result }];
    });
    if (result.narrative) {
      setChatHistory(prev => {
        const updated = [...prev, { role: 'assistant', content: result.narrative }];
        return updated.length > contextLimit ? updated.slice(-contextLimit) : updated;
      });
    }
  }, [contextLimit]);

  const appendError = useCallback((msg) => {
    setMessages(prev => {
      const filtered = prev.filter(m => m.role !== 'thinking');
      return [...filtered, { role: 'error', content: msg }];
    });
  }, []);

  const sendQuery = useCallback(async () => {
    const question = input.trim();
    if (!question || isLoading) return;
    setInput('');
    setCurrentQuestion(question);
    setIsLoading(true);
    setClarificationData(null);
    startTimer();
    resetTiming(question);

    const promptTokens = estimateTokens(question);
    setCurrentQueryTokens(promptTokens);
    setCurrentResponseTokens(0);
    setActiveDataTokens(0);
    setThinkingStep('Analyzing your question...');
    setThinkingReasoning('');
    setThinkingAnswer('');

    appendUser(question);
    setMessages(prev => [...prev, { role: 'thinking' }]);

    fetchNDJSON(apiUrl('/query_stream'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: question,
        db_name: dbName,
        user_id: userId,
        conversation_id: conversationId,
        chat_context: chatHistory,
        model_runtime: runtime,
        reasoning_model_enabled: reasoningEnabled,
        validation_enabled: validationEnabled,
        reasoning_model_runtime: reasoningRuntime,
        compute_mode: computeMode,
        hybrid_gpu_layers: hybridGpuLayers,
        hybrid_gpu_memory_mb: hybridGpuMemory,
        gguf_model_path: runtime === 'gguf' ? ggufModel : null,
        safetensors_model_id: runtime === 'safetensors' ? safeModel : null,
        ollama_model: runtime === 'ollama' ? ollamaModel : null,
        reasoning_gguf_model_path: reasoningRuntime === 'gguf' ? reasoningGgufModel : null,
        reasoning_safetensors_model_id: reasoningRuntime === 'safetensors' ? reasoningSafeModel : null,
        reasoning_ollama_model: reasoningRuntime === 'ollama' ? reasoningOllamaModel : null,
      }),
    }, {
      onProgress: (stage, message) => {
        beginTimingStage(stage || 'processing');
        if (message) {
          setThinkingStep(message);
        } else {
          const statusMessages = {
            intent: 'Analyzing your question...',
            pipeline: 'Formulating database query...',
            db_fetch: 'Retrieving records from database...',
            healing: 'Tuning search parameters and self-healing...',
            narrating: 'Compiling insights and formatting narrative...',
          };
          setThinkingStep(statusMessages[stage] || 'Processing...');
        }
      },
      onComplete: (data) => {
        const renderStarted = performance.now();
        if (data.rawData) {
          setActiveDataTokens(estimateTokens(data.rawData) + estimateTokens(data.pipeline || ''));
        }
        setCurrentResponseTokens(prev => prev || estimateTokens(data.response || data.narrative || ''));
        setClarificationData(data.needsClarification ? data : null);
        appendAI(data);
        addTimingStage('ui_rendering', performance.now() - renderStarted);
        finishTiming('ok');
        stopTimer();
        setIsLoading(false);
      },
      onError: (err) => {
        appendError(err);
        finishTiming('error');
        stopTimer();
        setIsLoading(false);
      },
      onToken: (state, token) => {
        if (state === 'THINKING') {
          setThinkingReasoning(prev => prev + token);
          setThinkingStep('Model reasoning...');
        } else {
          setThinkingAnswer(prev => prev + token);
          setThinkingStep('Composing final answer...');
        }
        setCurrentResponseTokens(prev => prev + Math.max(1, estimateTokens(token)));
      },
      onState: (state) => {
        if (state === 'THINKING') setThinkingStep('Model reasoning...');
        else if (state === 'ANSWER') setThinkingStep('Composing final answer...');
      },
    });
  }, [input, isLoading, dbName, userId, conversationId, chatHistory, runtime, reasoningEnabled, validationEnabled, reasoningRuntime, computeMode, hybridGpuLayers, hybridGpuMemory, ggufModel, safeModel, ollamaModel, reasoningGgufModel, reasoningSafeModel, reasoningOllamaModel, appendUser, appendAI, appendError, beginTimingStage, addTimingStage, finishTiming, startTimer, resetTiming, stopTimer]);

  const contextInfo = updateContextBadge();

  return {
    messages, isLoading, input, setInput,
    thinkingStep, thinkingReasoning, thinkingAnswer,
    responseTimer, contextInfo, timingPanelOpen, setTimingPanelOpen,
    timingState,
    suggestions, suggestionsLoading, templates,
    dbList, userList, modelOptions,
    dbName, setDbName, userId, setUserId,
    runtime, setRuntime,
    ggufModel, setGgufModel, safeModel, setSafeModel, ollamaModel, setOllamaModel,
    reasoningEnabled, setReasoningEnabled, reasoningRuntime, setReasoningRuntime,
    reasoningGgufModel, setReasoningGgufModel, reasoningSafeModel, setReasoningSafeModel,
    reasoningOllamaModel, setReasoningOllamaModel,
    validationEnabled, setValidationEnabled, computeMode, setComputeMode,
    hybridGpuLayers, setHybridGpuLayers, hybridGpuMemory, setHybridGpuMemory,
    apiBase, updateApiBase,
    conversationId, currentQuestion, clarificationData, setClarificationData,
    loadSuggestions, loadTemplates, loadModelOptions, clearChat, sendQuery,
    contextTokenLimit, setContextTokenLimit, baseTokens, setBaseTokens,
    currentQueryTokens, currentResponseTokens, activeDataTokens,
    chatHistory,
    trainingInfo, trainingRunning, loadTrainingStatus, handleTriggerTraining,
  };
}

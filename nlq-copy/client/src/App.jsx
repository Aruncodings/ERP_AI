import { useEffect, useCallback, useRef } from 'react';
import useChat from './hooks/useChat';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import WelcomeBlock, { UserMessage, ThinkingMessage, ErrorMessage } from './components/Messages';
import AIMessage from './components/AIMessage';
import InputArea from './components/InputArea';
import TimingPanel from './components/TimingPanel';
import ClarificationBar from './components/ClarificationBar';
import { fetchJSON } from './utils/api';
import { formatModelLabel } from './utils/helpers';
import StarsBackground from './components/StarsBackground';

export default function App() {
  const chat = useChat();
  const chatRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }, 50);
  }, []);

  useEffect(() => { scrollToBottom(); }, [chat.messages, scrollToBottom]);

  // Load initial data
  useEffect(() => {
    chat.loadModelOptions().then(data => {
      if (data) applyDefaults(data);
    });
    chat.loadTemplates();
    chat.loadSuggestions();
  }, []);

  const applyDefaults = useCallback((data) => {
    const defaults = data.defaults || {};
    if (defaults.model_runtime) chat.setRuntime(defaults.model_runtime);
    if (typeof defaults.validation_enabled === 'boolean') chat.setValidationEnabled(defaults.validation_enabled);
    if (typeof defaults.reasoning_model_enabled === 'boolean') chat.setReasoningEnabled(defaults.reasoning_model_enabled);
    if (defaults.reasoning_model_runtime) chat.setReasoningRuntime(defaults.reasoning_model_runtime);

    const ggufModels = data.gguf_models || [];
    const safeModels = data.safetensors_models || [];
    const ollamaModels = data.ollama_models || [];

    if (defaults.gguf_model_path && ggufModels.includes(defaults.gguf_model_path)) chat.setGgufModel(defaults.gguf_model_path);
    if (defaults.safetensors_model_id && safeModels.includes(defaults.safetensors_model_id)) chat.setSafeModel(defaults.safetensors_model_id);
    if (defaults.ollama_model && ollamaModels.includes(defaults.ollama_model)) chat.setOllamaModel(defaults.ollama_model);
    if (defaults.reasoning_gguf_model_path && ggufModels.includes(defaults.reasoning_gguf_model_path)) chat.setReasoningGgufModel(defaults.reasoning_gguf_model_path);
    if (defaults.reasoning_safetensors_model_id && safeModels.includes(defaults.reasoning_safetensors_model_id)) chat.setReasoningSafeModel(defaults.reasoning_safetensors_model_id);
    if (defaults.reasoning_ollama_model && ollamaModels.includes(defaults.reasoning_ollama_model)) chat.setReasoningOllamaModel(defaults.reasoning_ollama_model);

    const selectedGguf = ggufModels.find(m => String(m).toLowerCase().includes('qwen3.5-4b(reasoning)'));
    if (selectedGguf && !chat.reasoningGgufModel) chat.setReasoningGgufModel(selectedGguf);
  }, []);

  const handleApiBaseChange = useCallback((url) => {
    chat.updateApiBase(url);
    chat.loadModelOptions().then(data => { if (data) applyDefaults(data); });
    chat.loadTemplates();
    chat.loadSuggestions();
  }, []);

  const handleDbChange = useCallback((db) => {
    chat.setDbName(db);
    chat.loadTemplates({ dbName: db });
    chat.loadSuggestions({ dbName: db });
  }, []);

  const handleUserChange = useCallback((user) => {
    chat.setUserId(user);
    chat.loadTemplates({ userId: user });
    chat.loadSuggestions({ userId: user });
  }, []);

  const handleSendSuggestion = useCallback((text) => {
    chat.setInput(text);
    setTimeout(() => chat.sendQuery(), 10);
  }, []);

  const handleClarificationApply = useCallback((text) => {
    chat.setInput(text);
    chat.setClarificationData(null);
    setTimeout(() => chat.sendQuery(), 10);
  }, []);

  const handleSend = useCallback(() => {
    chat.sendQuery();
  }, [chat.sendQuery]);

  const handleFeedback = useCallback(async (verdict, result) => {
    if (verdict === 'up') {
      chat.setClarificationData(null);
      return;
    }
    try {
      const response = await fetchJSON('/query_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: chat.currentQuestion || '',
          feedback: verdict,
          db_name: chat.dbName,
          user_id: chat.userId,
          conversation_id: chat.conversationId,
          chat_context: [],
          model_runtime: chat.runtime,
          reasoning_model_enabled: chat.reasoningEnabled,
          validation_enabled: chat.validationEnabled,
          reasoning_model_runtime: chat.reasoningRuntime,
          compute_mode: chat.computeMode,
          hybrid_gpu_layers: chat.hybridGpuLayers,
          hybrid_gpu_memory_mb: chat.hybridGpuMemory,
          gguf_model_path: chat.runtime === 'gguf' ? chat.ggufModel : null,
          safetensors_model_id: chat.runtime === 'safetensors' ? chat.safeModel : null,
          ollama_model: chat.runtime === 'ollama' ? chat.ollamaModel : null,
          reasoning_gguf_model_path: chat.reasoningRuntime === 'gguf' ? chat.reasoningGgufModel : null,
          reasoning_safetensors_model_id: chat.reasoningRuntime === 'safetensors' ? chat.reasoningSafeModel : null,
          reasoning_ollama_model: chat.reasoningRuntime === 'ollama' ? chat.reasoningOllamaModel : null,
          collection: result.collectionName || '',
          table_choice: result.tableChoice || {},
          plan: result.pipeline || {},
          docs: Array.isArray(result.rawData) ? result.rawData.slice(0, 8) : [],
          total: Number(result.totalRecords || 0),
          table_columns: Array.isArray(result.tableColumns) ? result.tableColumns : [],
          narrative: String(result.narrative || ''),
        }),
      });
      const mapped = {
        narrative: response.response || 'No response.',
        followUps: Array.isArray(response.follow_ups) ? response.follow_ups : [],
        needsClarification: Boolean(response.needs_clarification),
      };
      chat.setClarificationData(mapped);
    } catch (err) {
      // fallback - will show error in chat
    }
  }, [chat]);

  const selectedModelLabel = formatModelLabel(
    chat.runtime === 'gguf' ? chat.ggufModel :
    chat.runtime === 'safetensors' ? chat.safeModel :
    chat.ollamaModel
  ) || 'Selected model';

  return (
    <div className="app-layout">
      <Sidebar
        suggestions={chat.suggestions}
        suggestionsLoading={chat.suggestionsLoading}
        onReloadSuggestions={chat.loadSuggestions}
        onSendSuggestion={handleSendSuggestion}
        templates={chat.templates}
        dbList={chat.dbList}
        userList={chat.userList}
        modelOptions={chat.modelOptions}
        dbName={chat.dbName}
        onDbNameChange={handleDbChange}
        userId={chat.userId}
        onUserIdChange={handleUserChange}
        apiBase={chat.apiBase}
        onApiBaseChange={handleApiBaseChange}
        runtime={chat.runtime}
        onRuntimeChange={chat.setRuntime}
        ggufModel={chat.ggufModel}
        onGgufModelChange={chat.setGgufModel}
        safeModel={chat.safeModel}
        onSafeModelChange={chat.setSafeModel}
        ollamaModel={chat.ollamaModel}
        onOllamaModelChange={chat.setOllamaModel}
        reasoningEnabled={chat.reasoningEnabled}
        onReasoningEnabledChange={chat.setReasoningEnabled}
        reasoningRuntime={chat.reasoningRuntime}
        onReasoningRuntimeChange={chat.setReasoningRuntime}
        reasoningGgufModel={chat.reasoningGgufModel}
        onReasoningGgufModelChange={chat.setReasoningGgufModel}
        reasoningSafeModel={chat.reasoningSafeModel}
        onReasoningSafeModelChange={chat.setReasoningSafeModel}
        reasoningOllamaModel={chat.reasoningOllamaModel}
        onReasoningOllamaModelChange={chat.setReasoningOllamaModel}
        validationEnabled={chat.validationEnabled}
        onValidationEnabledChange={chat.setValidationEnabled}
        computeMode={chat.computeMode}
        onComputeModeChange={chat.setComputeMode}
        hybridGpuLayers={chat.hybridGpuLayers}
        onHybridGpuLayersChange={chat.setHybridGpuLayers}
        hybridGpuMemory={chat.hybridGpuMemory}
        onHybridGpuMemoryChange={chat.setHybridGpuMemory}
      />

      <div className="main">
        <TopBar
          onClearChat={chat.clearChat}
          timingPanelOpen={chat.timingPanelOpen}
          onToggleTiming={() => chat.setTimingPanelOpen(!chat.timingPanelOpen)}
          contextInfo={chat.contextInfo}
          selectedModelLabel={selectedModelLabel}
          responseTimer={chat.responseTimer}
        />

        <div className="chat-area" ref={chatRef} style={{ position: 'relative' }}>
          <StarsBackground />
          {chat.messages.length === 0 && <WelcomeBlock />}
          {chat.messages.map((msg, i) => {
            if (msg.role === 'user') return <UserMessage key={i} content={msg.content} index={i} />;
            if (msg.role === 'thinking') return (
              <ThinkingMessage
                key={i}
                step={chat.thinkingStep}
                reasoning={chat.thinkingReasoning}
                answer={chat.thinkingAnswer}
              />
            );
            if (msg.role === 'error') return <ErrorMessage key={i} content={msg.content} />;
            if (msg.role === 'assistant') return (
              <AIMessage
                key={i}
                index={i}
                message={msg}
                currentQuestion={chat.currentQuestion}
                onFeedback={handleFeedback}
                onSuggestQuery={handleSendSuggestion}
              />
            );
            return null;
          })}
        </div>

        <ClarificationBar data={chat.clarificationData} onApply={handleClarificationApply} />

        <InputArea
          value={chat.input}
          onChange={chat.setInput}
          onSend={handleSend}
          isLoading={chat.isLoading}
        />
      </div>

      <TimingPanel
        open={chat.timingPanelOpen}
        onClose={() => chat.setTimingPanelOpen(false)}
        timingState={chat.timingState}
      />
    </div>
  );
}

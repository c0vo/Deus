"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  Terminal as TerminalIcon,
  FileText,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  AlertTriangle,
  ExternalLink,
  Globe,
  Loader2,
} from "lucide-react";
import FormattedText from "../components/FormattedText";
import { getApiUrl } from "../utils/api";

interface ArticleInfo {
  headline?: string;
  title?: string;
  url?: string;
  domain?: string;
  summary?: string;
  importance_score?: number;
  score?: number;
  source_type?: "rag" | "web";
}

interface Message {
  sender: "user" | "assistant";
  text: string;
  steps?: Array<{
    name: string;
    details: string;
  }>;
  sources?: ArticleInfo[];
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentSteps, setCurrentSteps] = useState<Array<{ name: string; details: string }>>([]);
  const [currentSources, setCurrentSources] = useState<ArticleInfo[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [stepsExpanded, setStepsExpanded] = useState(true);

  // Real-time Web Search Research states (debate_hub blueprint)
  const [researchPhase, setResearchPhase] = useState<"idle" | "searching" | "complete">("idle");
  const [researchQuery, setResearchQuery] = useState("");
  const [researchTotal, setResearchTotal] = useState(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingText, currentSteps, currentSources]);

  useEffect(() => {
    const query = new URLSearchParams(window.location.search).get("query");
    if (query) setInput(query);
  }, []);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    // Add user message
    const userMsg: Message = { sender: "user", text: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setCurrentSteps([]);
    setCurrentSources([]);
    setStreamingText("");
    setStepsExpanded(true);
    setResearchPhase("idle");
    setResearchQuery("");
    setResearchTotal(0);

    try {
      const response = await fetch(getApiUrl("/api/chat/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });

      if (!response.ok) {
        throw new Error(`Chat API error: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("Response body is not readable");

      const decoder = new TextDecoder("utf-8");
      let partialLine = "";
      let currentEvent = "";

      let currentAssistantText = "";
      let tempSteps: Array<{ name: string; details: string }> = [];
      let tempSources: ArticleInfo[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunk).split("\n");
        partialLine = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.substring(7).trim();
            continue;
          }

          if (line.startsWith("data: ")) {
            const rawData = line.substring(6);

            if (currentEvent === "token") {
              try {
                const parsed = JSON.parse(rawData);
                if (typeof parsed === "object" && parsed !== null && parsed.text !== undefined) {
                  currentAssistantText += parsed.text;
                } else if (typeof parsed === "string") {
                  currentAssistantText += parsed;
                } else {
                  currentAssistantText += rawData;
                }
              } catch {
                currentAssistantText += rawData;
              }
              setStreamingText(currentAssistantText);
              continue;
            }

            const dataStr = rawData.trim();
            if (dataStr === "") continue;

            if (currentEvent === "error") {
              currentAssistantText += `<b>ERROR:</b> ${dataStr}`;
              setStreamingText(currentAssistantText);
              continue;
            }

            if (currentEvent === "done") {
              continue;
            }

            try {
              const parsed = JSON.parse(dataStr);

              if (currentEvent === "research_start") {
                setResearchPhase("searching");
                setResearchQuery(parsed.query || query);
                setResearchTotal(0);
              } else if (currentEvent === "research_source") {
                setResearchTotal(parsed.total || 0);
                const newSrc: ArticleInfo = {
                  title: parsed.title || "",
                  headline: parsed.title || "",
                  url: parsed.url || "",
                  domain: parsed.domain || "WEB",
                  source_type: "web",
                };
                if (!tempSources.some((s) => s.url && s.url === newSrc.url)) {
                  tempSources = [...tempSources, newSrc];
                  setCurrentSources(tempSources);
                }
              } else if (currentEvent === "research_complete") {
                setResearchPhase("complete");
                if (parsed.sources_found !== undefined && parsed.sources_found >= 0) {
                  setResearchTotal(parsed.sources_found);
                }
              } else if (currentEvent === "step" || parsed.step) {
                let stepName = "";
                let stepDetails = "";
                if (parsed.step === "classification") {
                  stepName = "🧠 ROUTER CLASS";
                  stepDetails = `Intent: ${parsed.intent ? parsed.intent.toUpperCase() : "GENERAL"} | Reasoning: ${parsed.reasoning}`;
                } else if (parsed.step === "retrieval") {
                  stepName = "🔍 RAG RETRIEVAL";
                  stepDetails = parsed.context ? `Found news context in SQLite-vec.` : `No news context found. Falling back to general knowledge.`;
                } else if (parsed.step === "web_search") {
                  stepName = "🌐 WEB SEARCH";
                  stepDetails = parsed.reasoning || "Searching the web for latest information...";
                }

                if (stepName) {
                  tempSteps = [...tempSteps, { name: stepName, details: stepDetails }];
                  setCurrentSteps(tempSteps);
                }
              } else if (currentEvent === "sources" || parsed.articles) {
                const articlesList = parsed.articles || [];
                const formatted: ArticleInfo[] = articlesList.map((art: any) => ({
                  headline: art.headline || art.title || "",
                  title: art.title || art.headline || "",
                  url: art.url || "",
                  domain: art.domain || (art.url ? art.url.split("://")[1]?.split("/")[0]?.replace("www.", "") : "IN-HOUSE"),
                  summary: art.summary || "",
                  importance_score: art.importance_score,
                  source_type: art.source_type || (art.url ? "web" : "rag"),
                }));
                
                // Merge without duplicates
                const combined = [...tempSources];
                for (const item of formatted) {
                  if (!combined.some((c) => (c.url && c.url === item.url) || (c.headline && c.headline === item.headline))) {
                    combined.push(item);
                  }
                }
                tempSources = combined;
                setCurrentSources(tempSources);
              }
            } catch (err) {
              if (dataStr !== "[DONE]" && dataStr !== "") {
                currentAssistantText += dataStr;
                setStreamingText(currentAssistantText);
              }
            }
          }
        }
      }

      const assistantMsg: Message = {
        sender: "assistant",
        text: currentAssistantText || "Response complete.",
        steps: tempSteps,
        sources: tempSources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setStreamingText("");
      setCurrentSteps([]);
      setCurrentSources(tempSources);
      setLoading(false);

    } catch (err: any) {
      console.error(err);
      const errorMsg: Message = {
        sender: "assistant",
        text: `<div class="text-terminal-error font-bold flex items-center gap-2"><span class="animate-pulse">⚠️ ERROR:</span> ${err.message || "Failed to receive chat response."}</div>`,
      };
      setMessages((prev) => [...prev, errorMsg]);
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col md:flex-row overflow-hidden">
      
      {/* Main Chat Panel */}
      <div className="flex-1 flex flex-col border-r border-border-dim overflow-hidden bg-bg-main">
        
        {/* Chat Header */}
        <div className="px-6 h-12 border-b border-border-dim flex items-center bg-bg-card shrink-0 justify-between">
          <h2 className="text-xs font-bold text-terminal-text uppercase tracking-wider">
            Analyst
          </h2>
          {loading && (
            <div className="flex items-center gap-2 text-[10px] text-terminal-signal animate-pulse font-mono">
              <RefreshCw size={10} className="animate-spin" />
              <span>{researchPhase === "searching" ? "LIVE WEB SEARCHING..." : "ROUTING & RETRIEVING..."}</span>
            </div>
          )}
        </div>

        {/* Scrollable messages log */}
        <div className="flex-1 overflow-auto p-6 space-y-6 scrollbar-thin">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4 max-w-md mx-auto">
              <TerminalIcon size={48} className="text-terminal-signal/50" />
              <div className="space-y-2">
                <h3 className="text-sm font-bold text-terminal-text uppercase tracking-wider">Deus Conversational Analyst</h3>
                <p className="text-xs text-terminal-muted leading-relaxed">
                  Ask queries like <code className="text-terminal-violet font-bold">"Summarize Tesla news"</code> (routed to shallow RAG agent) or <code className="text-terminal-violet font-bold">"What are SK Hynix volatility drivers?"</code> (complex logic flow with live search).
                </p>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex flex-col space-y-2 ${msg.sender === "user" ? "items-end ml-auto max-w-[80%]" : "items-start w-full"}`}>
              {/* Sender Tag */}
              <span className={`text-[10px] uppercase font-bold tracking-wider ${msg.sender === "user" ? "text-terminal-signal" : "text-terminal-violet"}`}>
                {msg.sender === "user" ? "[user@client]" : "[deus@analyst]"}
              </span>

              {/* Message content bubble */}
              <div className={`p-5 border leading-relaxed text-sm ${
                msg.sender === "user"
                  ? "bg-bg-surface border-terminal-signal text-terminal-text rounded-[18px_18px_0_18px]"
                  : "bg-bg-card border-border-dim text-terminal-text font-sans rounded-[18px_18px_18px_0] w-full"
              }`}>
                <FormattedText text={msg.text} />
              </div>

              {/* Embedded Steps info in message */}
              {msg.steps && msg.steps.length > 0 && (
                <div className="border border-border-dim bg-bg-surface p-2.5 text-[10px] text-terminal-muted w-full space-y-1 rounded-sm">
                  <div className="font-bold border-b border-border-dim pb-1 mb-1 text-terminal-text uppercase flex items-center justify-between">
                    <span>Agent Routing Logs</span>
                    <span className="text-[9px] text-terminal-signal">{msg.steps.length} steps</span>
                  </div>
                  {msg.steps.map((s, idx) => (
                    <div key={idx} className="flex gap-2">
                      <span className="text-terminal-violet font-bold shrink-0">{s.name}:</span>
                      <span className="break-words">{s.details}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Current Streaming Assistant Message */}
          {(streamingText || currentSteps.length > 0) && (
            <div className="flex flex-col space-y-2 items-start w-full">
              <span className="text-[10px] uppercase font-bold tracking-wider text-terminal-violet">
                [deus@analyst]
              </span>

              {/* Streaming steps log */}
              {currentSteps.length > 0 && (
                <div className="border border-border-dim bg-bg-surface p-3 rounded-sm w-full">
                  <button
                    onClick={() => setStepsExpanded(!stepsExpanded)}
                    className="w-full flex items-center justify-between text-xs font-bold text-terminal-muted uppercase tracking-wider"
                  >
                    <span className="flex items-center gap-1.5">
                      <TerminalIcon size={12} className="text-terminal-violet animate-pulse" />
                      Step Execution Log ({currentSteps.length})
                    </span>
                    {stepsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  {stepsExpanded && (
                    <div className="mt-2 space-y-1.5 text-[10px] font-mono border-t border-border-dim/40 pt-2 text-terminal-muted">
                      {currentSteps.map((s, idx) => (
                        <div key={idx} className="flex gap-2">
                          <span className="text-terminal-signal font-bold shrink-0">{s.name}:</span>
                          <span className="break-all">{s.details}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Streaming text bubble */}
              {streamingText && (
                <div className="p-5 border border-border-dim bg-bg-card text-terminal-text rounded-[18px_18px_18px_0] w-full text-sm font-sans leading-relaxed">
                  <FormattedText text={streamingText} />
                  <span className="inline-block w-1.5 h-3.5 bg-terminal-text animate-pulse ml-1 align-middle" />
                </div>
              )}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Messaging Input Form */}
        <div className="p-4 border-t border-border-dim bg-bg-card shrink-0">
          <form onSubmit={handleSend} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              placeholder="Ask a question or input transaction command..."
              className="flex-1 bg-bg-surface border border-border-dim text-sm text-terminal-text px-4 py-2 focus:border-terminal-text focus:outline-none font-mono"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-4 py-2 border border-terminal-signal text-terminal-signal hover:bg-terminal-signal/10 text-sm font-bold flex items-center justify-center gap-1.5 transition-all disabled:opacity-50 disabled:pointer-events-none"
            >
              <Send size={14} />
              SEND
            </button>
          </form>
        </div>

      </div>

      {/* Citations & Real-Time Research Side Panel (Debate Hub Blueprint) */}
      <div className="w-full md:w-88 bg-bg-card border-t md:border-t-0 md:border-l border-border-dim flex flex-col shrink-0 overflow-hidden">
        
        {/* Drawer Header */}
        <div className="px-4 h-12 border-b border-border-dim flex items-center justify-between bg-bg-card shrink-0">
          <div className="flex items-center gap-2">
            <FileText size={16} className="text-terminal-violet" />
            <h3 className="text-xs font-bold text-terminal-text uppercase tracking-wider">
              Grounded_Citations
            </h3>
          </div>
          {researchPhase === "searching" && (
            <span className="text-[9px] text-terminal-signal tracking-widest animate-pulse flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-terminal-signal"></span>
              LIVE SEARCH
            </span>
          )}
          {researchPhase === "complete" && (
            <span className="text-[9px] text-terminal-muted tracking-widest">
              Done
            </span>
          )}
        </div>

        {/* Live Search Query Banner (if web search is running or completed) */}
        {researchQuery && (
          <div className="px-4 py-2 border-b border-border-dim/40 bg-bg-surface text-[10px] font-mono flex items-center justify-between shrink-0">
            <span className="text-terminal-muted truncate max-w-[200px]" title={researchQuery}>
              Query: {researchQuery}
            </span>
            {Math.max(researchTotal, currentSources.length) > 0 && (
              <span className="text-terminal-signal font-bold shrink-0">
                {currentSources.length}/{Math.max(researchTotal, currentSources.length)}
              </span>
            )}
          </div>
        )}

        {/* Citations List with Animated Source Cards (Debate Hub Style) */}
        <div className="flex-1 overflow-auto p-4 space-y-3 scrollbar-thin bg-bg-main">
          {currentSources.length === 0 && researchPhase !== "searching" && (
            <div className="h-full flex flex-col items-center justify-center text-center text-xs text-terminal-muted p-4 space-y-2">
              <AlertTriangle size={24} className="text-terminal-muted/40" />
              <span>No active query citations retrieved.</span>
            </div>
          )}

          {currentSources.map((art, idx) => (
            <div
              key={idx}
              className="border border-terminal-signal/20 bg-bg-surface/40 p-3 rounded-sm flex items-start gap-3 hover:border-terminal-signal/50 transition-all duration-300 group"
              style={{
                animation: `slideUpFade 0.35s ease-out ${idx * 60}ms both`,
              }}
            >
              {/* Domain / Source Icon */}
              <div className="shrink-0 mt-0.5">
                <div className="w-8 h-8 rounded-sm bg-terminal-signal/10 border border-terminal-signal/30 flex items-center justify-center group-hover:bg-terminal-signal/20 transition-colors">
                  {art.url ? (
                    <ExternalLink size={14} className="text-terminal-signal" />
                  ) : (
                    <Globe size={14} className="text-terminal-violet" />
                  )}
                </div>
              </div>

              {/* Source Details */}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 justify-between">
                  <span className="text-[9px] text-terminal-signal uppercase tracking-wider font-bold truncate">
                    {art.domain || (art.source_type === "web" ? "WEB SEARCH" : "IN-HOUSE")}
                  </span>
                  {art.importance_score ? (
                    <span className="text-[8px] border border-terminal-signal/50 text-terminal-signal px-1 py-0.2 rounded-xs font-mono">
                      IMP: {art.importance_score}
                    </span>
                  ) : (
                    <span className="text-[8px] text-terminal-signal/60 tracking-widest shrink-0">
                      Retrieved
                    </span>
                  )}
                </div>

                <p className="text-xs text-terminal-text mt-1 leading-relaxed font-sans line-clamp-2 font-medium">
                  {art.headline || art.title}
                </p>

                {art.summary && (
                  <p className="text-[10px] text-terminal-muted mt-1 leading-normal font-sans line-clamp-2">
                    {art.summary}
                  </p>
                )}

                {art.url && (
                  <a
                    href={art.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[9px] text-terminal-muted/70 font-mono mt-1 truncate block hover:text-terminal-signal transition-colors flex items-center gap-1"
                    title={art.url}
                  >
                    <span>{art.url.length > 45 ? art.url.slice(0, 45) + "..." : art.url}</span>
                  </a>
                )}
              </div>
            </div>
          ))}

          {/* Placeholder card during active search */}
          {researchPhase === "searching" && (
            <div className="border border-border-dim/30 bg-bg-surface/20 p-3 rounded-sm animate-pulse flex items-center gap-3">
              <div className="w-8 h-8 rounded-sm bg-bg-surface border border-border-dim/30 flex items-center justify-center">
                <Loader2 size={14} className="text-terminal-muted animate-spin" />
              </div>
              <div className="flex-1 space-y-1.5">
                <div className="h-2.5 bg-bg-surface rounded w-20" />
                <div className="h-3 bg-bg-surface rounded w-48" />
                <div className="h-2 bg-bg-surface rounded w-32" />
              </div>
            </div>
          )}
        </div>

        {/* Progress Footer */}
        {Math.max(researchTotal, currentSources.length) > 0 && (
          <div className="px-4 py-2 border-t border-border-dim/40 bg-bg-card shrink-0 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-terminal-muted">
                {researchPhase === "searching"
                  ? `Retrieving source ${Math.min(currentSources.length + 1, Math.max(researchTotal, currentSources.length))} of ${Math.max(researchTotal, currentSources.length)}...`
                  : `${currentSources.length} source${currentSources.length !== 1 ? "s" : ""} retrieved`}
              </span>
              <span className="text-terminal-signal font-bold">
                {Math.min(Math.round((currentSources.length / Math.max(researchTotal, currentSources.length)) * 100), 100)}%
              </span>
            </div>
            <div className="w-full h-1 bg-bg-surface rounded-full overflow-hidden">
              <div
                className="h-full bg-terminal-signal rounded-full transition-all duration-500 ease-out"
                style={{ width: `${Math.min((currentSources.length / Math.max(researchTotal, currentSources.length)) * 100, 100)}%` }}
              />
            </div>
          </div>
        )}

      </div>

    </div>
  );
}

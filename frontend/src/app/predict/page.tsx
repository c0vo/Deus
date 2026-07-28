"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Terminal as TerminalIcon,
  ShieldAlert,
  Play,
  ChevronDown,
  ChevronUp,
  Swords,
  TrendingUp,
  TrendingDown,
  Sparkles,
  Cpu,
  Columns,
  MessageSquare,
  Clock,
  ExternalLink,
  Globe,
  Search,
  Loader2
} from "lucide-react";
import FormattedText from "../components/FormattedText";
import { getApiUrl } from "../utils/api";

interface FeatureDict {
  [key: string]: number;
}

interface DebateVerdict {
  ticker: string;
  predicted_direction?: string;
  confidence?: number;
  final_advisory?: string;
  bull_report?: string;
  bear_report?: string;
  ml_prediction?: {
    predicted_direction?: string;
    confidence?: number;
    feature_snapshot?: string; // JSON string
  };
  debate_history?: string[];
  executive_summary?: string;
}

interface ResearchSource {
  title: string;
  url: string;
  domain: string;
}

function PredictContent() {
  const searchParams = useSearchParams();
  const [ticker, setTicker] = useState("");
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const [verdict, setVerdict] = useState<DebateVerdict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<FeatureDict>({});
  const [featuresExpanded, setFeaturesExpanded] = useState(false);

  // History states
  const [historyDates, setHistoryDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");

  // Structured round-by-round state
  const [bullRounds, setBullRounds] = useState<{ [round: number]: string }>({ 1: "", 2: "" });
  const [bearRounds, setBearRounds] = useState<{ [round: number]: string }>({ 1: "", 2: "" });
  const [streamedTrader, setStreamedTrader] = useState("");
  const [currentSpeaker, setCurrentSpeaker] = useState<"bull" | "bear" | "trader" | null>(null);
  const [currentRound, setCurrentRound] = useState<number>(0);
  const [viewMode, setViewMode] = useState<"arena" | "split">("arena");
  const [forceRefresh, setForceRefresh] = useState(false);

  // Tab state
  const [activeTab, setActiveTab] = useState<"new" | "recent">("new");
  const [recentDebates, setRecentDebates] = useState<{ticker: string; date: string}[]>([]);
  const [recentDebatesLoading, setRecentDebatesLoading] = useState(false);
  const [recentFilter, setRecentFilter] = useState("");

  // Research visualization state
  const [researchPhase, setResearchPhase] = useState<"idle" | "searching" | "summarizing" | "complete">("idle");
  const [researchSources, setResearchSources] = useState<ResearchSource[]>([]);
  const [researchQuery, setResearchQuery] = useState("");
  const [researchMsg, setResearchMsg] = useState("");
  const [researchTotal, setResearchTotal] = useState(0);

  const logsEndRef = useRef<HTMLDivElement>(null);
  const arenaEndRef = useRef<HTMLDivElement>(null);

  const fetchHistoryDates = async (symbol: string) => {
    if (!symbol) return;
    try {
      const res = await fetch(getApiUrl(`/api/predict/${symbol}/history`));
      if (res.ok) {
        const json = await res.json();
        setHistoryDates(json.data || []);
      } else {
        setHistoryDates([]);
      }
    } catch (err) {
      console.error("Failed to fetch history dates", err);
      setHistoryDates([]);
    }
  };

  const loadDebateByTickerDate = async (targetTicker: string, dateStr: string) => {
    if (!dateStr) return;
    setRunning(true);
    setError(null);
    setVerdict(null);
    setFeatures({});
    setBullRounds({ 1: "", 2: "" });
    setBearRounds({ 1: "", 2: "" });
    setStreamedTrader("");
    setCurrentSpeaker(null);
    setCurrentRound(0);
    setResearchPhase("idle");
    setResearchSources([]);
    setResearchQuery("");
    setResearchMsg("");
    setResearchTotal(0);
    setLogs([`Fetching debate details for ${targetTicker} on ${dateStr}…`]);
    try {
      const res = await fetch(getApiUrl(`/api/predict/${targetTicker}/history/${dateStr}`));
      if (!res.ok) {
        throw new Error(`Failed to load historical debate: ${res.statusText}`);
      }
      const json = await res.json();
      const stateObj = json.data;

      const debateHistory = stateObj.debate_history || [];
      const mappedVerdict: DebateVerdict = {
        ticker: targetTicker,
        predicted_direction: stateObj.ml_prediction?.predicted_direction || "UNKNOWN",
        confidence: stateObj.ml_prediction?.confidence || 0.0,
        final_advisory: stateObj.final_advisory,
        bull_report: stateObj.bull_report,
        bear_report: stateObj.bear_report,
        ml_prediction: stateObj.ml_prediction,
        debate_history: debateHistory,
      };

      setVerdict(mappedVerdict);

      // Parse debate history round-by-round
      const parsedBullRounds: { [key: number]: string } = {};
      const parsedBearRounds: { [key: number]: string } = {};
      let bullCount = 1;
      let bearCount = 1;

      if (debateHistory.length > 0) {
        for (const line of debateHistory) {
          if (line.startsWith("Bull: ")) {
            parsedBullRounds[bullCount] = line.substring(6);
            bullCount++;
          } else if (line.startsWith("Bear: ")) {
            parsedBearRounds[bearCount] = line.substring(6);
            bearCount++;
          }
        }
      } else {
        // Fallback split
        if (stateObj.bull_report) {
          const parts = stateObj.bull_report.split("\n\n");
          parsedBullRounds[1] = parts[0] || "";
          parsedBullRounds[2] = parts.slice(1).join("\n\n") || "";
        }
        if (stateObj.bear_report) {
          const parts = stateObj.bear_report.split("\n\n");
          parsedBearRounds[1] = parts[0] || "";
          parsedBearRounds[2] = parts.slice(1).join("\n\n") || "";
        }
      }

      setBullRounds(parsedBullRounds);
      setBearRounds(parsedBearRounds);

      if (stateObj.ml_prediction?.feature_snapshot) {
        try {
          const feat = JSON.parse(stateObj.ml_prediction.feature_snapshot);
          setFeatures(feat);
        } catch (err) {
          console.error("Failed to parse features", err);
        }
      }

      setLogs((prev) => [...prev, `Debate loaded.`]);
    } catch (err: any) {
      setError(err.message || "Failed to load historical debate.");
    } finally {
      setRunning(false);
    }
  };

  const handleSelectHistoryDate = async (dateStr: string) => {
    setSelectedDate(dateStr);
    if (!dateStr) return;
    await loadDebateByTickerDate(ticker, dateStr);
  };

  const fetchRecentDebates = async () => {
    setRecentDebatesLoading(true);
    try {
      const res = await fetch(getApiUrl("/api/predict/history/recent"));
      if (res.ok) {
        const json = await res.json();
        setRecentDebates(json.data || []);
      } else {
        setRecentDebates([]);
      }
    } catch (err) {
      console.error("Failed to fetch recent debates", err);
      setRecentDebates([]);
    } finally {
      setRecentDebatesLoading(false);
    }
  };

  useEffect(() => {
    const symbol = searchParams.get("ticker");
    if (symbol) {
      setTicker(symbol.toUpperCase());
      fetchHistoryDates(symbol.toUpperCase());
    }
  }, [searchParams]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  useEffect(() => {
    if (arenaEndRef.current && running) {
      arenaEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [bullRounds, bearRounds, streamedTrader, running]);

  const runAnalysis = async () => {
    const symbol = ticker.toUpperCase().trim();
    if (!symbol) return;

    setRunning(true);
    setLogs([]);
    setVerdict(null);
    setError(null);
    setFeatures({});
    setSelectedDate("");
    setBullRounds({ 1: "", 2: "" });
    setBearRounds({ 1: "", 2: "" });
    setStreamedTrader("");
    setCurrentSpeaker(null);
    setCurrentRound(0);
    setResearchPhase("idle");
    setResearchSources([]);
    setResearchQuery("");
    setResearchMsg("");
    setResearchTotal(0);
    setLogs([`Starting prediction agents for ${symbol}…`, `Connecting…`]);

    try {
      const response = await fetch(getApiUrl(`/api/predict/${symbol}/stream?refresh=${forceRefresh}`));
      if (!response.ok) {
        throw new Error(`Predict API error: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("Response body is not readable");

      const decoder = new TextDecoder("utf-8");
      let partialLine = "";
      let currentEvent = "";

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
            const dataStr = line.substring(6).trim();

            if (currentEvent === "agent_update") {
              setLogs((prev) => [...prev, `${dataStr}`]);
            } else if (currentEvent === "research_start") {
              try {
                const parsed = JSON.parse(dataStr);
                setResearchPhase("searching");
                setResearchQuery(parsed.query || "");
                setResearchSources([]);
                setResearchTotal(0);
              } catch (err) {
                console.error("Failed to parse research_start", err);
              }
            } else if (currentEvent === "research_source") {
              try {
                const parsed = JSON.parse(dataStr);
                setResearchTotal(parsed.total || 0);
                setResearchSources((prev) => [
                  ...prev,
                  {
                    title: parsed.title || "",
                    url: parsed.url || "",
                    domain: parsed.domain || "",
                  },
                ]);
              } catch (err) {
                console.error("Failed to parse research_source", err);
              }
            } else if (currentEvent === "research_summarizing") {
              try {
                const parsed = JSON.parse(dataStr);
                setResearchPhase("summarizing");
                setResearchMsg(parsed.message || "");
              } catch (err) {
                console.error("Failed to parse research_summarizing", err);
              }
            } else if (currentEvent === "research_complete") {
              try {
                const parsed = JSON.parse(dataStr);
                setResearchPhase("complete");
                if (parsed.sources_found !== undefined && parsed.sources_found >= 0) {
                  setResearchTotal(parsed.sources_found);
                }
              } catch (err) {
                console.error("Failed to parse research_complete", err);
              }
            } else if (currentEvent === "debate_chunk") {
              try {
                const parsed = JSON.parse(dataStr);
                const spk = parsed.speaker;
                const txt = parsed.text;
                const rnd = parsed.round || 1;
                
                setCurrentSpeaker(spk);
                setCurrentRound(rnd);

                if (spk === "bull") {
                  setBullRounds((prev) => ({
                    ...prev,
                    [rnd]: (prev[rnd] || "") + txt
                  }));
                } else if (spk === "bear") {
                  setBearRounds((prev) => ({
                    ...prev,
                    [rnd]: (prev[rnd] || "") + txt
                  }));
                } else if (spk === "trader") {
                  setStreamedTrader((prev) => prev + txt);
                }
              } catch (err) {
                console.error("Failed to parse debate chunk", err);
              }
            } else if (currentEvent === "verdict") {
              try {
                const res = JSON.parse(dataStr);
                setVerdict(res);
                
                // Finalize round data from verdict if needed
                const debateHistory = res.debate_history || [];
                const finalBull: { [key: number]: string } = {};
                const finalBear: { [key: number]: string } = {};
                let bIdx = 1;
                let rIdx = 1;
                for (const line of debateHistory) {
                  if (line.startsWith("Bull: ")) {
                    finalBull[bIdx++] = line.substring(6);
                  } else if (line.startsWith("Bear: ")) {
                    finalBear[rIdx++] = line.substring(6);
                  }
                }
                setBullRounds(finalBull);
                setBearRounds(finalBear);
                setCurrentSpeaker(null);
                setCurrentRound(0);

                if (res.ml_prediction?.feature_snapshot) {
                  try {
                    const feat = JSON.parse(res.ml_prediction.feature_snapshot);
                    setFeatures(feat);
                  } catch (err) {
                    console.error("Failed to parse features", err);
                  }
                }
                setLogs((prev) => [...prev, `Verdict generated.`]);
              } catch (err) {
                console.error("Failed to parse verdict", err);
              }
            } else if (currentEvent === "error") {
              setError(dataStr || "An error occurred during prediction.");
            } else if (currentEvent === "done") {
              // Stream complete
            }

            currentEvent = "";
          }
        }
      }

      fetchHistoryDates(symbol);
      setRunning(false);
    } catch (err: any) {
      setError(err.message || "Failed to start debate stream.");
      setRunning(false);
    }
  };

  const getStatusText = () => {
    if (!running) return "ARENA STANDBY";
    if (currentSpeaker === "bull") return `ROUND ${currentRound} - BULLISH RESEARCHER PROPOSING`;
    if (currentSpeaker === "bear") return `ROUND ${currentRound} - BEARISH RESEARCHER CHALLENGING`;
    if (currentSpeaker === "trader") return "VERDICT SYNTHESIS - TRADER MANAGER EVALUATING";
    return "ORCHESTRATING WORKFLOW...";
  };

  const hasDebateContent = bullRounds[1] || bearRounds[1] || bullRounds[2] || bearRounds[2] || verdict || streamedTrader;

  return (
    <div className="p-6 space-y-6">
      
      {/* Title */}
      <div className="border-b border-border-dim pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-terminal-text uppercase tracking-wider flex items-center gap-2">
            <Swords className="text-terminal-violet" size={24} />
            Debate
          </h1>
          <p className="text-xs text-terminal-muted">
            Launch multi-agent debate rooms and evaluate trade advisories
          </p>
        </div>

        {/* Console toggle & view toggles */}
        {hasDebateContent && (
          <div className="flex items-center gap-3 text-xs">
            <div className="flex border border-border-dim">
              <button
                onClick={() => setViewMode("arena")}
                className={`px-3 py-1.5 font-bold uppercase transition-colors flex items-center gap-1.5 ${
                  viewMode === "arena"
                    ? "bg-terminal-signal text-terminal-on-signal"
                    : "hover:bg-bg-surface text-terminal-text"
                }`}
              >
                <MessageSquare size={13} />
                Arena Thread
              </button>
              <button
                onClick={() => setViewMode("split")}
                className={`px-3 py-1.5 font-bold uppercase border-l border-border-dim transition-colors flex items-center gap-1.5 ${
                  viewMode === "split"
                    ? "bg-terminal-signal text-terminal-on-signal"
                    : "hover:bg-bg-surface text-terminal-text"
                }`}
              >
                <Columns size={13} />
                Split Panels
              </button>
            </div>
            
            <button
              onClick={() => setShowLogs(!showLogs)}
              className={`px-3 py-1.5 border border-border-dim hover:border-terminal-text font-bold uppercase transition-colors flex items-center gap-1.5 ${
                showLogs ? "bg-bg-surface text-terminal-text border-terminal-text" : "text-terminal-muted"
              }`}
            >
              <TerminalIcon size={13} />
              Logs
            </button>
          </div>
        )}
      </div>

      {/* === TAB NAVIGATION === */}
      <div className="flex border-b border-border-dim">
        <button
          onClick={() => setActiveTab("new")}
          className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition-colors flex items-center gap-1.5 ${
            activeTab === "new"
              ? "border-b-2 border-terminal-signal text-terminal-signal bg-bg-surface"
              : "text-terminal-muted hover:text-terminal-text border-b-2 border-transparent"
          }`}
        >
          <Swords size={13} />
          New Debate
        </button>
        <button
          onClick={() => {
            setActiveTab("recent");
            if (recentDebates.length === 0 && !recentDebatesLoading) {
              fetchRecentDebates();
            }
          }}
          className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition-colors flex items-center gap-1.5 ${
            activeTab === "recent"
              ? "border-b-2 border-terminal-signal text-terminal-signal bg-bg-surface"
              : "text-terminal-muted hover:text-terminal-text border-b-2 border-transparent"
          }`}
        >
          <Clock size={13} />
          Recent Debates
        </button>
      </div>

      {/* === TAB CONTENT === */}
      {activeTab === "new" ? (
        /* ── New Debate controls ── */
        <div className="border border-border-dim bg-bg-card p-4 flex flex-col md:flex-row items-stretch md:items-center gap-4">
          <div className="flex-1 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <div className="flex items-center gap-3">
              <span className="text-xs text-terminal-text font-bold uppercase shrink-0">ENTER TICKER:</span>
              <input
                type="text"
                value={ticker}
                onChange={(e) => {
                  setTicker(e.target.value);
                  setSelectedDate("");
                  setHistoryDates([]);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && ticker && !running) {
                    runAnalysis();
                  }
                }}
                onBlur={() => fetchHistoryDates(ticker.toUpperCase().trim())}
                disabled={running}
                placeholder="e.g. AAPL"
                className="w-[120px] bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none uppercase"
              />
            </div>

            {historyDates.length > 0 && (
              <div className="flex items-center gap-2 mt-2 sm:mt-0">
                <Clock size={12} className="text-terminal-violet" />
                <span className="text-[10px] text-terminal-muted uppercase tracking-wider shrink-0">PAST DEBATES:</span>
                <select
                  value={selectedDate}
                  onChange={(e) => handleSelectHistoryDate(e.target.value)}
                  disabled={running}
                  className="bg-bg-surface border border-border-dim text-xs text-terminal-text px-2 py-1.5 focus:border-terminal-text focus:outline-none font-mono"
                >
                  <option value="">-- select date --</option>
                  {historyDates.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="flex items-center gap-4 mt-2 sm:mt-0">
            <label className="flex items-center gap-1.5 cursor-pointer select-none text-xs text-terminal-muted hover:text-terminal-text transition-colors">
              <input
                type="checkbox"
                checked={forceRefresh}
                onChange={(e) => setForceRefresh(e.target.checked)}
                disabled={running}
                className="accent-terminal-signal h-3.5 w-3.5 border-border-dim bg-bg-surface rounded-sm focus:ring-0 cursor-pointer"
              />
              <span className="text-[10px] tracking-wider">Force refresh</span>
            </label>

            <button
              onClick={runAnalysis}
              disabled={running || !ticker}
              className="px-6 py-2 border border-terminal-signal text-terminal-signal hover:bg-terminal-signal/10 text-sm font-bold flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:pointer-events-none"
            >
              <Play size={16} />
              RUN DEBATE ANALYSIS
            </button>
          </div>
        </div>
      ) : (
        /* ── Recent Debates list ── */
        <div className="border border-border-dim bg-bg-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs text-terminal-muted uppercase tracking-wider">
              [All Past Debates]
            </h3>
            {recentDebates.length > 0 && (
              <button
                onClick={fetchRecentDebates}
                className="text-[10px] text-terminal-muted hover:text-terminal-text border border-border-dim px-2 py-0.5 transition-colors"
              >
                REFRESH
              </button>
            )}
          </div>

          <input
            type="text"
            value={recentFilter}
            onChange={(e) => setRecentFilter(e.target.value)}
            placeholder="Filter by ticker..."
            className="w-full bg-bg-surface border border-border-dim text-xs text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none uppercase"
          />

          {recentDebatesLoading && (
            <div className="text-center py-8 text-xs text-terminal-muted animate-pulse">
              Loading debate history...
            </div>
          )}

          {!recentDebatesLoading && recentDebates.length === 0 && (
            <div className="text-center py-8 text-xs text-terminal-muted">
              No past debates found. Run a debate analysis to get started.
            </div>
          )}

          {!recentDebatesLoading && recentDebates.length > 0 && (
            <div className="max-h-[420px] overflow-y-auto space-y-1 scrollbar-thin">
              {recentDebates
                .filter(
                  (d) =>
                    !recentFilter ||
                    d.ticker.includes(recentFilter.toUpperCase())
                )
                .map((debate) => (
                  <button
                    key={`${debate.ticker}-${debate.date}`}
                    onClick={() => {
                      setTicker(debate.ticker);
                      setSelectedDate(debate.date);
                      setActiveTab("new");
                      loadDebateByTickerDate(debate.ticker, debate.date);
                      fetchHistoryDates(debate.ticker);
                    }}
                    className="w-full text-left px-3 py-2.5 border border-border-dim hover:border-terminal-text hover:bg-bg-surface transition-colors flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-terminal-text text-sm font-mono">
                        {debate.ticker}
                      </span>
                      <span className="text-[10px] text-terminal-muted font-mono">
                        {debate.date}
                      </span>
                    </div>
                    <ExternalLink
                      size={12}
                      className="text-terminal-muted opacity-0 group-hover:opacity-100 transition-opacity"
                    />
                  </button>
                ))}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="border border-terminal-error bg-bg-card p-4 flex items-start gap-3 text-sm text-terminal-error animate-shake">
          <ShieldAlert size={20} className="shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold uppercase mb-1">Execution Failure</h4>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Loading animation if running but no content returned yet */}
      {running && !hasDebateContent && researchPhase === "idle" && (
        <div className="border border-border-dim bg-bg-card p-12 flex flex-col items-center justify-center gap-6 min-h-[300px]">
          <div className="relative w-20 h-20">
            <div className="absolute inset-0 rounded-full border-4 border-terminal-signal/10 border-t-terminal-signal animate-spin" />
            <div className="absolute inset-2 rounded-full border-4 border-terminal-violet/10 border-b-terminal-violet animate-spin [animation-duration:1.5s]" />
            <div className="absolute inset-0 flex items-center justify-center text-terminal-signal">
              <Swords size={28} className="animate-pulse" />
            </div>
          </div>
          <div className="text-center space-y-2">
            <h4 className="text-sm font-bold text-terminal-text uppercase tracking-widest animate-pulse">
              Assembling AI Agent Arena...
            </h4>
            <p className="text-xs text-terminal-muted max-w-md mx-auto">
              The debate agents are gathering market intelligence, sentiment feeds, and technical indicator reports to draft their arguments.
            </p>
          </div>
          <div className="w-full max-w-lg mt-4 border border-border-dim bg-bg-main p-3 font-mono text-[10px] text-terminal-signal/80 overflow-y-auto max-h-[120px] rounded-sm text-left">
            {logs.slice(-3).map((logLine, idx) => (
              <div key={idx} className="truncate">{logLine}</div>
            ))}
            <div className="animate-pulse">_</div>
          </div>
        </div>
      )}

      {/* Research Agent Panel — animated source cards as web search happens */}
      {running && !hasDebateContent && researchPhase !== "idle" && (
        <div className="border border-terminal-signal/30 bg-bg-card p-6 space-y-4 animate-fadeIn">
          {/* Header */}
          <div className="flex items-center gap-3 border-b border-border-dim pb-3">
            <div className="relative">
              <Globe size={20} className="text-terminal-signal" />
              {researchPhase === "searching" && (
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-terminal-signal rounded-full animate-ping" />
              )}
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-sm font-bold text-terminal-signal uppercase tracking-wider flex items-center gap-2">
                Research Agent Deployed
                {researchPhase === "searching" && (
                  <span className="text-[9px] text-terminal-signal animate-pulse tracking-widest">
                    Fetching
                  </span>
                )}
                {researchPhase === "summarizing" && (
                  <span className="text-[9px] text-terminal-violet animate-pulse tracking-widest">
                    Analyzing
                  </span>
                )}
                {researchPhase === "complete" && (
                  <span className="text-[9px] text-terminal-muted tracking-widest">
                    Done
                  </span>
                )}
              </h3>
              <p className="text-[10px] text-terminal-muted font-mono truncate max-w-lg">
                Query: {researchQuery}
              </p>
            </div>
            <div className="shrink-0">
              <Search size={14} className="text-terminal-muted animate-pulse" />
            </div>
          </div>

          {/* Source Cards */}
          <div className="space-y-2 max-h-[340px] overflow-y-auto pr-1 scrollbar-thin">
            {researchSources.map((src, idx) => (
              <div
                key={idx}
                className="border border-terminal-signal/20 bg-bg-surface/40 p-3 rounded-sm flex items-start gap-3 hover:border-terminal-signal/50 transition-all duration-300 group"
                style={{
                  animation: `slideUpFade 0.35s ease-out ${idx * 100}ms both`,
                }}
              >
                {/* Domain icon */}
                <div className="shrink-0 mt-0.5">
                  <div className="w-8 h-8 rounded-sm bg-terminal-signal/10 border border-terminal-signal/30 flex items-center justify-center group-hover:bg-terminal-signal/20 transition-colors">
                    <ExternalLink size={14} className="text-terminal-signal" />
                  </div>
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] text-terminal-signal uppercase tracking-wider font-bold">
                      {src.domain}
                    </span>
                    <span className="text-[8px] text-terminal-signal/60 tracking-widest">
                      Retrieved
                    </span>
                  </div>
                  <p className="text-xs text-terminal-text mt-0.5 leading-relaxed line-clamp-2">
                    {src.title}
                  </p>
                  <a
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[9px] text-terminal-muted/50 font-mono mt-0.5 truncate block hover:text-terminal-signal/70 transition-colors"
                    title={src.url}
                  >
                    {src.url.length > 60 ? src.url.slice(0, 60) + "..." : src.url}
                  </a>
                </div>
              </div>
            ))}

            {/* Placeholder for in-flight sources */}
            {researchPhase === "searching" && researchTotal > 0 && researchSources.length < researchTotal && (
              <div className="border border-border-dim/30 bg-bg-surface/20 p-3 rounded-sm animate-pulse flex items-center gap-3">
                <div className="w-8 h-8 rounded-sm bg-bg-surface border border-border-dim/30 flex items-center justify-center">
                  <Loader2 size={14} className="text-terminal-muted animate-spin" />
                </div>
                <div className="flex-1 space-y-1.5">
                  <div className="h-2.5 bg-bg-surface rounded w-20" />
                  <div className="h-3 bg-bg-surface rounded w-56" />
                  <div className="h-2 bg-bg-surface rounded w-40" />
                </div>
              </div>
            )}

            {/* Empty state: no sources found */}
            {researchPhase === "complete" && researchSources.length === 0 && (
              <div className="text-center py-6 text-xs text-terminal-muted font-mono">
                No web sources found for this ticker. Using in-house news only.
              </div>
            )}
          </div>

          {/* Progress footer */}
          <div className="space-y-1.5 pt-1">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-terminal-muted">
                {researchPhase === "searching" && researchTotal > 0
                  ? `Retrieving source ${Math.min(researchSources.length + 1, researchTotal)} of ${researchTotal}...`
                  : researchPhase === "searching"
                  ? "Searching the web..."
                  : researchPhase === "summarizing"
                  ? researchMsg
                  : researchSources.length > 0
                  ? `${researchSources.length} source${researchSources.length !== 1 ? "s" : ""} analyzed`
                  : "Research complete"}
              </span>
              {researchTotal > 0 && (
                <span className="text-terminal-signal font-bold">
                  {researchSources.length}/{researchTotal}
                </span>
              )}
            </div>
            {researchTotal > 0 && (
              <div className="w-full h-1 bg-bg-surface rounded-full overflow-hidden">
                <div
                  className="h-full bg-terminal-signal rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${(researchSources.length / researchTotal) * 100}%` }}
                />
              </div>
            )}
            {researchPhase === "summarizing" && (
              <div className="w-full h-1 bg-bg-surface rounded-full overflow-hidden">
                <div className="h-full w-2/3 bg-terminal-violet rounded-full animate-pulse" />
              </div>
            )}
          </div>

          {/* Mini log tail */}
          <div className="border-t border-border-dim/30 pt-2">
            <div className="font-mono text-[10px] text-terminal-signal/60 overflow-y-auto max-h-[60px] rounded-sm text-left space-y-0.5">
              {logs.slice(-2).map((logLine, idx) => (
                <div key={idx} className="truncate">{logLine}</div>
              ))}
              <div className="animate-pulse text-terminal-signal">_</div>
            </div>
          </div>
        </div>
      )}

      {/* Main Face-Off Arena Dashboard */}
      {hasDebateContent && (
        <div className="space-y-6">
          
          {/* Visual Face-Off Header Panel */}
          <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-center gap-4 border border-border-dim bg-bg-main p-6 overflow-hidden relative">
            
            {/* Background Grid Pattern */}
            <div className="absolute inset-0 bg-grid-pattern opacity-5 pointer-events-none" />

            {/* Left Fighter: Bull Agent */}
            <div 
              className={`border border-terminal-green bg-bg-card p-4 transition-all duration-300 rounded-sm relative ${
                currentSpeaker === "bear" ? "opacity-40 scale-[0.98] blur-[0.5px]" : "opacity-100 scale-100"
              } ${
                currentSpeaker === "bull" ? "border-terminal-green ring-1 ring-terminal-green/30" : "border-border-dim"
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-sm bg-terminal-green/10 border border-terminal-green/30 flex items-center justify-center text-terminal-green">
                  <TrendingUp size={22} className={currentSpeaker === "bull" ? "animate-bounce" : ""} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-terminal-green uppercase tracking-wider flex items-center gap-1.5">
                    Bull agent
                  </h3>
                  <span className="text-[10px] text-terminal-muted font-mono block">deepseek-v4-pro</span>
                </div>
              </div>
              {currentSpeaker === "bull" && (
                <div className="absolute top-2 right-2 text-[9px] bg-terminal-green/20 border border-terminal-green text-terminal-green px-1.5 py-0.5 rounded-sm animate-pulse tracking-widest font-bold">
                  SPEAKING
                </div>
              )}
            </div>

            {/* Center clash VS indicator */}
            <div className="flex flex-col items-center justify-center min-w-[140px] py-2 md:py-0">
              <div className="w-12 h-12 rounded-full border-2 border-border-dim bg-bg-card flex items-center justify-center relative shadow-lg">
                <Swords size={20} className={`text-terminal-violet ${running ? "animate-pulse" : ""}`} />
                {running && (
                  <span className="absolute inset-0 rounded-full border border-terminal-violet animate-ping opacity-75" />
                )}
              </div>
              <span className="text-[10px] font-bold text-terminal-violet tracking-widest mt-2 uppercase text-center animate-pulse">
                {getStatusText()}
              </span>
            </div>

            {/* Right Fighter: Bear Agent */}
            <div 
              className={`border border-terminal-red bg-bg-card p-4 transition-all duration-300 rounded-sm relative ${
                currentSpeaker === "bull" ? "opacity-40 scale-[0.98] blur-[0.5px]" : "opacity-100 scale-100"
              } ${
                currentSpeaker === "bear" ? "border-terminal-red ring-1 ring-terminal-red/30" : "border-border-dim"
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-sm bg-terminal-red/10 border border-terminal-red/30 flex items-center justify-center text-terminal-red">
                  <TrendingDown size={22} className={currentSpeaker === "bear" ? "animate-bounce" : ""} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-terminal-red uppercase tracking-wider flex items-center gap-1.5">
                    Bear agent
                  </h3>
                  <span className="text-[10px] text-terminal-muted font-mono block">deepseek-v4-pro</span>
                </div>
              </div>
              {currentSpeaker === "bear" && (
                <div className="absolute top-2 right-2 text-[9px] bg-terminal-red/20 border border-terminal-red text-terminal-red px-1.5 py-0.5 rounded-sm animate-pulse tracking-widest font-bold">
                  SPEAKING
                </div>
              )}
            </div>

          </div>

          {/* Collapsible log pane */}
          {showLogs && (
            <div className="border border-border-dim bg-bg-surface p-4 flex flex-col h-[260px] animate-fadeIn">
              <div className="flex items-center gap-2 text-xs text-terminal-muted border-b border-border-dim pb-2 mb-3 select-none">
                <TerminalIcon size={14} />
                <span>AGENT_WORKFLOW_STREAM.log</span>
                {running && <span className="text-terminal-green animate-pulse ml-auto">Live</span>}
              </div>
              <div className="flex-1 overflow-auto space-y-1 font-mono text-[10px] text-terminal-muted scrollbar-thin">
                {logs.map((log, index) => (
                  <div key={index} className="leading-relaxed whitespace-pre-wrap break-words" dangerouslySetInnerHTML={{ __html: log }} />
                ))}
                {running && (
                  <div className="flex items-center gap-1.5 text-terminal-green">
                    <span>Thinking…</span>
                    <span className="w-1.5 h-3.5 bg-terminal-green animate-pulse" />
                  </div>
                )}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}

          {/* VIEW: Arena Thread Mode (Alternating speech bubbles) */}
          {viewMode === "arena" && (
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2 scrollbar-thin">
              
              {/* Round 1 */}
              {(bullRounds[1] || bearRounds[1] || (currentRound === 1 && running)) && (
                <div className="space-y-4">
                  <div className="flex items-center justify-center">
                    <span className="px-3 py-1 border border-border-dim text-[10px] text-terminal-muted uppercase tracking-widest bg-bg-card">
                      --- ROUND 1: INITIAL CLAIMS ---
                    </span>
                  </div>

                  {/* Bull Round 1 */}
                  {(bullRounds[1] || (currentSpeaker === "bull" && currentRound === 1)) && (
                    <div className="flex justify-start animate-slideUp">
                      <div className="max-w-[85%] border border-terminal-green/30 bg-bg-card p-4 rounded-sm relative">
                        <div className="text-[10px] text-terminal-green font-bold uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                          <TrendingUp size={12} />
                          Bullish Case (Opening)
                        </div>
                        <FormattedText
                          text={bullRounds[1] || "Formulating opening argument..."}
                          className="text-xs text-terminal-text leading-relaxed font-sans"
                        />
                        {currentSpeaker === "bull" && currentRound === 1 && (
                          <span className="inline-block w-1.5 h-3.5 bg-terminal-green animate-pulse ml-1 align-middle" />
                        )}
                      </div>
                    </div>
                  )}

                  {/* Bear Round 1 */}
                  {(bearRounds[1] || (currentSpeaker === "bear" && currentRound === 1)) && (
                    <div className="flex justify-end animate-slideUp">
                      <div className="max-w-[85%] border border-terminal-red/30 bg-bg-card p-4 rounded-sm relative text-right">
                        <div className="text-[10px] text-terminal-red font-bold uppercase tracking-wider mb-1.5 flex items-center justify-end gap-1.5">
                          Bearish Attack (Opening Rebuttal)
                          <TrendingDown size={12} />
                        </div>
                        <FormattedText
                          text={bearRounds[1] || "Formulating opening attack..."}
                          className="text-xs text-terminal-text leading-relaxed font-sans text-left"
                        />
                        {currentSpeaker === "bear" && currentRound === 1 && (
                          <span className="inline-block w-1.5 h-3.5 bg-terminal-red animate-pulse ml-1 align-middle" />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Round 2 */}
              {(bullRounds[2] || bearRounds[2] || (currentRound === 2 && running)) && (
                <div className="space-y-4 pt-4 border-t border-border-dim/20">
                  <div className="flex items-center justify-center">
                    <span className="px-3 py-1 border border-border-dim text-[10px] text-terminal-muted uppercase tracking-widest bg-bg-card">
                      --- ROUND 2: REBUTTALS & DEFENSE ---
                    </span>
                  </div>

                  {/* Bull Round 2 */}
                  {(bullRounds[2] || (currentSpeaker === "bull" && currentRound === 2)) && (
                    <div className="flex justify-start animate-slideUp">
                      <div className="max-w-[85%] border border-terminal-green/30 bg-bg-card p-4 rounded-sm relative">
                        <div className="text-[10px] text-terminal-green font-bold uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                          <TrendingUp size={12} />
                          Bullish Case (Rebuttal & News Catalyst Analysis)
                        </div>
                        <FormattedText
                          text={bullRounds[2] || "Formulating counter-arguments..."}
                          className="text-xs text-terminal-text leading-relaxed font-sans"
                        />
                        {currentSpeaker === "bull" && currentRound === 2 && (
                          <span className="inline-block w-1.5 h-3.5 bg-terminal-green animate-pulse ml-1 align-middle" />
                        )}
                      </div>
                    </div>
                  )}

                  {/* Bear Round 2 */}
                  {(bearRounds[2] || (currentSpeaker === "bear" && currentRound === 2)) && (
                    <div className="flex justify-end animate-slideUp">
                      <div className="max-w-[85%] border border-terminal-red/30 bg-bg-card p-4 rounded-sm relative text-right">
                        <div className="text-[10px] text-terminal-red font-bold uppercase tracking-wider mb-1.5 flex items-center justify-end gap-1.5">
                          Bearish Attack (Final Defense Counter-Attack)
                          <TrendingDown size={12} />
                        </div>
                        <FormattedText
                          text={bearRounds[2] || "Formulating final challenge..."}
                          className="text-xs text-terminal-text leading-relaxed font-sans text-left"
                        />
                        {currentSpeaker === "bear" && currentRound === 2 && (
                          <span className="inline-block w-1.5 h-3.5 bg-terminal-red animate-pulse ml-1 align-middle" />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              <div ref={arenaEndRef} />
            </div>
          )}

          {/* VIEW: Split Column Mode */}
          {viewMode === "split" && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
              
              {/* Bullish card column */}
              <div className="space-y-4 border border-terminal-green/30 bg-bg-card p-4">
                <h4 className="text-xs font-bold text-terminal-green uppercase tracking-widest border-b border-terminal-green/20 pb-1.5 mb-2 flex items-center gap-1.5">
                  <TrendingUp size={14} />
                  Bull Researcher arguments
                </h4>
                
                {/* Round 1 */}
                {(bullRounds[1] || (currentSpeaker === "bull" && currentRound === 1)) && (
                  <div className="space-y-1">
                    <span className="text-[9px] text-terminal-muted uppercase block">Round 1 · Opening argument</span>
                    <div className="bg-bg-surface/40 p-3 border border-border-dim/50 rounded-sm">
                      <FormattedText
                        text={bullRounds[1] || "Formulating opening argument..."}
                        className="text-xs text-terminal-text leading-relaxed font-sans"
                      />
                      {currentSpeaker === "bull" && currentRound === 1 && (
                        <span className="inline-block w-1.5 h-3.5 bg-terminal-green animate-pulse ml-1 align-middle" />
                      )}
                    </div>
                  </div>
                )}

                {/* Round 2 */}
                {(bullRounds[2] || (currentSpeaker === "bull" && currentRound === 2)) && (
                  <div className="space-y-1">
                    <span className="text-[9px] text-terminal-muted uppercase block">Round 2 · Rebuttal</span>
                    <div className="bg-bg-surface/40 p-3 border border-border-dim/50 rounded-sm">
                      <FormattedText
                        text={bullRounds[2] || "Formulating rebuttal..."}
                        className="text-xs text-terminal-text leading-relaxed font-sans"
                      />
                      {currentSpeaker === "bull" && currentRound === 2 && (
                        <span className="inline-block w-1.5 h-3.5 bg-terminal-green animate-pulse ml-1 align-middle" />
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Bearish card column */}
              <div className="space-y-4 border border-terminal-red/30 bg-bg-card p-4">
                <h4 className="text-xs font-bold text-terminal-red uppercase tracking-widest border-b border-terminal-red/20 pb-1.5 mb-2 flex items-center gap-1.5">
                  <TrendingDown size={14} />
                  Bear Researcher arguments
                </h4>

                {/* Round 1 */}
                {(bearRounds[1] || (currentSpeaker === "bear" && currentRound === 1)) && (
                  <div className="space-y-1">
                    <span className="text-[9px] text-terminal-muted uppercase block">Round 1 · Counter-thesis</span>
                    <div className="bg-bg-surface/40 p-3 border border-border-dim/50 rounded-sm">
                      <FormattedText
                        text={bearRounds[1] || "Formulating opening attack..."}
                        className="text-xs text-terminal-text leading-relaxed font-sans"
                      />
                      {currentSpeaker === "bear" && currentRound === 1 && (
                        <span className="inline-block w-1.5 h-3.5 bg-terminal-red animate-pulse ml-1 align-middle" />
                      )}
                    </div>
                  </div>
                )}

                {/* Round 2 */}
                {(bearRounds[2] || (currentSpeaker === "bear" && currentRound === 2)) && (
                  <div className="space-y-1">
                    <span className="text-[9px] text-terminal-muted uppercase block">Round 2 · Counter-rebuttal</span>
                    <div className="bg-bg-surface/40 p-3 border border-border-dim/50 rounded-sm">
                      <FormattedText
                        text={bearRounds[2] || "Formulating final challenge..."}
                        className="text-xs text-terminal-text leading-relaxed font-sans"
                      />
                      {currentSpeaker === "bear" && currentRound === 2 && (
                        <span className="inline-block w-1.5 h-3.5 bg-terminal-red animate-pulse ml-1 align-middle" />
                      )}
                    </div>
                  </div>
                )}
              </div>

            </div>
          )}

          {/* Head Trader Verdict (Synthesized) */}
          {(verdict || streamedTrader) && (
            <div className="border border-terminal-signal bg-bg-card p-5 relative overflow-hidden animate-slideUp">
              <div className="absolute top-0 right-0 px-3 py-1 bg-terminal-signal text-terminal-on-signal text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5">
                <Sparkles size={11} className="animate-spin" />
                Head Trader Verdict
              </div>
              
              <h3 className="text-xs font-bold text-terminal-violet uppercase tracking-wider mb-4 flex items-center gap-1.5">
                <Cpu size={14} className="animate-pulse" />
                [Verbatim Gemini Synthesis & Risk Assessment]
              </h3>
              
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 bg-bg-surface/30 p-3 border border-border-dim/40">
                  <div>
                    <span className="text-[10px] text-terminal-muted uppercase block">Recommended direction</span>
                    <span className={`text-2xl font-black uppercase tracking-wide ${(verdict?.predicted_direction || verdict?.ml_prediction?.predicted_direction || "HOLD") === "UP" ? "text-terminal-green" : "text-terminal-red"}`}>
                      {verdict?.predicted_direction || verdict?.ml_prediction?.predicted_direction || "HOLD"}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-terminal-muted uppercase block">Advisory confidence</span>
                    <span className="text-2xl font-black text-terminal-text tracking-wide">
                      {verdict?.confidence !== undefined ? `${(verdict.confidence * 100).toFixed(1)}%` : (verdict?.ml_prediction?.confidence ? `${(verdict.ml_prediction.confidence * 100).toFixed(1)}%` : "N/A")}
                    </span>
                  </div>
                </div>

                <div className="border-t border-border-dim/30 pt-3">
                  <span className="text-[10px] text-terminal-muted uppercase block mb-1.5">Decision narrative</span>
                  <FormattedText
                    text={verdict?.final_advisory || streamedTrader}
                    className="text-sm text-terminal-text leading-relaxed font-sans"
                  />
                </div>
              </div>
            </div>
          )}

          {/* 23-Feature Vector Panel */}
          {Object.keys(features).length > 0 && (
            <div className="border border-border-dim bg-bg-card">
              <button
                onClick={() => setFeaturesExpanded(!featuresExpanded)}
                className="w-full p-4 flex items-center justify-between text-xs font-bold text-terminal-text uppercase tracking-wider hover:bg-bg-surface transition-colors"
              >
                <span>[23-Feature Quantitative Vector Snapshot]</span>
                {featuresExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              {featuresExpanded && (
                <div className="p-4 border-t border-border-dim grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 text-[10px] bg-bg-surface/30">
                  {Object.keys(features).sort().map((key) => {
                    const val = features[key];
                    return (
                      <div key={key} className="border border-border-dim/50 p-2 flex flex-col justify-between">
                        <span className="text-terminal-muted truncate uppercase">{key}</span>
                        <span className="text-terminal-text font-bold text-right mt-1">
                          {typeof val === "number" ? val.toFixed(4) : String(val)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

        </div>
      )}

    </div>
  );
}

export default function Predict() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-terminal-muted">Loading Debate Arena...</div>}>
      <PredictContent />
    </Suspense>
  );
}

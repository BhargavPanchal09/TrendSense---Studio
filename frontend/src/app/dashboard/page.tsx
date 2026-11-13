"use client";

import {
  useState, useRef, useEffect, DragEvent, ChangeEvent, useCallback,
} from "react";
import Link from "next/link";
import {
  ArrowLeft, Activity, Sparkles, BrainCircuit,
  BarChart3, Loader2, Send, Cpu, ImagePlus, X,
  Radio, Flame, Zap, TrendingUp, Eye, Users,
  AlertTriangle, CheckCircle2,
} from "lucide-react";

/* ── Types ── */
interface AnalyzeResponse {
  title: string;
  base_structural_score: number;
  historical_model_score?: number;
  evidence_adjustment?: {
    points: number; maximum_points: number;
    components: { youtube_chart: number; google_trends: number; news_coverage: number };
    matched_items: { youtube_chart: number; google_trends: number; news_publishers: number };
    method: string;
  };
  entity_tier: "S" | "A" | "B" | "None";
  entity_multiplier: number;
  final_virality_score: number;
  cohesion_score: number;
  thumbnail_analysis: string;
  thumbnail_analysis_status?: "available" | "unavailable";
  coach_mode?: "qwen_assisted" | "rule_based_fallback";
  ai_strategy_report: string;
  live_trends?: TrendFeed;
  topic_evidence?: TopicEvidence;
  search_interest?: SearchInterest;
  model_evaluation?: { test_mae: number; test_rows: number; within_10_points_percent: number; within_20_points_percent: number };
}

interface EngineHealth {
  api: string; core: string; ollama: string; llava: string; qwen: string;
  models?: { llava: string; qwen: string };
}

interface TrendTopic {
  topic: string; source: string; region: string; source_url: string; published_at: string; semantic_similarity?: number;
  growth?: { status: string; window_hours?: number; views_added?: number; views_per_hour?: number };
}
interface TrendFeed {
  status: string; region: string; fetched_at: string | null; history_status?: string;
  matches?: TrendTopic[]; momentum?: { status: string; label: string; interpretation: string; missing_sources?: string[] };
  sources?: { name: string; status: string; items: number }[];
  live_pulse?: {
    status: string; method: string; items_analyzed: number; ranking?: string; note: string;
    clusters: {
      topic_id: number; name: string; item_count: number; source_count: number;
      sources: string[]; regions: string[]; publisher_count?: number; country_count?: number;
      observed_views_per_hour: number | null;
      sample_items: { title: string; source: string; region: string; source_url: string; publisher?: string; country?: string; views?: number; chart_rank?: number }[];
    }[];
  };
}
interface TopicEvidence {
  query: string; status: string; window_hours: number; fetched_at: string | null;
  article_count: number; publisher_count: number;
  candidate_count?: number;
  sources?: { region: string; status: string; recent_items: number }[];
  articles: { title: string; url: string; publisher: string; published_at: string; semantic_similarity: number }[];
}
interface SearchInterest {
  status: string; topic: string; region: string; unique_browsers_7d: number;
  window_days: number; scope?: string; note: string;
}

function getAnonymousVisitorId(): string {
  const key = "trendsense_anonymous_browser_id";
  try {
    const existing = window.localStorage.getItem(key);
    if (existing) return existing;
    const generated = window.crypto?.randomUUID?.() ?? `browser-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    window.localStorage.setItem(key, generated);
    return generated;
  } catch {
    // Without durable browser storage, skip counting instead of creating a
    // fresh ID on every request and inflating the apparent audience.
    return "";
  }
}


/* ─────────────────────────────────────────────────────
   MARKDOWN RENDERER
   Handles **bold**, *italic*, bullet lists,
   numbered lists, ### headings — the exact format
   Llama 3 sends in ai_strategy_report.
   No external lib needed.
───────────────────────────────────────────────────── */
function renderInline(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i} className="text-cream font-semibold">{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i} className="text-cream/70 italic">{part.slice(1, -1)}</em>;
    return part;
  });
}

function MarkdownBlock({ text }: { text: string }) {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];

  lines.forEach((line, i) => {
    const t = line.trim();
    if (!t) { nodes.push(<div key={i} className="h-2" />); return; }

    // Section heading: ### or **Heading** alone on a line
    const isHeading = t.startsWith("###") ||
      (t.startsWith("**") && t.endsWith("**") && !t.slice(2, -2).includes("**") && t.length < 80);
    if (isHeading) {
      const heading = t.replace(/^#+\s*/, "").replace(/\*\*/g, "");
      nodes.push(
        <p key={i} className="text-gold text-[10px] font-mono uppercase tracking-widest mt-6 mb-2 first:mt-0">
          {heading}
        </p>
      );
      return;
    }

    // Bullet
    if (t.startsWith("* ") || t.startsWith("- ")) {
      nodes.push(
        <div key={i} className="flex gap-2 items-start pl-3">
          <span className="text-burgundy mt-[3px] shrink-0 text-base leading-none">·</span>
          <span className="text-cream/75 text-sm leading-relaxed">{renderInline(t.slice(2))}</span>
        </div>
      );
      return;
    }

    // Numbered list
    const numMatch = t.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      nodes.push(
        <div key={i} className="flex gap-3 items-start pl-3">
          <span className="text-gold font-mono text-xs mt-0.5 shrink-0 w-4">{numMatch[1]}.</span>
          <span className="text-cream/75 text-sm leading-relaxed">{renderInline(numMatch[2])}</span>
        </div>
      );
      return;
    }

    // Normal paragraph
    nodes.push(
      <p key={i} className="text-cream/75 text-sm leading-relaxed">
        {renderInline(t)}
      </p>
    );
  });

  return <div className="space-y-2">{nodes}</div>;
}

/* ── Entity Tier Badge ── */
function TierBadge({ tier }: { tier: "S" | "A" | "B" | "None" }) {
  const cfg = {
    S:    { label: "S-TIER ENTITY",   cls: "bg-burgundy/20 border-burgundy/50 text-burgundy" },
    A:    { label: "A-TIER ENTITY",   cls: "bg-gold/15 border-gold/50 text-gold"             },
    B:    { label: "B-TIER ENTITY",   cls: "bg-emerald/15 border-emerald/40 text-emerald"    },
    None: { label: "NO ENTITY BOOST", cls: "bg-white/5 border-cream/15 text-cream/35"        },
  }[tier];
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-mono uppercase tracking-widest ${cfg.cls}`}>
      <Zap className="w-3 h-3" />{cfg.label}
    </span>
  );
}

function cohesionLabel(score: number) {
  if (score >= 0.35) return { label: "Higher visual-text match · heuristic", color: "text-emerald", icon: <CheckCircle2 className="w-3.5 h-3.5" /> };
  if (score >= 0.28) return { label: "Moderate visual-text match · heuristic", color: "text-gold", icon: <CheckCircle2 className="w-3.5 h-3.5" /> };
  if (score >= 0.23) return { label: "Partial visual-text match · heuristic", color: "text-gold", icon: <AlertTriangle className="w-3.5 h-3.5" /> };
  if (score >= 0.21) return { label: "Low visual-text match · heuristic", color: "text-burgundy", icon: <AlertTriangle className="w-3.5 h-3.5" /> };
  return { label: "Very low visual-text match · heuristic", color: "text-red-600", icon: <AlertTriangle className="w-3.5 h-3.5" /> };
}

function ProgressBar({ value, color = "bg-burgundy", delay = 0 }: { value: number; color?: string; delay?: number }) {
  const [width, setWidth] = useState(0);
  useEffect(() => { const t = setTimeout(() => setWidth(value), delay); return () => clearTimeout(t); }, [value, delay]);
  return (
    <div className="w-full h-1 bg-emerald/10 rounded-full overflow-hidden">
      <div className={`h-full ${color} rounded-full transition-all duration-1000 ease-out`} style={{ width: `${width}%` }} />
    </div>
  );
}

const inputCls =
  "w-full bg-white border border-emerald/25 rounded-xl px-5 py-4 text-emerald placeholder:text-emerald/30 outline-none focus:border-burgundy focus:ring-2 focus:ring-burgundy/10 transition-all duration-300 text-sm font-medium shadow-sm";

/* ── Main ── */
export default function DashboardPage() {
  const [region, setRegion] = useState("MULTI");
  const [topicQuery, setTopicQuery] = useState("");
  const [trendFeed, setTrendFeed] = useState<TrendFeed | null>(null);
  const [engineHealth, setEngineHealth] = useState<EngineHealth | null>(null);
  useEffect(() => {
    let active = true;
    const refreshHealth = () => {
      fetch("http://localhost:8000/health")
        .then(response => { if (!response.ok) throw new Error("Health check failed"); return response.json(); })
        .then((health: EngineHealth) => { if (active) setEngineHealth(health); })
        .catch(() => { if (active) setEngineHealth({ api: "offline", core: "unavailable", ollama: "unknown", llava: "unknown", qwen: "unknown" }); });
    };
    refreshHealth();
    const timer = setInterval(refreshHealth, 15000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`http://localhost:8000/trends?region=${region}`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error("Unavailable"); return response.json(); })
      .then(setTrendFeed)
      .catch(() => { if (!controller.signal.aborted) setTrendFeed(null); });
    return () => controller.abort();
  }, [region]);
  const [title, setTitle]               = useState("");
  const [hook, setHook]                 = useState("");
  const [entities, setEntities]         = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl]     = useState<string | null>(null);
  const [isDragging, setIsDragging]     = useState(false);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState<string | null>(null);
  const [result, setResult]             = useState<AnalyzeResponse | null>(null);
  const [phase, setPhase]               = useState("");
  const phaseRef     = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const PHASES = [
    "Letterboxing thumbnail...",
    "Running PyTorch math engine...",
    "LLaVA vision analysis...",
    "Checking current topic evidence...",
    "Preparing Coach suggestions...",
  ];

  useEffect(() => { return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }; }, [previewUrl]);

  useEffect(() => {
    if (!loading) { phaseRef.current = 0; setPhase(""); return; }
    setPhase(PHASES[0]);
    const iv = setInterval(() => {
      phaseRef.current = (phaseRef.current + 1) % PHASES.length;
      setPhase(PHASES[phaseRef.current]);
    }, 4000);
    return () => clearInterval(iv);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) { setError("Upload a valid image (.jpg, .png, .webp)"); return; }
    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
  }, [previewUrl]);

  const handleDragOver  = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };
  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); };
  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation(); setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const hookWords = hook.trim() === "" ? 0 : hook.trim().split(/\s+/).length;
  const handleHookChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const words = e.target.value.trim().split(/\s+/);
    if (e.target.value === "" || words.length <= 150) setHook(e.target.value);
  };

  const canSubmit = !!(title.trim() && hook.trim() && selectedFile && !loading);

  const handleAnalyze = async () => {
    if (!canSubmit) { setError("Title, hook, and thumbnail are all required."); return; }
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append("title", title); fd.append("hook", hook); fd.append("thumbnail", selectedFile!);
    fd.append("region", region);
    fd.append("visitor_id", getAnonymousVisitorId());
    if (topicQuery.trim()) fd.append("topic_query", topicQuery.trim());
    if (entities.trim()) fd.append("user_entities", entities.trim());
    try {
      const res = await fetch("http://localhost:8000/analyze", { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Engine error: HTTP ${res.status}`);
      setResult(await res.json());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg.includes("fetch") ? "Cannot reach Neural Core — is api.py running on port 8000?" : msg);
    } finally { setLoading(false); }
  };

  const coh = result ? cohesionLabel(result.cohesion_score) : null;
  const engineLabel = !engineHealth
    ? "Checking local services"
    : engineHealth.api !== "online"
      ? "Backend offline"
      : `Core ${engineHealth.core === "ready" ? "ready" : "starting"} · LLaVA ${engineHealth.llava.replaceAll("_", " ")} · Qwen ${engineHealth.qwen.replaceAll("_", " ")}`;
  const engineDot = engineHealth?.api !== "online"
    ? "bg-burgundy"
    : engineHealth.core === "ready" && engineHealth.llava === "ready" && engineHealth.qwen === "ready"
      ? "bg-emerald"
      : "bg-gold";
  const llavaHelp = result?.thumbnail_analysis.includes("did not identify visible content")
    ? "Try a clearer image with the main subject easy to see. The vision model responded, but did not describe this image."
    : engineHealth?.llava === "ollama_offline"
    ? "The backend is trying to start Ollama. Wait a few seconds, then run the analysis again."
    : engineHealth?.llava === "model_missing"
      ? `Install the configured vision model (${engineHealth.models?.llava ?? "LLaVA"}) in Ollama.`
      : "The LLaVA request failed. Check the backend terminal for its exact error; it may be a timeout or memory limit.";

  return (
    <main className="w-full min-h-screen bg-cream text-emerald font-body flex flex-col pb-20">

      {/* Nav */}
      <nav className="w-full flex items-center justify-between px-6 py-4 border-b border-emerald/15 bg-cream sticky top-0 z-[100] shadow-sm">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-emerald/50 hover:text-burgundy transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <span className="font-heading text-xl text-emerald">
            TrendSense<span className="text-burgundy">.</span>
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${engineDot} ${engineHealth?.api === "online" ? "animate-pulse" : ""}`} />
          <span title={engineLabel} className="text-[10px] uppercase tracking-widest text-emerald/50 font-mono hidden sm:block">
            {engineLabel}
          </span>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 pt-10 pb-24 space-y-10 w-full">

        {/* Header */}
        <div>
          <h1 className="font-heading text-5xl md:text-6xl leading-tight text-emerald">
            Virality <em className="not-italic italic text-emerald/20">Engine</em>
          </h1>
          <p className="text-emerald/40 text-xs uppercase tracking-[0.2em] font-mono mt-1">
            PyTorch · SBERT · CLIP · Local Ollama caption and Coach
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-10 items-start">

          {/* ─── LEFT: Input ─── */}
          <div className="space-y-6">
            <p className="text-[10px] uppercase tracking-widest text-emerald/40 font-mono border-b border-emerald/10 pb-3">
              01 — Input Signal
            </p>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-emerald/60 mb-2">Video Title</label>
              <input type="text" value={title} onChange={e => setTitle(e.target.value)}
                placeholder="50 Hours Surviving In Solitary Confinement..." className={inputCls} />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-emerald/60">Script Hook</label>
                <span className={`text-xs font-mono font-semibold ${hookWords >= 140 ? "text-burgundy" : "text-emerald/40"}`}>
                  {hookWords}/150 words
                </span>
              </div>
              <textarea value={hook} onChange={handleHookChange}
                placeholder="The first 150 words of your script — what makes the viewer stay..."
                rows={5} className={`${inputCls} resize-none leading-relaxed`} />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-emerald/60 mb-2">
                Featured Entities <span className="text-emerald/30 normal-case font-normal">(optional)</span>
              </label>
              <div className="relative">
                <Users className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-emerald/30" />
                <input type="text" value={entities} onChange={e => setEntities(e.target.value)}
                  placeholder="Influencers, meme icons, or cultural figures in your video..."
                  className={`${inputCls} pl-11`} />
              </div>
            </div>

            <div>
              <label htmlFor="topic-query" className="block text-xs font-semibold uppercase tracking-wider text-emerald/60 mb-2">Topic to verify (optional)</label>
              <input id="topic-query" value={topicQuery} maxLength={160} onChange={e => setTopicQuery(e.target.value)} placeholder="e.g. Jio Platforms IPO" className={inputCls} />
              <p className="text-xs text-emerald/60 mt-2">Checks recent, topic-specific news coverage.</p>
              <p className="text-[11px] text-emerald/50 mt-1 mb-4">For up to 30 days, this backend keeps the short topic, region, a hashed anonymous browser ID, and summary counts from public news/YouTube sources. Your title, hook, and thumbnail are not saved for search counts.</p>
              <label htmlFor="evidence-region" className="block text-xs font-semibold uppercase tracking-wider text-emerald/60 mb-2">Evidence coverage</label>
              <select id="evidence-region" value={region} disabled={loading} onChange={e => { setRegion(e.target.value); setTrendFeed(null); }} className={inputCls}>
                <option value="MULTI">India, US, UK, Japan and Brazil</option><option value="IN">India</option><option value="US">United States</option><option value="GB">United Kingdom</option><option value="JP">Japan</option><option value="BR">Brazil</option>
              </select>
              <p className="text-xs text-emerald/60 mt-2">Sampled sources only; worldwide coverage is not exhaustive.</p>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-emerald/60 mb-2">Thumbnail</label>
              <input type="file" ref={fileInputRef} onChange={handleFileInput}
                accept="image/png,image/jpeg,image/webp" className="hidden" />
              {!previewUrl ? (
                <div onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
                  className={`w-full border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                    isDragging ? "border-burgundy bg-burgundy/5" : "border-emerald/20 bg-white hover:border-emerald/40"
                  }`}
                >
                  <ImagePlus className={`w-8 h-8 mb-3 ${isDragging ? "text-burgundy" : "text-emerald/25"}`} />
                  <span className="text-xs font-semibold text-emerald/45 uppercase tracking-widest">
                    {isDragging ? "Drop to engage" : "Drop thumbnail or click to upload"}
                  </span>
                  <span className="text-[10px] text-emerald/25 mt-1 font-mono">JPG · PNG · WEBP</span>
                </div>
              ) : (
                <div className="relative w-full rounded-xl border border-emerald/20 overflow-hidden group/thumb shadow-sm">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrl} alt="Thumbnail preview" className="w-full h-52 object-cover" />
                  <div className="absolute inset-0 bg-charcoal/50 opacity-0 group-hover/thumb:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="font-mono text-xs uppercase tracking-widest text-cream">Click × to replace</span>
                  </div>
                  <button onClick={clearFile}
                    className="absolute top-3 right-3 w-8 h-8 rounded-full bg-charcoal/80 hover:bg-burgundy flex items-center justify-center transition-colors">
                    <X className="w-4 h-4 text-cream" />
                  </button>
                </div>
              )}
            </div>

            {error && (
              <div className="flex items-start gap-3 p-4 rounded-xl bg-burgundy/8 border border-burgundy/30">
                <AlertTriangle className="w-4 h-4 text-burgundy shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-bold text-burgundy uppercase tracking-wide">Engine Error</p>
                  <p className="text-sm text-emerald/70 mt-0.5">{error}</p>
                </div>
              </div>
            )}

            <button onClick={handleAnalyze} disabled={!canSubmit}
              className="w-full flex items-center justify-center gap-3 py-4 rounded-xl font-bold uppercase tracking-wider text-sm transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed bg-burgundy hover:bg-charcoal active:scale-[0.98] text-cream shadow-lg shadow-burgundy/20">
              {loading
                ? <><Loader2 className="w-4 h-4 animate-spin" />{phase || "Processing..."}</>
                : <><Send className="w-4 h-4" />Run Engine</>
              }
            </button>
          </div>

          {/* ─── RIGHT: Output ─── */}
          <div className="space-y-5">
            <p className="text-[10px] uppercase tracking-widest text-emerald/40 font-mono border-b border-emerald/10 pb-3">
              02 — Telemetry Output
            </p>

            {/* Idle */}
            {!loading && !result && (
              <div className="flex flex-col items-center justify-center min-h-[520px] rounded-2xl border-2 border-dashed border-emerald/15 bg-white/50 text-emerald/25 gap-4">
                <Activity className="w-10 h-10" />
                <p className="font-mono text-xs uppercase tracking-widest">Standing by for raw data.</p>
              </div>
            )}

            {/* Loading — concentric rings, no blob orb */}
            {loading && !result && (
              <div className="flex flex-col items-center justify-center min-h-[520px] rounded-2xl border border-emerald/15 bg-white/50 gap-6">
                <div className="relative flex items-center justify-center w-32 h-32">
                  <div className="absolute inset-0 rounded-full border border-emerald/10 animate-ping" style={{ animationDuration: "2s" }} />
                  <div className="absolute inset-3 rounded-full border border-burgundy/20 animate-ping" style={{ animationDuration: "2.4s", animationDelay: "0.4s" }} />
                  <div className="absolute inset-6 rounded-full border border-gold/20 animate-ping" style={{ animationDuration: "2.8s", animationDelay: "0.8s" }} />
                  <div className="w-10 h-10 rounded-full bg-burgundy/10 border border-burgundy/25 flex items-center justify-center">
                    <BrainCircuit className="w-5 h-5 text-burgundy" />
                  </div>
                </div>
                <div className="text-center space-y-2">
                  <p className="font-mono text-xs uppercase tracking-widest text-emerald/55 animate-pulse">{phase}</p>
                  <p className="text-[10px] text-emerald/30 font-mono">This may take 30–90 seconds on GPU</p>
                </div>
              </div>
            )}

            {/* Result */}
            {result && !loading && (
              <div className="space-y-4">

                {/* Score card — dark with subtle gradient bg */}
                <div className="relative rounded-2xl bg-charcoal border border-cream/8 overflow-hidden p-8">
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute -right-16 -top-16 w-56 h-56 rounded-full opacity-25"
                      style={{ background: "radial-gradient(circle, #A6824A 0%, transparent 70%)" }} />
                    <div className="absolute -left-8 -bottom-8 w-44 h-44 rounded-full opacity-15"
                      style={{ background: "radial-gradient(circle, #5D1E21 0%, transparent 70%)" }} />
                  </div>
                  <div className="relative z-10">
                    <p className="text-[10px] font-mono uppercase tracking-widest text-cream/30 mb-4 flex items-center gap-2">
                      <BarChart3 className="w-3.5 h-3.5" /> Historical Estimate + Live Evidence
                    </p>
                    <div className="flex items-baseline gap-3 mb-4">
                      <span className="font-heading text-8xl leading-none text-cream tabular-nums">
                        {result.final_virality_score.toFixed(1)}
                      </span>
                      <span className="text-gold font-bold text-2xl">/100</span>
                    </div>
                    <ProgressBar value={result.final_virality_score} color="bg-gold" delay={100} />
                    <div className="mt-5 flex flex-wrap gap-3">
                      <span className="text-xs text-cream/60">
                        Historical model {result.historical_model_score?.toFixed(1) ?? result.base_structural_score.toFixed(1)}
                        {result.evidence_adjustment && ` + ${result.evidence_adjustment.points.toFixed(1)} live-evidence points`}
                        {result.evidence_adjustment && ` (up to ${result.evidence_adjustment.maximum_points})`}
                      </span>
                      {result.evidence_adjustment && result.evidence_adjustment.points > 0 && (
                        <span className="text-[10px] text-cream/45">
                          YouTube +{result.evidence_adjustment.components.youtube_chart} · Google Trends +{result.evidence_adjustment.components.google_trends} · News coverage +{result.evidence_adjustment.components.news_coverage}. Experimental evidence lift; not included in the model test MAE.
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Breakdown */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-white border border-emerald/15 p-5 shadow-sm">
                    <p className="text-[10px] font-mono uppercase tracking-widest text-emerald/40 mb-2">Held-out Test MAE</p>
                    <p className="text-3xl font-heading text-emerald tabular-nums mb-2">
                      {result.model_evaluation?.test_mae.toFixed(1) ?? "—"}
                      <span className="text-xs font-sans text-emerald/45 ml-2">points</span>
                    </p>
                    <p className="text-[11px] text-emerald/50">Average test-set error across {result.model_evaluation?.test_rows ?? "—"} examples; lower is better.</p>
                  </div>
                  <div className="rounded-xl bg-white border border-emerald/15 p-5 shadow-sm">
                    <p className="text-[10px] font-mono uppercase tracking-widest text-emerald/40 mb-2">CLIP Similarity</p>
                    <p className="text-3xl font-heading text-emerald tabular-nums mb-2">
                      {result.cohesion_score.toFixed(3)}
                    </p>
                    {coh && (
                      <p className={`text-[10px] font-mono mt-2 flex items-center gap-1 ${coh.color}`}>
                        {coh.icon} {coh.label}
                      </p>
                    )}
                    <p className="text-[11px] text-emerald/50 mt-2">How closely the thumbnail matches the title visually; not a CTR or accuracy percentage.</p>
                  </div>
                </div>

                {result.live_trends?.momentum && (
                  <div className="rounded-xl bg-white border border-emerald/25 p-5 shadow-sm">
                    <p className="text-xs uppercase tracking-widest text-emerald/50 mb-2">Google Trends + YouTube chart check</p>
                    <p className="font-semibold text-emerald">{result.live_trends.momentum.label}</p>
                    <p className="text-sm text-emerald/70 mt-2">{result.live_trends.momentum.interpretation}</p>
                    <p className="text-xs text-emerald/55 mt-2">This check uses sampled Google Trends and YouTube popular-chart items. Google News coverage and community searches appear separately.</p>
                    {result.topic_evidence?.status === "recent_coverage_found" && <p className="text-xs text-emerald/65 mt-2">Separate news evidence: {result.topic_evidence.article_count} related articles from {result.topic_evidence.publisher_count} publishers in the past {result.topic_evidence.window_hours} hours.</p>}
                    {result.search_interest?.status === "counted" && <p className="text-xs text-emerald/65 mt-2">Community evidence: {result.search_interest.unique_browsers_7d} distinct browser{result.search_interest.unique_browsers_7d === 1 ? "" : "s"} searched this topic in the past {result.search_interest.window_days} days.</p>}
                    {result.live_trends.momentum.missing_sources?.map(source => <p key={source} className="text-xs text-emerald/60 mt-2">Source not connected: {source}</p>)}
                    {result.live_trends.history_status === "unavailable" && <p className="text-sm mt-2">Observation history could not be saved; growth measurements are unavailable.</p>}
                    <ul className="mt-4 space-y-3">
                      {result.live_trends.matches?.map(item => (
                        <li key={`${item.source}-${item.region}-${item.source_url}-${item.topic}`} className="text-sm text-emerald/80">
                          <a href={item.source_url} target="_blank" rel="noreferrer" className="underline">{item.topic}</a>
                          <p className="text-xs mt-1">{item.source} · {item.region}</p>
                          {item.growth?.status === "measured" ? (
                            <p className="text-xs mt-1">+{item.growth.views_added?.toLocaleString()} views over {item.growth.window_hours} hours ({item.growth.views_per_hour?.toLocaleString()} views/hour). Observed video growth; not a worldwide forecast.</p>
                          ) : item.growth?.status === "collecting_history" ? (
                            <p className="text-xs mt-1">Collecting history. Check again after at least 30 minutes to measure view growth.</p>
                          ) : item.growth && <p className="text-xs mt-1">View growth unavailable: the count is missing or was revised.</p>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.topic_evidence && (
                  <div className="rounded-xl bg-white border border-emerald/25 p-5 shadow-sm text-sm text-emerald">
                    <p className="font-heading text-xl mb-2">Recent coverage: {result.topic_evidence.query}</p>
                    <p className="font-semibold">{result.topic_evidence.status === "recent_coverage_found" ? "Related coverage found" : result.topic_evidence.status === "unavailable" ? "Topic lookup unavailable" : "No related coverage found in this search"}</p>
                    <p className="mt-2">{result.topic_evidence.article_count} articles from {result.topic_evidence.publisher_count} publisher domains · past {result.topic_evidence.window_hours} hours</p>
                    {result.topic_evidence.candidate_count !== undefined && <p className="text-xs mt-2">{result.topic_evidence.candidate_count} recent search results before relevance filtering. {result.topic_evidence.sources?.map(source => `${source.region}: ${source.status}`).join(" · ")}</p>}
                    {result.topic_evidence.article_count === 0 && <p className="text-sm mt-3">Try a short topic such as “Jio IPO”. Detailed claims, figures and long sentences can make the search too restrictive. No result does not verify or disprove the claims in your script.</p>}
                    <ul className="space-y-3 mt-3">{result.topic_evidence.articles.slice(0, 5).map(article => <li key={article.url}><a href={article.url} target="_blank" rel="noreferrer" className="underline">{article.title}</a><p className="text-xs opacity-70 mt-1">{article.publisher} · {new Date(article.published_at).toLocaleString()}</p></li>)}</ul>
                    <p className="text-xs mt-3">Recent coverage indicates discussion, not measured audience demand.</p>
                  </div>
                )}

                {result.search_interest && (
                  <div className="rounded-xl bg-white border border-emerald/25 p-5 shadow-sm text-sm text-emerald">
                    <p className="text-[10px] font-mono uppercase tracking-widest text-emerald/55 mb-2">Community search interest</p>
                    {result.search_interest.status === "counted" ? (
                      <>
                        <p className="font-heading text-xl">{result.search_interest.unique_browsers_7d} distinct browser{result.search_interest.unique_browsers_7d === 1 ? "" : "s"}</p>
                        <p className="mt-1">searched “{result.search_interest.topic}” in {result.search_interest.region} during the past {result.search_interest.window_days} days.</p>
                        <p className="text-xs mt-2">Counted on {result.search_interest.scope}. Repeat searches from the same browser count once per day. This signal is not yet part of the model score.</p>
                      </>
                    ) : (
                      <p>{result.search_interest.note}</p>
                    )}
                  </div>
                )}

                {/* LLaVA Vision */}
                <div className="rounded-xl bg-white border border-emerald/15 p-5 shadow-sm">
                  <p className="text-[10px] font-mono uppercase tracking-widest text-emerald/40 mb-3 flex items-center gap-2">
                    <Eye className="w-3.5 h-3.5" /> LLaVA Short Caption · {result.thumbnail_analysis_status === "unavailable" ? "Unavailable" : "Verify Details"}
                  </p>
                  <p className={`text-sm leading-relaxed ${result.thumbnail_analysis_status === "unavailable" ? "text-burgundy/80" : "text-emerald/65 italic"}`}>
                    &ldquo;{result.thumbnail_analysis.trim()}&rdquo;
                  </p>
                  {result.thumbnail_analysis_status === "unavailable" && <p className="text-xs text-burgundy/70 mt-2">{llavaHelp}</p>}
                </div>

                {/* ── AI Coach — evidence report with Qwen or contextual fallback ── */}
                <div className="rounded-xl bg-charcoal border border-cream/8 overflow-hidden">
                  {/* Header */}
                  <div className="flex items-center gap-3 px-6 py-5 border-b border-cream/8">
                    <div className="p-2 rounded-lg bg-burgundy/25 shrink-0">
                      <BrainCircuit className="w-4 h-4 text-burgundy" />
                    </div>
                    <span className="font-heading text-2xl text-cream">AI Coach</span>
                    <Sparkles className="w-3.5 h-3.5 text-gold" />
                  </div>

                  {/* Coach report is generated from evidence and bounded advice codes. */}
                  <div className="px-6 py-5">
                    <MarkdownBlock text={result.ai_strategy_report} />
                  </div>

                  <div className="px-6 pb-5 flex items-center gap-2 text-cream/20">
                    <Cpu className="w-3 h-3" />
                    <span className="font-mono text-[10px] uppercase tracking-widest">
                      trendsense_core_v5.pt · CUDA · {result.coach_mode === "qwen_assisted" ? "Qwen-assisted Coach" : "Rule-based Coach fallback"} · {result.thumbnail_analysis_status === "unavailable" ? "LLaVA unavailable" : "LLaVA"}
                    </span>
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>

        {/* Live Pulse */}
        <section className="relative rounded-2xl border border-emerald/15 bg-white p-6 overflow-hidden shadow-sm">
          <div className="relative z-10 flex flex-col md:flex-row md:items-center gap-6">
            <div className="flex items-center gap-4 shrink-0 md:border-r md:border-emerald/10 md:pr-6">
              <div className="relative flex items-center justify-center">
                <span className="absolute w-5 h-5 bg-burgundy/25 rounded-full animate-ping" />
                <span className="w-2.5 h-2.5 bg-burgundy rounded-full relative z-10" />
              </div>
              <div>
                <h3 className="font-heading text-xl text-emerald flex items-center gap-2">
                  Live Pulse <Radio className="w-4 h-4 text-burgundy" />
                </h3>
                <p className="text-[10px] font-mono uppercase tracking-widest text-emerald/35 mt-0.5">
                  Global News and Video Topics from Live Sources
                </p>
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap gap-x-4 gap-y-2">
                {trendFeed?.sources?.map(source => <p key={source.name} className="text-xs text-emerald/70">{source.name}: {source.status.replaceAll("_", " ")} · {source.items} items</p>)}
              </div>
              {trendFeed?.live_pulse && (
                <div className="mt-4 border-t border-emerald/10 pt-4">
                  <p className="text-[10px] font-mono uppercase tracking-widest text-emerald/45">
                    Topic clusters · {trendFeed.live_pulse.items_analyzed} distinct feed items · BERTopic / HDBSCAN
                  </p>
                  <p className="text-xs text-emerald/55 mt-1">{trendFeed.live_pulse.note}</p>
                  {trendFeed.live_pulse.ranking && <p className="text-[11px] text-emerald/50 mt-1">Ordering: {trendFeed.live_pulse.ranking}.</p>}
                  {trendFeed.live_pulse.clusters.length > 0 ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">
                      {trendFeed.live_pulse.clusters.map(cluster => (
                        <div key={cluster.topic_id} className="rounded-lg border border-emerald/10 bg-cream/50 p-3">
                          <p className="font-semibold text-sm text-emerald">{cluster.name}</p>
                          <p className="text-[11px] text-emerald/55 mt-1">
                            {cluster.item_count} related items · {cluster.publisher_count ?? 0} news publishers · {cluster.country_count ?? 0} countries · {cluster.source_count} source types
                            {cluster.observed_views_per_hour !== null && ` · +${cluster.observed_views_per_hour.toLocaleString()} observed YouTube views/hour`}
                          </p>
                          <ul className="mt-2 space-y-1">
                            {cluster.sample_items.slice(0, 2).map(item => (
                              <li key={`${cluster.topic_id}-${item.source}-${item.source_url}-${item.title}`} className="text-xs text-emerald/75 truncate">
                                <a href={item.source_url} target="_blank" rel="noreferrer" className="underline">{item.title}</a>
                                <span className="text-emerald/45"> · {item.source}{item.publisher ? ` · ${item.publisher}` : ""}{item.country ? ` · ${item.country}` : item.region ? ` · ${item.region}` : ""}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-emerald/60 mt-3">No stable multi-item topics were found in this feed snapshot.</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}

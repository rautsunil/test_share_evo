import { useState, useCallback } from "react";

// ─── Token estimation (approx 4 chars per token) ───────────────────────────
const estimateTokens = (text) => Math.ceil((text || "").length / 4);

// ─── Layer definitions ──────────────────────────────────────────────────────
const LAYER_DEFS = [
  {
    id: "role",
    label: "Role & Persona",
    icon: "⬡",
    color: "#7C3AED",
    accent: "#A78BFA",
    placeholder: "Define who the AI is. E.g. You are a CRM analyst for a B2C e-commerce company specializing in customer segmentation and campaign strategy.",
    default: "You are a senior CRM data scientist at a B2C e-commerce company. You specialize in customer segmentation, behavioral analysis, and personalized campaign strategy.",
  },
  {
    id: "instructions",
    label: "Instructions & Rules",
    icon: "⬡",
    color: "#0F766E",
    accent: "#2DD4BF",
    placeholder: "Precise rules, output format, constraints. E.g. Always respond in JSON. Be specific not generic. Prioritize recency signals.",
    default: `Always respond in structured JSON with keys: segment_label, insight, recommended_action, channel, urgency, confidence_score.
Rules:
- Be specific, never generic
- Base urgency on recency of last purchase
- Prefer push notifications for mobile-active users
- Confidence score must be 0.0 to 1.0`,
  },
  {
    id: "memory",
    label: "Memory & History",
    icon: "⬡",
    color: "#B45309",
    accent: "#FCD34D",
    placeholder: "Past conversation context, prior decisions, historical patterns. E.g. Last campaign had 34% open rate.",
    default: `Past campaign performance:
- Email campaign 2 weeks ago → 12% conversion, 34% open rate (above average)
- Push notification last month → 8% CTR
- Segment historically responds well to urgency-based messaging`,
  },
  {
    id: "tools",
    label: "Tools & Capabilities",
    icon: "⬡",
    color: "#0369A1",
    accent: "#38BDF8",
    placeholder: "Available tools, APIs, functions the model can call. E.g. search_customer_db(), get_segment_stats().",
    default: `Available tools:
- get_segment_stats(segment_id): Returns live engagement metrics
- fetch_product_catalog(): Returns current inventory and pricing
- schedule_campaign(channel, message, segment_id): Queues a campaign`,
  },
  {
    id: "state",
    label: "State & Environment",
    icon: "⬡",
    color: "#BE185D",
    accent: "#F472B6",
    placeholder: "Current date, system state, runtime variables. E.g. Today: 2026-03-23. Peak season active.",
    default: `Current context:
- Date: 2026-03-23 (Monday)
- Active promotion: End of Quarter Sale (ends March 31)
- Inventory alert: Electronics category 40% stock remaining
- Platform: Mobile app (85% of user traffic)`,
  },
  {
    id: "examples",
    label: "Examples (Few-Shot)",
    icon: "⬡",
    color: "#047857",
    accent: "#34D399",
    placeholder: "Concrete input→output examples to guide model behavior.",
    default: `Example 1:
Input: Age 25-30, last purchase 10 days ago, high AOV
Output: {"segment_label":"High-Value Recent","insight":"Active buyer with premium spending","recommended_action":"Cross-sell premium accessories","channel":"push","urgency":"medium","confidence_score":0.87}

Example 2:
Input: Age 35-45, last purchase 90 days ago, low engagement
Output: {"segment_label":"At-Risk Dormant","insight":"Lapsing customer needs reactivation","recommended_action":"Win-back offer with 20% discount","channel":"email","urgency":"high","confidence_score":0.91}`,
  },
  {
    id: "query",
    label: "User Query / Input",
    icon: "⬡",
    color: "#6D28D9",
    accent: "#C4B5FD",
    placeholder: "The actual question or data being processed in this context.",
    default: "Segment data: Age 28-35, City: Bengaluru, Last purchase: 45 days ago, Mobile-active: Yes, Avg order value: ₹2200, Category affinity: Electronics",
  },
];

// ─── Token Bar ───────────────────────────────────────────────────────────────
function TokenBar({ tokens, maxTokens = 4000, color }) {
  const pct = Math.min((tokens / maxTokens) * 100, 100);
  const warn = pct > 75;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 4, background: "#1e1e2e", borderRadius: 4, overflow: "hidden"
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: warn ? "#f59e0b" : color,
          borderRadius: 4,
          transition: "width 0.3s ease"
        }} />
      </div>
      <span style={{
        fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
        color: warn ? "#f59e0b" : "#6b7280", minWidth: 60, textAlign: "right"
      }}>
        ~{tokens.toLocaleString()} tok
      </span>
    </div>
  );
}

// ─── Layer Card ──────────────────────────────────────────────────────────────
function LayerCard({ layer, value, enabled, onChange, onToggle, index }) {
  const tokens = estimateTokens(value);
  const [expanded, setExpanded] = useState(true);

  return (
    <div style={{
      background: enabled ? "#0f0f1a" : "#080810",
      border: `1px solid ${enabled ? layer.color + "55" : "#1e1e2e"}`,
      borderRadius: 12,
      overflow: "hidden",
      transition: "all 0.3s ease",
      opacity: enabled ? 1 : 0.45,
      fontFamily: "'Inter', sans-serif",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "10px 14px",
        background: enabled ? layer.color + "18" : "transparent",
        borderBottom: `1px solid ${enabled ? layer.color + "33" : "#1e1e2e"}`,
        cursor: "pointer",
      }}>
        {/* Layer index badge */}
        <div style={{
          width: 22, height: 22, borderRadius: 6,
          background: enabled ? layer.color : "#1e1e2e",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 11, fontWeight: 700, color: "#fff",
          flexShrink: 0,
          transition: "background 0.3s"
        }}>
          {index + 1}
        </div>

        {/* Label */}
        <div style={{ flex: 1 }} onClick={() => setExpanded(e => !e)}>
          <div style={{
            fontSize: 12, fontWeight: 600,
            color: enabled ? layer.accent : "#4b5563",
            letterSpacing: "0.04em", textTransform: "uppercase"
          }}>
            {layer.label}
          </div>
          {!expanded && (
            <div style={{ fontSize: 11, color: "#4b5563", marginTop: 1 }}>
              {value.slice(0, 60)}{value.length > 60 ? "…" : ""}
            </div>
          )}
        </div>

        {/* Token count pill */}
        <div style={{
          fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
          color: enabled ? layer.accent + "cc" : "#374151",
          background: enabled ? layer.color + "22" : "#111",
          padding: "2px 8px", borderRadius: 20,
          border: `1px solid ${enabled ? layer.color + "44" : "#1e1e2e"}`,
        }}>
          ~{tokens} tok
        </div>

        {/* Expand toggle */}
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "#4b5563", fontSize: 14, padding: "0 4px",
            transition: "transform 0.2s"
          }}
        >
          {expanded ? "▲" : "▼"}
        </button>

        {/* Enable/disable toggle */}
        <div
          onClick={onToggle}
          style={{
            width: 36, height: 20, borderRadius: 10,
            background: enabled ? layer.color : "#1e1e2e",
            border: `1px solid ${enabled ? layer.color : "#374151"}`,
            position: "relative", cursor: "pointer",
            transition: "all 0.25s ease", flexShrink: 0,
          }}
        >
          <div style={{
            width: 14, height: 14, borderRadius: "50%",
            background: "#fff",
            position: "absolute", top: 2,
            left: enabled ? 18 : 2,
            transition: "left 0.25s ease",
            boxShadow: "0 1px 3px rgba(0,0,0,0.4)"
          }} />
        </div>
      </div>

      {/* Body */}
      {expanded && (
        <div style={{ padding: "10px 14px 12px" }}>
          <textarea
            value={value}
            onChange={e => onChange(e.target.value)}
            disabled={!enabled}
            placeholder={layer.placeholder}
            style={{
              width: "100%", minHeight: 90,
              background: "#07070f",
              border: `1px solid ${enabled ? layer.color + "33" : "#1a1a2e"}`,
              borderRadius: 8, padding: "10px 12px",
              color: enabled ? "#e2e8f0" : "#374151",
              fontSize: 12.5,
              fontFamily: "'JetBrains Mono', monospace",
              lineHeight: 1.6, resize: "vertical",
              outline: "none", boxSizing: "border-box",
              transition: "all 0.2s",
            }}
            onFocus={e => { if (enabled) e.target.style.borderColor = layer.color + "88"; }}
            onBlur={e => { e.target.style.borderColor = enabled ? layer.color + "33" : "#1a1a2e"; }}
          />
          <div style={{ marginTop: 6 }}>
            <TokenBar tokens={tokens} color={layer.color} />
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Token Summary Bar ───────────────────────────────────────────────────────
function TokenSummary({ layers, values, enabled }) {
  const MAX = 8000;
  const total = layers.reduce((sum, l) => sum + (enabled[l.id] ? estimateTokens(values[l.id]) : 0), 0);
  const pct = Math.min((total / MAX) * 100, 100);

  return (
    <div style={{
      background: "#0a0a15",
      border: "1px solid #1e1e35",
      borderRadius: 12, padding: "14px 18px",
      fontFamily: "'Inter', sans-serif",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>
          Total Context Budget
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontSize: 13, fontFamily: "'JetBrains Mono', monospace",
            color: pct > 80 ? "#f59e0b" : "#a78bfa", fontWeight: 700
          }}>
            ~{total.toLocaleString()}
          </span>
          <span style={{ fontSize: 11, color: "#374151" }}>/ {MAX.toLocaleString()} tokens</span>
        </div>
      </div>

      {/* Stacked bar */}
      <div style={{ display: "flex", height: 8, borderRadius: 6, overflow: "hidden", background: "#111", gap: 1 }}>
        {layers.map(l => {
          const layerPct = enabled[l.id] ? (estimateTokens(values[l.id]) / MAX) * 100 : 0;
          return layerPct > 0 ? (
            <div key={l.id} title={`${l.label}: ~${estimateTokens(values[l.id])} tokens`}
              style={{ width: `${layerPct}%`, background: l.color, transition: "width 0.3s ease" }} />
          ) : null;
        })}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", marginTop: 10 }}>
        {layers.map(l => {
          const t = enabled[l.id] ? estimateTokens(values[l.id]) : 0;
          return (
            <div key={l.id} style={{ display: "flex", alignItems: "center", gap: 5, opacity: enabled[l.id] ? 1 : 0.3 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: l.color, flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "#6b7280" }}>{l.label}</span>
              <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: l.accent }}>
                {t > 0 ? `~${t}` : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Output Panel ─────────────────────────────────────────────────────────────
function OutputPanel({ output, loading, error }) {
  const formatOutput = (text) => {
    if (!text) return null;
    const clean = text.replace(/```json|```/g, "").trim();
    try {
      const parsed = JSON.parse(clean);
      return (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {Object.entries(parsed).map(([k, v]) => (
            <div key={k} style={{
              display: "flex", gap: 12, alignItems: "flex-start",
              padding: "8px 12px", background: "#07070f",
              border: "1px solid #1e1e2e", borderRadius: 8
            }}>
              <span style={{
                fontSize: 10, color: "#7C3AED", fontFamily: "'JetBrains Mono', monospace",
                textTransform: "uppercase", letterSpacing: "0.08em",
                minWidth: 130, paddingTop: 2, fontWeight: 600
              }}>{k}</span>
              <span style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.5 }}>
                {typeof v === "number" ? v.toFixed(2) : String(v)}
              </span>
            </div>
          ))}
        </div>
      );
    } catch {
      return (
        <p style={{
          fontSize: 13, color: "#cbd5e1", lineHeight: 1.7,
          fontFamily: "'JetBrains Mono', monospace", whiteSpace: "pre-wrap"
        }}>{text}</p>
      );
    }
  };

  return (
    <div style={{
      background: "#0a0a15", border: "1px solid #1e1e35",
      borderRadius: 12, overflow: "hidden", fontFamily: "'Inter', sans-serif",
    }}>
      <div style={{
        padding: "10px 16px", background: "#0d0d1f",
        borderBottom: "1px solid #1e1e35",
        display: "flex", alignItems: "center", gap: 8
      }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: loading ? "#f59e0b" : output ? "#34d399" : "#374151",
          boxShadow: loading ? "0 0 8px #f59e0b" : output ? "0 0 8px #34d399" : "none",
          animation: loading ? "pulse 1s infinite" : "none"
        }} />
        <span style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>
          LLM Output
        </span>
        {output && (
          <span style={{
            marginLeft: "auto", fontSize: 10,
            color: "#374151", fontFamily: "'JetBrains Mono', monospace"
          }}>
            ~{estimateTokens(output)} tokens
          </span>
        )}
      </div>

      <div style={{ padding: 16, minHeight: 120 }}>
        {loading && (
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ display: "flex", gap: 4 }}>
              {[0, 1, 2].map(i => (
                <div key={i} style={{
                  width: 6, height: 6, borderRadius: "50%", background: "#7C3AED",
                  animation: `bounce 1s infinite ${i * 0.15}s`
                }} />
              ))}
            </div>
            <span style={{ fontSize: 13, color: "#6b7280" }}>Calling LLM with engineered context…</span>
          </div>
        )}
        {error && (
          <div style={{
            padding: "10px 14px", background: "#1a0a0a",
            border: "1px solid #7f1d1d", borderRadius: 8,
            fontSize: 13, color: "#fca5a5"
          }}>{error}</div>
        )}
        {!loading && !error && output && formatOutput(output)}
        {!loading && !error && !output && (
          <p style={{ fontSize: 13, color: "#374151", fontStyle: "italic" }}>
            Configure your context layers and click Run →
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function ContextWorkbench() {
  const [values, setValues] = useState(
    Object.fromEntries(LAYER_DEFS.map(l => [l.id, l.default]))
  );
  const [enabled, setEnabled] = useState(
    Object.fromEntries(LAYER_DEFS.map(l => [l.id, true]))
  );
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const toggleLayer = useCallback((id) => {
    setEnabled(e => ({ ...e, [id]: !e[id] }));
  }, []);

  const enableAll = () => setEnabled(Object.fromEntries(LAYER_DEFS.map(l => [l.id, true])));
  const disableAll = () => setEnabled(Object.fromEntries(LAYER_DEFS.map(l => [l.id, false])));

  const buildContext = () => {
    const systemParts = [];
    const queryLayer = values["query"];

    if (enabled["role"]) systemParts.push(`## Role\n${values["role"]}`);
    if (enabled["instructions"]) systemParts.push(`## Instructions\n${values["instructions"]}`);
    if (enabled["tools"]) systemParts.push(`## Available Tools\n${values["tools"]}`);
    if (enabled["examples"]) systemParts.push(`## Examples\n${values["examples"]}`);

    const system = systemParts.join("\n\n");

    const userParts = [];
    if (enabled["memory"]) userParts.push(`### Memory & History\n${values["memory"]}`);
    if (enabled["state"]) userParts.push(`### Current State\n${values["state"]}`);
    if (enabled["query"]) userParts.push(`### Query\n${queryLayer}`);

    const userMessage = userParts.join("\n\n");
    return { system, userMessage };
  };

  const handleRun = async () => {
    const activeCount = Object.values(enabled).filter(Boolean).length;
    if (activeCount === 0) {
      setError("Enable at least one context layer before running.");
      return;
    }

    setLoading(true);
    setOutput("");
    setError("");

    const { system, userMessage } = buildContext();

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: system || "You are a helpful assistant.",
          messages: [{ role: "user", content: userMessage || "Hello" }],
        }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error.message);
      setOutput(data.content?.[0]?.text || "No response");
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const activeCount = Object.values(enabled).filter(Boolean).length;
  const totalTokens = LAYER_DEFS.reduce((sum, l) =>
    sum + (enabled[l.id] ? estimateTokens(values[l.id]) : 0), 0);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060610; }
        textarea { transition: border-color 0.2s, background 0.2s; }
        textarea:focus { background: #0a0a18 !important; }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-6px); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0a15; }
        ::-webkit-scrollbar-thumb { background: #1e1e35; border-radius: 3px; }
      `}</style>

      <div style={{
        minHeight: "100vh", background: "#060610",
        color: "#e2e8f0", fontFamily: "'Syne', sans-serif",
        padding: "24px 20px",
      }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>

          {/* Header */}
          <div style={{ marginBottom: 24, animation: "fadeIn 0.5s ease" }}>
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
              <div>
                <div style={{
                  fontSize: 11, color: "#6b7280", textTransform: "uppercase",
                  letterSpacing: "0.15em", fontWeight: 600, marginBottom: 4,
                  fontFamily: "'JetBrains Mono', monospace"
                }}>
                  ◈ Context Engineering
                </div>
                <h1 style={{
                  fontSize: 26, fontWeight: 800, color: "#f1f5f9",
                  letterSpacing: "-0.02em", lineHeight: 1
                }}>
                  Workbench
                </h1>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {/* Enable/disable all */}
                <button onClick={enableAll} style={{
                  fontSize: 11, padding: "6px 12px", borderRadius: 8,
                  background: "transparent", border: "1px solid #1e1e35",
                  color: "#6b7280", cursor: "pointer", fontFamily: "'JetBrains Mono', monospace",
                  transition: "all 0.2s",
                }}
                  onMouseEnter={e => { e.target.style.borderColor = "#34d399"; e.target.style.color = "#34d399"; }}
                  onMouseLeave={e => { e.target.style.borderColor = "#1e1e35"; e.target.style.color = "#6b7280"; }}
                >
                  Enable All
                </button>
                <button onClick={disableAll} style={{
                  fontSize: 11, padding: "6px 12px", borderRadius: 8,
                  background: "transparent", border: "1px solid #1e1e35",
                  color: "#6b7280", cursor: "pointer", fontFamily: "'JetBrains Mono', monospace",
                  transition: "all 0.2s",
                }}
                  onMouseEnter={e => { e.target.style.borderColor = "#ef4444"; e.target.style.color = "#ef4444"; }}
                  onMouseLeave={e => { e.target.style.borderColor = "#1e1e35"; e.target.style.color = "#6b7280"; }}
                >
                  Disable All
                </button>

                {/* Run button */}
                <button
                  onClick={handleRun}
                  disabled={loading}
                  style={{
                    padding: "8px 22px", borderRadius: 10,
                    background: loading ? "#1e1e35" : "linear-gradient(135deg, #7C3AED, #4F46E5)",
                    border: "none", color: "#fff",
                    fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
                    fontFamily: "'Syne', sans-serif", letterSpacing: "0.03em",
                    boxShadow: loading ? "none" : "0 0 20px #7C3AED44",
                    transition: "all 0.25s",
                    display: "flex", alignItems: "center", gap: 8
                  }}
                >
                  {loading ? (
                    <><span style={{ animation: "pulse 1s infinite" }}>●</span> Running…</>
                  ) : (
                    <>▶ Run  <span style={{
                      fontSize: 10, background: "rgba(255,255,255,0.15)",
                      padding: "2px 6px", borderRadius: 5,
                      fontFamily: "'JetBrains Mono', monospace"
                    }}>{activeCount} layers · ~{totalTokens} tok</span></>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Main layout */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 16, alignItems: "start" }}>

            {/* Left: Layers */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {/* Token summary */}
              <TokenSummary layers={LAYER_DEFS} values={values} enabled={enabled} />

              {/* Layer cards */}
              {LAYER_DEFS.map((layer, i) => (
                <div key={layer.id} style={{ animation: `fadeIn 0.4s ease ${i * 0.05}s both` }}>
                  <LayerCard
                    layer={layer}
                    value={values[layer.id]}
                    enabled={enabled[layer.id]}
                    onChange={v => setValues(prev => ({ ...prev, [layer.id]: v }))}
                    onToggle={() => toggleLayer(layer.id)}
                    index={i}
                  />
                </div>
              ))}
            </div>

            {/* Right: Output + Stats */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12, position: "sticky", top: 20 }}>

              {/* Active layers stat */}
              <div style={{
                background: "#0a0a15", border: "1px solid #1e1e35",
                borderRadius: 12, padding: "12px 16px",
                display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8,
              }}>
                {[
                  { label: "Active Layers", value: activeCount, color: "#a78bfa" },
                  { label: "Total Tokens", value: `~${totalTokens}`, color: "#2dd4bf" },
                  { label: "Disabled", value: LAYER_DEFS.length - activeCount, color: "#f87171" },
                ].map(s => (
                  <div key={s.label} style={{ textAlign: "center" }}>
                    <div style={{
                      fontSize: 20, fontWeight: 800, color: s.color,
                      fontFamily: "'JetBrains Mono', monospace", lineHeight: 1
                    }}>{s.value}</div>
                    <div style={{ fontSize: 9, color: "#4b5563", marginTop: 3, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                      {s.label}
                    </div>
                  </div>
                ))}
              </div>

              {/* Per-layer token breakdown */}
              <div style={{
                background: "#0a0a15", border: "1px solid #1e1e35",
                borderRadius: 12, padding: "12px 16px",
              }}>
                <div style={{ fontSize: 10, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10, fontWeight: 600 }}>
                  Token Breakdown
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                  {LAYER_DEFS.map(l => {
                    const t = estimateTokens(values[l.id]);
                    const active = enabled[l.id];
                    return (
                      <div key={l.id} style={{ opacity: active ? 1 : 0.3, transition: "opacity 0.2s" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span style={{ fontSize: 10, color: l.accent, fontWeight: 600 }}>{l.label}</span>
                          <span style={{
                            fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                            color: active ? l.accent : "#374151"
                          }}>
                            {active ? `~${t}` : "off"}
                          </span>
                        </div>
                        <TokenBar tokens={active ? t : 0} color={l.color} />
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Output */}
              <OutputPanel output={output} loading={loading} error={error} />

              {/* Context preview toggle */}
              {!loading && (
                <details style={{
                  background: "#0a0a15", border: "1px solid #1e1e35",
                  borderRadius: 12, overflow: "hidden",
                }}>
                  <summary style={{
                    padding: "10px 16px", cursor: "pointer",
                    fontSize: 10, color: "#4b5563", textTransform: "uppercase",
                    letterSpacing: "0.1em", fontWeight: 600, userSelect: "none",
                  }}>
                    ◎ Raw Context Preview
                  </summary>
                  <div style={{ padding: "0 16px 14px" }}>
                    <pre style={{
                      fontSize: 10, color: "#374151",
                      fontFamily: "'JetBrains Mono', monospace",
                      whiteSpace: "pre-wrap", lineHeight: 1.5,
                      maxHeight: 220, overflowY: "auto"
                    }}>
                      {(() => {
                        const { system, userMessage } = buildContext();
                        return `[SYSTEM]\n${system || "(empty)"}\n\n[USER]\n${userMessage || "(empty)"}`;
                      })()}
                    </pre>
                  </div>
                </details>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

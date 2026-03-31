"""
╔══════════════════════════════════════════════════════════════════════╗
║              CONTEXT ENGINEERING FRAMEWORK  v3.0                     ║
║       Micro-Signal Trend Analysis & Campaign Recommendation          ║
║                                                                      ║
║  Optimised with 2026 best practices from:                            ║
║    • Anthropic — Effective Context Engineering for AI Agents         ║
║    • Manus    — KV-cache hit rate as #1 production metric            ║
║    • ACE      — Agentic Context Engineering (arXiv 2510.04618)       ║
║    • Factory  — Anchored iterative compression                       ║
║    • Kubiya   — 12-Factor Agent / micro-agent pattern                ║
║    • Neo4j    — Context poisoning prevention                         ║
║    • Liu 2023 — Lost in the Middle                                   ║
║                                                                      ║
║  KEY OPTIMISATIONS OVER v2:                                          ║
║    1. Cache-aware prefix design (stable system prompt prefix)        ║
║    2. Token budget accounting per zone                               ║
║    3. Context compaction with anchored summaries                     ║
║    4. Context poisoning prevention (input validation layer)          ║
║    5. Goldilocks altitude prompts (not too brittle, not too vague)   ║
║    6. Evolving playbook memory (ACE-style reflection + curation)     ║
║    7. Confidence-weighted context trimming                           ║
║    8. Few-shot as canonical examples (Anthropic's "pictures")        ║
║    9. Structured output with YAML                                    ║
║   10. Observability hooks for production monitoring                  ║
║                                                                      ║
║  OUTPUT FORMAT: YAML                                                 ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:

    ┌──────────────────────────────────────────────────────────────┐
    │  STABLE PREFIX  (cacheable — never changes between calls)    │
    │    Role + Column Defs + Output Schema + Constraints          │
    │    → Maximises KV-cache hit rate (Manus: #1 metric)          │
    ├──────────────────────────────────────────────────────────────┤
    │  DYNAMIC ZONE   (changes per call — not cached)              │
    │    Pre-conditioned data + trend memory                       │
    │    → Minimised via confidence-weighted trimming              │
    ├──────────────────────────────────────────────────────────────┤
    │  STABLE SUFFIX  (cacheable — identical across calls)         │
    │    Task restatement + few-shot examples + guardrails         │
    │    → Re-anchors attention at end (Lost-in-the-Middle)        │
    └──────────────────────────────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
import yaml
import re
import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# ═════════════════════════════════════════════════════════════════════
#  OBSERVABILITY — Token Budget + Production Metrics
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TokenBudget:
    """Explicit token accounting per zone."""
    zone_a_system: int = 0
    zone_b_data:   int = 0
    zone_c_suffix: int = 0
    total_limit:   int = 120_000

    @property
    def total_used(self) -> int:
        return self.zone_a_system + self.zone_b_data + self.zone_c_suffix

    @property
    def remaining(self) -> int:
        return max(0, self.total_limit - self.total_used)

    @property
    def cache_ratio(self) -> float:
        cacheable = self.zone_a_system + self.zone_c_suffix
        return cacheable / max(1, self.total_used)

    def is_within_budget(self) -> bool:
        return self.total_used <= self.total_limit

    def to_dict(self) -> dict:
        return {
            "zone_a_tokens": self.zone_a_system,
            "zone_b_tokens": self.zone_b_data,
            "zone_c_tokens": self.zone_c_suffix,
            "total_used": self.total_used,
            "total_limit": self.total_limit,
            "remaining": self.remaining,
            "cache_ratio": round(self.cache_ratio, 3),
            "within_budget": self.is_within_budget(),
        }


@dataclass
class PipelineMetrics:
    """Production observability for every pipeline run."""
    run_id: str = ""
    timestamp: str = ""
    step1_rows: int = 0
    step1_cols: int = 0
    step2_token_budget: dict = field(default_factory=dict)
    step2_prompt_hash: str = ""
    step3_token_budget: dict = field(default_factory=dict)
    step3_prompt_hash: str = ""
    step4_checks: int = 0
    step4_passed: int = 0
    step4_critical: int = 0
    validation_status: str = ""
    total_latency_ms: float = 0

    def to_yaml(self) -> str:
        return yaml.dump({"pipeline_metrics": asdict(self)}, default_flow_style=False)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════
#  CONTEXT POISONING PREVENTION
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    is_clean: bool = True
    rows_before: int = 0
    rows_after: int = 0
    rows_dropped: int = 0
    issues: List[str] = field(default_factory=list)


def validate_and_clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Pre-flight context poisoning check.
    Removes NaN, negative counts, extreme outliers, duplicates,
    and invalid categoricals before data enters the context window.
    """
    report = DataQualityReport(rows_before=len(df))
    df = df.copy()

    for col in ["wow_purchase_velocity_pct", "purchase_rate", "active_users", "trend_flag"]:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                report.issues.append(f"Dropped {n} NaN in {col}")
                df = df.dropna(subset=[col])

    if "active_users" in df.columns:
        neg = (df["active_users"] < 0).sum()
        if neg > 0:
            report.issues.append(f"Dropped {neg} rows with negative active_users")
            df = df[df["active_users"] >= 0]

    if "wow_purchase_velocity_pct" in df.columns:
        cap = 5.0
        outliers = (df["wow_purchase_velocity_pct"].abs() > cap).sum()
        if outliers > 0:
            report.issues.append(f"Capped {outliers} velocity outliers at ±{cap}")
            df["wow_purchase_velocity_pct"] = df["wow_purchase_velocity_pct"].clip(-cap, cap)

    dup = df.duplicated().sum()
    if dup > 0:
        report.issues.append(f"Dropped {dup} duplicates")
        df = df.drop_duplicates()

    valid_ag = {"Gen-Z", "Millennial", "Gen-X", "Boomer", "Unknown"}
    if "age_group" in df.columns:
        inv = ~df["age_group"].isin(valid_ag)
        if inv.sum() > 0:
            report.issues.append(f"Dropped {inv.sum()} invalid age_group rows")
            df = df[~inv]

    report.rows_after = len(df)
    report.rows_dropped = report.rows_before - report.rows_after
    report.is_clean = len(report.issues) == 0
    return df, report


# ═════════════════════════════════════════════════════════════════════
#  STABLE PREFIX — Cacheable System Prompt Components
# ═════════════════════════════════════════════════════════════════════

COLUMN_DEFINITIONS = """\
<COLUMN_DEFINITIONS>
  age_group                   : Generational cohort (Gen-Z, Millennial, Gen-X, Boomer)
  product_category            : Product line identifier
  trend_flag                  : RISING (>+10% WoW) | FALLING (<-10% WoW) | STABLE (±10%)
  wow_purchase_velocity_pct   : Week-over-week % change in purchases for this cohort
  gender                      : Male / Female
  city                        : City of the user cohort
  active_users                : Distinct active users in cohort this week
  purchasers                  : Distinct users who purchased
  purchase_count              : Total purchase events
  total_revenue               : Sum of revenue (INR)
  page_views                  : Total page views
  sessions                    : Total sessions
  add_to_cart_events          : Total add-to-cart events
  purchase_rate               : purchasers / active_users
  cart_conversion_rate        : purchase_count / add_to_cart_events
  pages_per_session           : page_views / sessions
  web_footprint_vs_national   : Ratio vs national avg (>1 = above average)
</COLUMN_DEFINITIONS>"""

TREND_SCHEMA_YAML = """\
trends:
  - rank: 1
    trend_title: "Short descriptive title"
    age_group: "Gen-Z | Millennial | Gen-X | Boomer"
    product_category: "Product from data"
    gender: "All | Male | Female"
    city: "All | specific city"
    signal_type: "purchase_velocity | web_footprint | cart_conversion | engagement"
    direction: "RISING | FALLING"
    metric_value: 0.0
    metric_name: "column name"
    benchmark_comparison: "X% above/below benchmark"
    confidence: "HIGH | MEDIUM | LOW"
    evidence_summary: "1-2 sentence grounded explanation"
data_quality_notes: "Caveats about sample sizes"
"""

REC_SCHEMA_YAML = """\
recommendations:
  - rank: 1
    campaign_name: "Campaign name"
    cited_trend_ids: ["TREND-1"]
    target_segment:
      age_group: "Gen-Z"
      gender: "All | Male | Female"
      city: "All | specific city"
    product_focus: "Product-X"
    campaign_type: "ACQUISITION | RETENTION | REACTIVATION | UPSELL"
    recommended_channels: ["PUSH_NOTIFICATION", "EMAIL"]
    budget_tier: "TIER_1 | TIER_2 | TIER_3"
    estimated_reach: "user count from data"
    key_message: "Value proposition"
    timing: "Immediate | Next 7 days | Next 30 days"
    success_metric: "KPI to track"
    expected_lift: "% improvement with reasoning"
    rationale: "2-3 sentences grounded in cited trend"
campaigns_not_recommended:
  - segment: "description"
    reason: "Why excluded"
"""

GROUNDING_RULES = """\
<GROUNDING_RULES>
1. Every insight MUST reference a specific (age_group, product_category) pair.
2. Every numeric claim MUST trace to a value in the data table.
3. Do NOT infer trends absent from the data.
4. Weak signals (|velocity| < 5%) are NOT top trends.
5. Flag confidence: HIGH (large cohort + strong signal), MEDIUM, LOW.
6. Return YAML only. No markdown fences. No commentary outside YAML.
</GROUNDING_RULES>"""

CAMPAIGN_CONSTRAINTS = """\
<CAMPAIGN_CONSTRAINTS>
Targeting: At least one of age_group, gender, city must be specific.
Channels:
  PUSH_NOTIFICATION : low cost, high reach, low conversion
  EMAIL             : low cost, medium reach, medium conversion
  SMS               : medium cost, medium reach, high conversion (opt-in required)
  IN_APP_BANNER     : no cost, high reach, low conversion
  SOCIAL_ADS        : high cost, high reach, variable conversion
  WHATSAPP          : medium cost, medium reach, high conversion (opt-in required)
Budget tiers:
  TIER_1 (INR 0-50K)  : push, in-app, email only
  TIER_2 (INR 50K-2L) : + SMS, WhatsApp
  TIER_3 (INR 2L+)    : + social ads
Rules:
  - No targeting under-18 users.
  - FALLING trends → RETENTION, not ACQUISITION.
</CAMPAIGN_CONSTRAINTS>"""

# Assembled stable prefixes
TREND_STABLE_PREFIX = f"""\
<ROLE>
You are a Senior CRM Analytics Strategist. Identify the TOP 5 most
actionable demographic-behavioural trends from the data below.
</ROLE>

{COLUMN_DEFINITIONS}

{GROUNDING_RULES}

<OUTPUT_FORMAT>
Return YAML matching this schema:

{TREND_SCHEMA_YAML}
</OUTPUT_FORMAT>"""

REC_STABLE_PREFIX = f"""\
<ROLE>
You are a CRM Campaign Strategist. Design the TOP 5 targeted marketing
campaigns based on the trend insights below.
</ROLE>

{CAMPAIGN_CONSTRAINTS}

<OUTPUT_FORMAT>
Return YAML matching this schema:

{REC_SCHEMA_YAML}
</OUTPUT_FORMAT>"""

# Stable suffixes
TREND_STABLE_SUFFIX = """\
<TASK>
Identify TOP 5 trends. Prioritise:
  (a) Large cohort (active_users > 500)
  (b) Strong signal (|wow_purchase_velocity_pct| > 15% OR web_footprint_vs_national > 1.3)
  (c) Actionable for demographic-targeted campaigns
Rank by business_impact = revenue × cohort_size × signal_strength.
Every claim must cite a specific value from the table. No hallucinations.
Return YAML only.
</TASK>"""

REC_STABLE_SUFFIX = """\
<EXAMPLE>
A good recommendation:
- rank: 1
  campaign_name: "Gen-Z Product-X Flash Sale"
  cited_trend_ids: ["TREND-1"]
  target_segment:
    age_group: "Gen-Z"
    gender: "All"
    city: "All"
  product_focus: "Product-X"
  campaign_type: "ACQUISITION"
  recommended_channels: ["PUSH_NOTIFICATION", "IN_APP_BANNER"]
  budget_tier: "TIER_1"
  estimated_reach: "~12,000 active Gen-Z users"
  key_message: "Product-X is trending with your peers"
  timing: "Immediate"
  success_metric: "purchase_rate for Gen-Z x Product-X"
  expected_lift: "8-12% based on 18% WoW velocity"
  rationale: >
    TREND-1 shows Gen-Z purchase velocity on Product-X is RISING
    at +18% WoW with HIGH confidence. Low-cost push + in-app amplifies momentum.
</EXAMPLE>

<TASK>
Design TOP 5 campaigns for {n_trends} trends above.
Rules:
  1. Every recommendation MUST cite a TREND-ID.
  2. No campaigns for segments without a supporting trend.
  3. FALLING → RETENTION. RISING → ACQUISITION or UPSELL.
  4. Match channels to budget tier.
  5. Ground estimated_reach in active_users from trends.
Return YAML only.
</TASK>"""


# ═════════════════════════════════════════════════════════════════════
#  RETRIEVAL — SQL + Synthetic Data
# ═════════════════════════════════════════════════════════════════════

AGGREGATION_SQL = """
-- Context Engineering: Micro-Signal Query (Lost-in-the-Middle column order)
WITH base_events AS (
    SELECT user_id, event_timestamp, event_type, product_id, product_category, revenue, session_id
    FROM `project.dataset.user_events`
    WHERE event_timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 WEEK)
),
user_demographics AS (
    SELECT user_id, age, gender, city,
        CASE WHEN age BETWEEN 18 AND 27 THEN 'Gen-Z'
             WHEN age BETWEEN 28 AND 43 THEN 'Millennial'
             WHEN age BETWEEN 44 AND 59 THEN 'Gen-X'
             WHEN age >= 60 THEN 'Boomer' ELSE 'Unknown' END AS age_group
    FROM `project.dataset.user_profiles`
),
weekly_signals AS (
    SELECT d.age_group, d.gender, d.city, e.product_category,
        DATE_TRUNC(e.event_timestamp, WEEK(MONDAY)) AS week_start,
        COUNT(DISTINCT CASE WHEN e.event_type='purchase' THEN e.user_id END) AS purchasers,
        COUNT(CASE WHEN e.event_type='purchase' THEN 1 END) AS purchase_count,
        COALESCE(SUM(CASE WHEN e.event_type='purchase' THEN e.revenue END),0) AS total_revenue,
        COUNT(CASE WHEN e.event_type='page_view' THEN 1 END) AS page_views,
        COUNT(DISTINCT CASE WHEN e.event_type='page_view' THEN e.session_id END) AS sessions,
        COUNT(CASE WHEN e.event_type='add_to_cart' THEN 1 END) AS add_to_cart_events,
        COUNT(DISTINCT e.user_id) AS active_users
    FROM base_events e JOIN user_demographics d ON e.user_id = d.user_id
    GROUP BY 1,2,3,4,5
),
with_trends AS (
    SELECT ws.*,
        LAG(ws.purchase_count) OVER (PARTITION BY ws.age_group, ws.gender, ws.city, ws.product_category ORDER BY ws.week_start) AS prev_week_purchases,
        SAFE_DIVIDE(ws.purchasers, ws.active_users) AS purchase_rate,
        SAFE_DIVIDE(ws.purchase_count, ws.add_to_cart_events) AS cart_conversion_rate,
        SAFE_DIVIDE(ws.page_views, ws.sessions) AS pages_per_session,
        AVG(ws.page_views) OVER (PARTITION BY ws.product_category, ws.week_start) AS national_avg_page_views
    FROM weekly_signals ws
)
SELECT age_group, gender, city, product_category, week_start,
    active_users, purchasers, purchase_count, total_revenue, page_views, sessions, add_to_cart_events,
    purchase_rate, cart_conversion_rate, pages_per_session, national_avg_page_views,
    SAFE_DIVIDE(purchase_count - prev_week_purchases, prev_week_purchases) AS wow_purchase_velocity_pct,
    SAFE_DIVIDE(page_views, national_avg_page_views) AS web_footprint_vs_national,
    CASE WHEN SAFE_DIVIDE(purchase_count - prev_week_purchases, prev_week_purchases) > 0.10 THEN 'RISING'
         WHEN SAFE_DIVIDE(purchase_count - prev_week_purchases, prev_week_purchases) < -0.10 THEN 'FALLING'
         ELSE 'STABLE' END AS trend_flag
FROM with_trends WHERE week_start >= DATE_SUB(CURRENT_DATE(), INTERVAL 4 WEEK)
ORDER BY age_group, product_category, week_start DESC
"""


def generate_synthetic_data(n_weeks: int = 4, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    ag_list = ["Gen-Z", "Millennial", "Gen-X", "Boomer"]
    g_list = ["Male", "Female"]
    c_list = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"]
    p_list = ["Product-X", "Product-Y", "Product-Z", "Product-W", "Product-V"]
    weeks = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_weeks, freq="W-MON")
    rows = []
    for wi, ws in enumerate(weeks):
        for ag in ag_list:
            for g in g_list:
                for c in c_list:
                    for p in p_list:
                        au = np.random.randint(200, 2000)
                        bp = int(au * np.random.uniform(0.02, 0.12))
                        pv = int(au * np.random.uniform(5, 25))
                        ss = int(au * np.random.uniform(1.5, 5))
                        atc = int(bp * np.random.uniform(1.5, 4))
                        if ag == "Gen-Z" and p == "Product-X": bp = int(bp * (1 + 0.18 * wi))
                        if ag == "Millennial" and p == "Product-Y": bp = int(bp * max(0.3, 1 - 0.15 * wi))
                        if g == "Male" and c == "Bengaluru": pv = int(pv * 1.6)
                        if ag == "Gen-X" and p == "Product-Z" and g == "Female": bp = int(bp * 1.4)
                        if ag == "Boomer" and p == "Product-W" and c == "Delhi": atc = int(atc * 2.5); bp = max(1, int(bp * 0.4))
                        rows.append({"age_group": ag, "gender": g, "city": c, "product_category": p,
                                     "week_start": ws.strftime("%Y-%m-%d"), "active_users": au,
                                     "purchasers": max(1, bp), "purchase_count": max(1, bp),
                                     "total_revenue": round(bp * np.random.uniform(200, 3000), 2),
                                     "page_views": pv, "sessions": ss, "add_to_cart_events": atc})
    df = pd.DataFrame(rows)
    df["purchase_rate"] = (df["purchasers"] / df["active_users"]).round(4)
    df["cart_conversion_rate"] = (df["purchase_count"] / df["add_to_cart_events"]).round(4)
    df["pages_per_session"] = (df["page_views"] / df["sessions"]).round(2)
    navg = df.groupby(["product_category", "week_start"])["page_views"].transform("mean")
    df["national_avg_page_views"] = navg.round(2)
    df["web_footprint_vs_national"] = (df["page_views"] / navg).round(4)
    df = df.sort_values(["age_group", "gender", "city", "product_category", "week_start"])
    df["prev_week_purchases"] = df.groupby(["age_group", "gender", "city", "product_category"])["purchase_count"].shift(1)
    df["wow_purchase_velocity_pct"] = ((df["purchase_count"] - df["prev_week_purchases"]) / df["prev_week_purchases"]).round(4)
    df["trend_flag"] = np.where(df["wow_purchase_velocity_pct"] > 0.10, "RISING",
                                np.where(df["wow_purchase_velocity_pct"] < -0.10, "FALLING", "STABLE"))
    return df


# ═════════════════════════════════════════════════════════════════════
#  CONTEXT PROCESSING — Trimming + Serialisation
# ═════════════════════════════════════════════════════════════════════

COL_ORDER = ["age_group", "product_category", "trend_flag", "wow_purchase_velocity_pct",
             "gender", "city", "week_start", "active_users", "purchasers", "purchase_count",
             "total_revenue", "page_views", "sessions", "add_to_cart_events",
             "purchase_rate", "cart_conversion_rate", "pages_per_session",
             "web_footprint_vs_national", "national_avg_page_views"]


def confidence_weighted_trim(df, min_users=50, top_n=5, max_rows=100):
    df = df[df["active_users"] >= min_users].copy()
    latest = df["week_start"].max()
    dl = df[df["week_start"] == latest].copy()
    dl["_score"] = dl["wow_purchase_velocity_pct"].abs() * np.log1p(dl["active_users"])
    dt = dl.sort_values("_score", ascending=False).groupby(["age_group", "product_category"]).head(top_n).drop(columns=["_score"]).head(max_rows)
    sm = dl.groupby("product_category").agg(
        avg_purchase_rate=("purchase_rate", "mean"), avg_velocity=("wow_purchase_velocity_pct", "mean"),
        avg_web_footprint=("web_footprint_vs_national", "mean"), total_users=("active_users", "sum")).reset_index()
    return dt, sm


def serialise(df, max_rows=100):
    cols = [c for c in COL_ORDER if c in df.columns]
    return df[cols].head(max_rows).to_markdown(index=False)


def export_csv(df, output_dir="outputs"):
    cols = [c for c in COL_ORDER if c in df.columns]
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(output_dir, f"micro_signals_{ts}.csv")
    df[cols].to_csv(p, index=False)
    return p


# ═════════════════════════════════════════════════════════════════════
#  MEMORY — ACE Playbook + Inter-Call State
# ═════════════════════════════════════════════════════════════════════

@dataclass
class PlaybookEntry:
    trend_id: str; title: str; segment: str; product: str
    signal: str; direction: str; value: float; metric: str
    benchmark: str; confidence: str; evidence: str
    occurrences: int = 1
    prior_actions: List[str] = field(default_factory=list)

    def to_block(self):
        lines = [f"[{self.trend_id}]",
                 f"  Title      : {self.title}", f"  Segment    : {self.segment}",
                 f"  Product    : {self.product}", f"  Signal     : {self.signal} → {self.direction}",
                 f"  Value      : {self.metric} = {self.value}", f"  Benchmark  : {self.benchmark}",
                 f"  Confidence : {self.confidence}", f"  Persistence: {self.occurrences} week(s)"]
        if self.prior_actions: lines.append(f"  Prior campaigns: {', '.join(self.prior_actions)}")
        lines.append(f"  Evidence   : {self.evidence}")
        return "\n".join(lines)


def build_playbook(ty):
    if isinstance(ty, str): ty = yaml.safe_load(ty)
    return [PlaybookEntry(trend_id=f"TREND-{t['rank']}", title=t["trend_title"],
            segment=f"{t['age_group']} / {t['gender']} / {t.get('city','All')}",
            product=t["product_category"], signal=t["signal_type"], direction=t["direction"],
            value=t["metric_value"], metric=t["metric_name"], benchmark=t["benchmark_comparison"],
            confidence=t["confidence"], evidence=t["evidence_summary"]) for t in ty.get("trends", [])]


def format_playbook(entries, notes=""):
    blocks = [e.to_block() for e in entries]
    if notes: blocks.append(f"\n[DATA-QUALITY-NOTES]\n  {notes}")
    return "\n\n".join(blocks)


# ═════════════════════════════════════════════════════════════════════
#  PROMPT ASSEMBLY — Cache-Aware
# ═════════════════════════════════════════════════════════════════════

def build_trend_prompt(df, budget=None):
    if budget is None: budget = TokenBudget()
    zone_a = TREND_STABLE_PREFIX
    budget.zone_a_system = estimate_tokens(zone_a)
    dt, sm = confidence_weighted_trim(df)
    latest = df["week_start"].max()
    zone_b = f"\n---\n## DATA — Latest Week: {latest}\n\n{serialise(dt)}\n\n---\n### National Summary\n\n{serialise(sm)}\n---"
    budget.zone_b_data = estimate_tokens(zone_b)
    zone_c = TREND_STABLE_SUFFIX
    budget.zone_c_suffix = estimate_tokens(zone_c)
    if not budget.is_within_budget():
        for mr in [80, 60, 40]:
            dt, sm = confidence_weighted_trim(df, max_rows=mr)
            zone_b = f"\n---\n## DATA — Latest Week: {latest}\n\n{serialise(dt, mr)}\n\n---\n### National Summary\n\n{serialise(sm)}\n---"
            budget.zone_b_data = estimate_tokens(zone_b)
            if budget.is_within_budget(): break
    return f"{zone_a}\n{zone_b}\n{zone_c}", budget


def build_rec_prompt(trend_yaml, budget=None):
    if budget is None: budget = TokenBudget()
    zone_a = REC_STABLE_PREFIX
    budget.zone_a_system = estimate_tokens(zone_a)
    entries = build_playbook(trend_yaml)
    notes = trend_yaml.get("data_quality_notes", "")
    zone_b = f"\n---\n## TRENDS\n\nEach has a TREND-ID to cite.\n\n{format_playbook(entries, notes)}\n---"
    budget.zone_b_data = estimate_tokens(zone_b)
    n = len(trend_yaml.get("trends", []))
    zone_c = REC_STABLE_SUFFIX.format(n_trends=n)
    budget.zone_c_suffix = estimate_tokens(zone_c)
    return f"{zone_a}\n{zone_b}\n{zone_c}", budget


# ═════════════════════════════════════════════════════════════════════
#  TOOLS — Parser + Compaction
# ═════════════════════════════════════════════════════════════════════

def parse_yaml_output(raw):
    c = re.sub(r'^```ya?ml\s*', '', raw.strip(), flags=re.MULTILINE)
    c = re.sub(r'^```\s*$', '', c, flags=re.MULTILINE).strip()
    try: return yaml.safe_load(c)
    except: return {"error": "Parse failed", "raw": raw[:500]}


def compact_state(trend_yaml, rec_yaml, status):
    """Anchored compaction: persistent summary for next pipeline run."""
    return yaml.dump({"pipeline_state": {
        "generated_at": datetime.now().isoformat(),
        "trends": [{"id": f"TREND-{t['rank']}", "segment": f"{t['age_group']} x {t['product_category']}",
                     "direction": t["direction"], "magnitude": t["metric_value"], "confidence": t["confidence"]}
                    for t in trend_yaml.get("trends", [])],
        "campaigns": [{"rank": r["rank"], "name": r["campaign_name"], "cites": r["cited_trend_ids"],
                        "type": r["campaign_type"], "status": "PENDING"}
                       for r in rec_yaml.get("recommendations", [])],
        "validation": status,
        "data_quality": trend_yaml.get("data_quality_notes", "None")}}, default_flow_style=False)


# ═════════════════════════════════════════════════════════════════════
#  VALIDATION — 6-Layer Deterministic Guard Rail
# ═════════════════════════════════════════════════════════════════════

class Severity(Enum):
    CRITICAL = "CRITICAL"; WARNING = "WARNING"; INFO = "INFO"

@dataclass
class ValidationIssue:
    layer: str; severity: Severity; field_path: str; message: str
    expected: Any = None; actual: Any = None

@dataclass
class ValidationReport:
    is_valid: bool = True; total_checks: int = 0; passed: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, i):
        self.issues.append(i)
        if i.severity == Severity.CRITICAL: self.is_valid = False

    def add_pass(self): self.passed += 1

    def summary(self):
        cr = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        wr = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        lines = ["=" * 65, "GROUNDING VALIDATION REPORT", "=" * 65,
                 f"  Status: {'PASSED' if self.is_valid else 'FAILED'}  |  Checks: {self.total_checks}  |  Passed: {self.passed}",
                 f"  Critical: {cr}  |  Warnings: {wr}", "-" * 65]
        for i in self.issues:
            ic = {"CRITICAL": "[!!]", "WARNING": "[! ]", "INFO": "[i ]"}[i.severity.value]
            lines.append(f"  {ic} {i.layer}: {i.field_path} — {i.message}")
            if i.expected: lines.append(f"      Expected: {i.expected}  |  Actual: {i.actual}")
        if not self.issues: lines.append("  All checks passed.")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_yaml(self):
        return yaml.dump({"validation_report": {
            "status": "PASSED" if self.is_valid else "FAILED",
            "total_checks": self.total_checks, "passed": self.passed,
            "critical": sum(1 for i in self.issues if i.severity == Severity.CRITICAL),
            "warnings": sum(1 for i in self.issues if i.severity == Severity.WARNING),
            "issues": [{"layer": i.layer, "severity": i.severity.value, "field": i.field_path,
                         "message": i.message} for i in self.issues]}}, default_flow_style=False)

REQ_FIELDS = {"rank", "campaign_name", "cited_trend_ids", "target_segment", "product_focus",
              "campaign_type", "recommended_channels", "budget_tier", "key_message", "rationale"}
SEG_FIELDS = {"age_group", "gender", "city"}
VALID_TYPES = {"ACQUISITION", "RETENTION", "REACTIVATION", "UPSELL"}
VALID_CH = {"PUSH_NOTIFICATION", "EMAIL", "SMS", "IN_APP_BANNER", "SOCIAL_ADS", "WHATSAPP"}
VALID_TIERS = {"TIER_1", "TIER_2", "TIER_3"}
TIER_CH = {"TIER_1": {"PUSH_NOTIFICATION", "IN_APP_BANNER", "EMAIL"},
           "TIER_2": {"PUSH_NOTIFICATION", "IN_APP_BANNER", "EMAIL", "SMS", "WHATSAPP"},
           "TIER_3": {"PUSH_NOTIFICATION", "IN_APP_BANNER", "EMAIL", "SMS", "WHATSAPP", "SOCIAL_ADS"}}


def validate_recommendations(rec_yaml, trend_yaml, df):
    rp = ValidationReport()
    recs = rec_yaml.get("recommendations", [])
    tids = {f"TREND-{t['rank']}" for t in trend_yaml.get("trends", [])}
    tmap = {f"TREND-{t['rank']}": t for t in trend_yaml.get("trends", [])}
    vag = set(df["age_group"].unique()) | {"All"}
    vg = set(df["gender"].unique()) | {"All"}
    vc = set(df["city"].unique()) | {"All"}
    vp = set(df["product_category"].unique()) | {"All"}
    pre = re.compile(r'Product-[A-Z]')
    if not recs:
        rp.add_issue(ValidationIssue("SCHEMA", Severity.CRITICAL, "recommendations", "Empty"))
        return rp
    for idx, r in enumerate(recs):
        p = f"rec[{idx}]"
        seg = r.get("target_segment", {})
        # L1 Schema
        rp.total_checks += 1
        m = REQ_FIELDS - set(r.keys())
        if m: rp.add_issue(ValidationIssue("SCHEMA", Severity.CRITICAL, p, f"Missing: {m}"))
        else: rp.add_pass()
        rp.total_checks += 1
        if SEG_FIELDS - set(seg.keys()): rp.add_issue(ValidationIssue("SCHEMA", Severity.WARNING, f"{p}.seg", f"Missing: {SEG_FIELDS - set(seg.keys())}"))
        else: rp.add_pass()
        for fn, vs in [("campaign_type", VALID_TYPES), ("budget_tier", VALID_TIERS)]:
            rp.total_checks += 1
            v = r.get(fn, "")
            if v not in vs: rp.add_issue(ValidationIssue("SCHEMA", Severity.WARNING, f"{p}.{fn}", f"Invalid: {v}", vs, v))
            else: rp.add_pass()
        rp.total_checks += 1
        bc = set(r.get("recommended_channels", [])) - VALID_CH
        if bc: rp.add_issue(ValidationIssue("SCHEMA", Severity.WARNING, f"{p}.channels", f"Invalid: {bc}"))
        else: rp.add_pass()
        # L2 Citation
        cited = r.get("cited_trend_ids", [])
        rp.total_checks += 1
        if not cited: rp.add_issue(ValidationIssue("CITATION", Severity.CRITICAL, f"{p}.cited", "No citations"))
        else:
            for ci in cited:
                rp.total_checks += 1
                if ci not in tids: rp.add_issue(ValidationIssue("CITATION", Severity.CRITICAL, f"{p}.cited", f"'{ci}' missing", tids, ci))
                else: rp.add_pass()
        # L3 Segment
        for fn, vs in [("age_group", vag), ("gender", vg), ("city", vc)]:
            rp.total_checks += 1
            v = seg.get(fn, "All")
            if v not in vs: rp.add_issue(ValidationIssue("SEGMENT", Severity.CRITICAL, f"{p}.{fn}", f"Hallucinated: '{v}'", vs, v))
            else: rp.add_pass()
        rp.total_checks += 1
        pf = r.get("product_focus", "All")
        if pf not in vp: rp.add_issue(ValidationIssue("SEGMENT", Severity.CRITICAL, f"{p}.product", f"Hallucinated: '{pf}'", vp, pf))
        else: rp.add_pass()
        # L4 Metric
        rat = r.get("rationale", "")
        pcts = re.findall(r'(\d+(?:\.\d+)?)\s*%', rat)
        for ci in cited:
            if ci not in tmap: continue
            actual = abs(tmap[ci].get("metric_value", 0))
            if pcts:
                rp.total_checks += 1
                if any(abs(float(x)/100 - actual) <= 0.05 for x in pcts): rp.add_pass()
                else: rp.add_issue(ValidationIssue("METRIC", Severity.WARNING, f"{p}.rationale", f"Pcts {pcts}% vs {actual:.2%}"))
        # L5 Business
        rp.total_checks += 1
        tier = r.get("budget_tier", "TIER_1")
        ic = set(r.get("recommended_channels", [])) - TIER_CH.get(tier, set())
        if ic: rp.add_issue(ValidationIssue("BUSINESS", Severity.WARNING, f"{p}.channels", f"{ic} not in {tier}"))
        else: rp.add_pass()
        for ci in cited:
            if ci not in tmap: continue
            d = tmap[ci].get("direction", ""); ct = r.get("campaign_type", "")
            rp.total_checks += 1
            if d == "FALLING" and ct == "ACQUISITION": rp.add_issue(ValidationIssue("BUSINESS", Severity.WARNING, f"{p}.type", f"ACQUISITION for FALLING {ci}"))
            else: rp.add_pass()
        # L6 Hallucination
        for fn in ["rationale", "key_message"]:
            txt = r.get(fn, ""); found = pre.findall(txt)
            rp.total_checks += 1
            for fp in found:
                if fp not in df["product_category"].unique(): rp.add_issue(ValidationIssue("HALLUCINATION", Severity.CRITICAL, f"{p}.{fn}", f"'{fp}' not in data"))
                else: rp.add_pass()
    return rp


# ═════════════════════════════════════════════════════════════════════
#  SAMPLE OUTPUTS
# ═════════════════════════════════════════════════════════════════════

SAMPLE_TRENDS = {
    "trends": [
        {"rank": 1, "trend_title": "Gen-Z purchase surge on Product-X", "age_group": "Gen-Z", "product_category": "Product-X", "gender": "All", "city": "All", "signal_type": "purchase_velocity", "direction": "RISING", "metric_value": 0.54, "metric_name": "wow_purchase_velocity_pct", "benchmark_comparison": "54% above previous week", "confidence": "HIGH", "evidence_summary": "Gen-Z +54% WoW on Product-X, 3200+ active users."},
        {"rank": 2, "trend_title": "Millennial Product-Y decline", "age_group": "Millennial", "product_category": "Product-Y", "gender": "All", "city": "All", "signal_type": "purchase_velocity", "direction": "FALLING", "metric_value": -0.45, "metric_name": "wow_purchase_velocity_pct", "benchmark_comparison": "45% below previous week", "confidence": "HIGH", "evidence_summary": "Millennial velocity -45% WoW on Product-Y."},
        {"rank": 3, "trend_title": "Male Bengaluru web spike", "age_group": "All", "product_category": "All", "gender": "Male", "city": "Bengaluru", "signal_type": "web_footprint", "direction": "RISING", "metric_value": 1.62, "metric_name": "web_footprint_vs_national", "benchmark_comparison": "62% above national avg", "confidence": "HIGH", "evidence_summary": "Male Bengaluru page views 62% above national."},
        {"rank": 4, "trend_title": "Gen-X female Product-Z demand", "age_group": "Gen-X", "product_category": "Product-Z", "gender": "Female", "city": "All", "signal_type": "purchase_velocity", "direction": "RISING", "metric_value": 0.22, "metric_name": "wow_purchase_velocity_pct", "benchmark_comparison": "22% above previous week", "confidence": "MEDIUM", "evidence_summary": "Gen-X females +22% WoW on Product-Z."},
        {"rank": 5, "trend_title": "Boomer cart drop Product-W Delhi", "age_group": "Boomer", "product_category": "Product-W", "gender": "All", "city": "Delhi", "signal_type": "cart_conversion", "direction": "FALLING", "metric_value": 0.16, "metric_name": "cart_conversion_rate", "benchmark_comparison": "16% vs 40% category avg", "confidence": "MEDIUM", "evidence_summary": "Boomers Delhi 16% cart conversion on Product-W."},
    ],
    "data_quality_notes": "Boomer small cities < 100 users — treat with caution."
}

SAMPLE_RECS = {
    "recommendations": [
        {"rank": 1, "campaign_name": "Gen-Z Product-X momentum push", "cited_trend_ids": ["TREND-1"], "target_segment": {"age_group": "Gen-Z", "gender": "All", "city": "All"}, "product_focus": "Product-X", "campaign_type": "ACQUISITION", "recommended_channels": ["PUSH_NOTIFICATION", "IN_APP_BANNER"], "budget_tier": "TIER_1", "estimated_reach": "~3200 Gen-Z users", "key_message": "Product-X is trending in your age group", "timing": "Immediate", "success_metric": "purchase_rate Gen-Z x Product-X", "expected_lift": "10-15%", "rationale": "TREND-1: Gen-Z velocity +54% WoW, HIGH. Push+in-app amplifies at TIER_1."},
        {"rank": 2, "campaign_name": "Millennial Product-Y win-back", "cited_trend_ids": ["TREND-2"], "target_segment": {"age_group": "Millennial", "gender": "All", "city": "All"}, "product_focus": "Product-Y", "campaign_type": "RETENTION", "recommended_channels": ["EMAIL", "WHATSAPP"], "budget_tier": "TIER_2", "estimated_reach": "~5000 Millennial users", "key_message": "We refreshed Product-Y for you", "timing": "Next 7 days", "success_metric": "purchase_rate Millennial x Product-Y", "expected_lift": "5-8%", "rationale": "TREND-2: Millennial velocity -45% WoW. RETENTION via email+WhatsApp."},
        {"rank": 3, "campaign_name": "Bengaluru male browser-to-buyer", "cited_trend_ids": ["TREND-3"], "target_segment": {"age_group": "All", "gender": "Male", "city": "Bengaluru"}, "product_focus": "All", "campaign_type": "ACQUISITION", "recommended_channels": ["IN_APP_BANNER", "PUSH_NOTIFICATION"], "budget_tier": "TIER_1", "estimated_reach": "Male Bengaluru high-browse users", "key_message": "Personalised deal for you", "timing": "Immediate", "success_metric": "cart_conversion Male x Bengaluru", "expected_lift": "8-12%", "rationale": "TREND-3: Male Bengaluru web footprint 62% above national."},
        {"rank": 4, "campaign_name": "Gen-X women Product-Z loyalty", "cited_trend_ids": ["TREND-4"], "target_segment": {"age_group": "Gen-X", "gender": "Female", "city": "All"}, "product_focus": "Product-Z", "campaign_type": "UPSELL", "recommended_channels": ["EMAIL", "SMS"], "budget_tier": "TIER_2", "estimated_reach": "~1500 Gen-X female purchasers", "key_message": "Exclusive Product-Z bundles", "timing": "Next 7 days", "success_metric": "AOV Gen-X Female x Product-Z", "expected_lift": "15-20%", "rationale": "TREND-4: Gen-X females +22% WoW on Product-Z. Upsell bundles."},
        {"rank": 5, "campaign_name": "Delhi Boomer Product-W checkout", "cited_trend_ids": ["TREND-5"], "target_segment": {"age_group": "Boomer", "gender": "All", "city": "Delhi"}, "product_focus": "Product-W", "campaign_type": "RETENTION", "recommended_channels": ["SMS", "WHATSAPP"], "budget_tier": "TIER_2", "estimated_reach": "~800 Boomer Delhi users", "key_message": "Help with your Product-W order", "timing": "Immediate", "success_metric": "cart_conversion Boomer x Product-W x Delhi", "expected_lift": "20-25%", "rationale": "TREND-5: Boomers Delhi 16% cart conversion. SMS/WhatsApp assist."},
    ],
    "campaigns_not_recommended": [{"segment": "Boomer small cities", "reason": "< 100 users."}]
}


# ═════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════

def run_pipeline():
    t0 = time.time()
    m = PipelineMetrics(run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", timestamp=datetime.now().isoformat())
    os.makedirs("outputs", exist_ok=True)

    print("╔" + "═" * 63 + "╗")
    print("║  CONTEXT ENGINEERING v3.0 — Optimised for 2026               ║")
    print("╚" + "═" * 63 + "╝")

    # Step 1
    print("\n─── STEP 1: Retrieval + Poisoning Prevention ───")
    df_raw = generate_synthetic_data()
    df, dq = validate_and_clean_dataframe(df_raw)
    csv_path = export_csv(df)
    m.step1_rows, m.step1_cols = len(df), len(df.columns)
    print(f"  {dq.rows_before} → {dq.rows_after} rows (dropped {dq.rows_dropped})")
    for iss in dq.issues: print(f"  [CLEAN] {iss}")

    # Step 2
    print("\n─── STEP 2: Trend Prompt (cache-aware) ───")
    tp, tb = build_trend_prompt(df)
    with open("outputs/trend_prompt.txt", "w") as f: f.write(tp)
    ph = stable_hash(TREND_STABLE_PREFIX)
    m.step2_token_budget, m.step2_prompt_hash = tb.to_dict(), ph
    print(f"  Tokens: {tb.total_used:,}  |  Cache ratio: {tb.cache_ratio:.0%}  |  Prefix: {ph}")

    to = SAMPLE_TRENDS
    with open("outputs/trend_output.yml", "w") as f: yaml.dump(to, f, default_flow_style=False)
    print(f"  Trends: {len(to['trends'])}")

    # Step 3
    print("\n─── STEP 3: Recommendation Prompt (playbook memory) ───")
    rp, rb = build_rec_prompt(to)
    with open("outputs/rec_prompt.txt", "w") as f: f.write(rp)
    rph = stable_hash(REC_STABLE_PREFIX)
    m.step3_token_budget, m.step3_prompt_hash = rb.to_dict(), rph
    print(f"  Tokens: {rb.total_used:,}  |  Cache ratio: {rb.cache_ratio:.0%}  |  Prefix: {rph}")

    ro = SAMPLE_RECS
    with open("outputs/rec_output.yml", "w") as f: yaml.dump(ro, f, default_flow_style=False)

    # Step 4
    print("\n─── STEP 4: 6-Layer Validation ───")
    vr = validate_recommendations(ro, to, df)
    m.step4_checks, m.step4_passed = vr.total_checks, vr.passed
    m.step4_critical = sum(1 for i in vr.issues if i.severity == Severity.CRITICAL)
    m.validation_status = "PASSED" if vr.is_valid else "FAILED"
    print(vr.summary())
    with open("outputs/validation.yml", "w") as f: f.write(vr.to_yaml())

    # Step 5
    print("\n─── STEP 5: Compaction ───")
    cs = compact_state(to, ro, m.validation_status)
    with open("outputs/compacted_state.yml", "w") as f: f.write(cs)
    print(f"  Compacted: {len(cs)} chars (~{estimate_tokens(cs)} tokens)")

    m.total_latency_ms = round((time.time() - t0) * 1000, 2)
    with open("outputs/metrics.yml", "w") as f: f.write(m.to_yaml())

    print("\n" + "═" * 65)
    print("ARTIFACTS:")
    for fn in sorted(os.listdir("outputs")):
        print(f"  {fn:<40s} {os.path.getsize(os.path.join('outputs', fn)):>8,} bytes")
    print(f"\n  Status: {m.validation_status}  |  Latency: {m.total_latency_ms:.0f}ms")
    print(f"  Cache: trend={tb.cache_ratio:.0%}  rec={rb.cache_ratio:.0%}")
    return {"df": df, "trends": to, "recs": ro, "validation": vr, "metrics": m}


if __name__ == "__main__":
    run_pipeline()

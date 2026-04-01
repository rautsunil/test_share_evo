"""
╔══════════════════════════════════════════════════════════════════════╗
║          GENERIC CONTEXT ENGINEERING FRAMEWORK  v4.0                 ║
║          Schema-Agnostic · Registry-Driven · Any Signal              ║
║                                                                      ║
║  PROBLEM:                                                            ║
║    20+ signals, each with different columns. Purchase velocity has   ║
║    (age_group, product, city). Email engagement has (campaign_type,  ║
║    send_day, segment). App sessions has (os, feature, version).     ║
║    We need ONE framework that handles ALL of them.                   ║
║                                                                      ║
║  SOLUTION: Registry-driven architecture                              ║
║                                                                      ║
║    ┌────────────────┐      ┌────────────────┐                        ║
║    │ Signal Registry │──────│  Signal Config  │  (YAML per signal)   ║
║    │   (catalog)     │      │  - dimensions   │                      ║
║    │                 │      │  - metrics      │                      ║
║    │                 │      │  - trend_col    │                      ║
║    │                 │      │  - thresholds   │                      ║
║    └───────┬────────┘      └────────────────┘                        ║
║            │                                                         ║
║            ▼                                                         ║
║    ┌────────────────┐                                                ║
║    │ Context Engine  │  Reads config → auto-generates:               ║
║    │   (generic)     │    • Zone A (column defs from config)         ║
║    │                 │    • Zone B (data, trimmed by config)          ║
║    │                 │    • Zone C (task, shaped by config)           ║
║    └───────┬────────┘                                                ║
║            │                                                         ║
║            ▼                                                         ║
║    ┌────────────────┐                                                ║
║    │  Validator      │  Validates against config, not hardcoded      ║
║    │   (generic)     │  rules. Works for ANY signal schema.          ║
║    └────────────────┘                                                ║
║                                                                      ║
║  INSPIRED BY:                                                        ║
║    • Samsung Universal CRM Feature Store (YAML-based config)         ║
║    • Anthropic — Cache-aware prefix design                           ║
║    • Manus — KV-cache hit rate as #1 metric                          ║
║    • ACE — Evolving playbook memory                                  ║
║    • Liu 2023 — Lost in the Middle                                   ║
╚══════════════════════════════════════════════════════════════════════╝
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
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from copy import deepcopy


# ═════════════════════════════════════════════════════════════════════
#  CORE ABSTRACTIONS — Signal Config + Registry
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ColumnSpec:
    """Describes a single column in a signal's schema."""
    name:        str
    role:        str        # "dimension" | "metric" | "trend" | "timestamp" | "meta"
    dtype:       str        # "categorical" | "numeric" | "datetime" | "boolean"
    description: str = ""
    valid_values: Optional[List[str]] = None   # For categoricals
    is_primary:  bool = False                  # Primary dimension for grouping


@dataclass
class SignalConfig:
    """
    Universal signal definition. One per signal type.

    This is the SINGLE SOURCE OF TRUTH for:
      - What columns exist
      - Which are dimensions vs metrics
      - How to compute trends
      - What thresholds define RISING/FALLING
      - What the LLM should know about each column
    """
    signal_id:          str                        # e.g. "purchase_velocity"
    signal_name:        str                        # e.g. "Week-over-Week Purchase Velocity"
    description:        str                        # Human-readable purpose
    columns:            List[ColumnSpec]            # Full schema
    trend_column:       str                        # Which metric is the primary trend signal
    rising_threshold:   float = 0.10               # > this = RISING
    falling_threshold:  float = -0.10              # < this = FALLING
    min_cohort_size:    int = 50                   # Minimum group size to include
    cohort_size_column: str = "active_users"       # Column that measures cohort size
    timestamp_column:   str = "week_start"         # Time dimension
    top_n_per_group:    int = 5                    # Max rows per group in context
    max_context_rows:   int = 100                  # Hard cap on data rows in prompt

    @property
    def dimensions(self) -> List[ColumnSpec]:
        return [c for c in self.columns if c.role == "dimension"]

    @property
    def metrics(self) -> List[ColumnSpec]:
        return [c for c in self.columns if c.role in ("metric", "trend")]

    @property
    def primary_dimensions(self) -> List[str]:
        return [c.name for c in self.columns if c.role == "dimension" and c.is_primary]

    @property
    def all_dimension_names(self) -> List[str]:
        return [c.name for c in self.columns if c.role == "dimension"]

    @property
    def all_metric_names(self) -> List[str]:
        return [c.name for c in self.columns if c.role in ("metric", "trend")]

    @property
    def all_valid_categoricals(self) -> Dict[str, Set[str]]:
        """Map of dimension_name → set of valid values (from config)."""
        return {
            c.name: set(c.valid_values)
            for c in self.columns
            if c.role == "dimension" and c.valid_values
        }

    def generate_column_definitions(self) -> str:
        """Auto-generate COLUMN_DEFINITIONS block for Zone A."""
        lines = ["<COLUMN_DEFINITIONS>"]
        for c in self.columns:
            tag = f"[{c.role.upper()}]"
            desc = c.description or c.name
            if c.valid_values:
                desc += f" (values: {', '.join(c.valid_values)})"
            lines.append(f"  {c.name:<35s}: {desc}  {tag}")
        lines.append("</COLUMN_DEFINITIONS>")
        return "\n".join(lines)

    def generate_trend_flag_logic(self) -> str:
        """Explain how trend_flag is computed for this signal."""
        return (
            f"trend_flag is derived from `{self.trend_column}`:\n"
            f"  RISING  = {self.trend_column} > {self.rising_threshold:+.0%}\n"
            f"  FALLING = {self.trend_column} < {self.falling_threshold:+.0%}\n"
            f"  STABLE  = between {self.falling_threshold:+.0%} and {self.rising_threshold:+.0%}"
        )

    def generate_yaml_schema(self) -> str:
        """Auto-generate output YAML schema from signal config."""
        dim_example = ", ".join(
            f'{d.name}: "{d.valid_values[0] if d.valid_values else "value"}"'
            for d in self.dimensions[:3]
        )
        return f"""\
trends:
  - rank: 1
    trend_title: "Short descriptive title"
    segment:
      {chr(10).join(f'      {d.name}: "value or All"' for d in self.dimensions)}
    signal_type: "{self.signal_id}"
    direction: "RISING | FALLING"
    metric_value: 0.0
    metric_name: "{self.trend_column}"
    benchmark_comparison: "X% above/below benchmark"
    confidence: "HIGH | MEDIUM | LOW"
    evidence_summary: "1-2 sentence grounded in data"
data_quality_notes: "Caveats"
"""

    def to_yaml(self) -> str:
        """Serialise config to YAML for persistence."""
        d = {
            "signal_id": self.signal_id,
            "signal_name": self.signal_name,
            "description": self.description,
            "trend_column": self.trend_column,
            "rising_threshold": self.rising_threshold,
            "falling_threshold": self.falling_threshold,
            "min_cohort_size": self.min_cohort_size,
            "cohort_size_column": self.cohort_size_column,
            "timestamp_column": self.timestamp_column,
            "columns": [
                {
                    "name": c.name, "role": c.role, "dtype": c.dtype,
                    "description": c.description, "is_primary": c.is_primary,
                    "valid_values": c.valid_values,
                }
                for c in self.columns
            ],
        }
        return yaml.dump(d, default_flow_style=False)


class SignalRegistry:
    """
    Catalog of all signal types. Add a signal once, use it everywhere.

    Usage:
        registry = SignalRegistry()
        registry.register(my_signal_config)
        config = registry.get("purchase_velocity")
    """

    def __init__(self):
        self._signals: Dict[str, SignalConfig] = {}

    def register(self, config: SignalConfig):
        self._signals[config.signal_id] = config

    def get(self, signal_id: str) -> SignalConfig:
        if signal_id not in self._signals:
            raise KeyError(
                f"Signal '{signal_id}' not registered. "
                f"Available: {list(self._signals.keys())}"
            )
        return self._signals[signal_id]

    def list_signals(self) -> List[str]:
        return list(self._signals.keys())

    def auto_register_from_dataframe(
        self,
        signal_id: str,
        signal_name: str,
        df: pd.DataFrame,
        trend_column: str,
        timestamp_column: str = "week_start",
        cohort_size_column: str = "active_users",
        primary_dimensions: Optional[List[str]] = None,
        description: str = "",
    ) -> SignalConfig:
        """
        Auto-discover schema from a DataFrame and register it.

        Infers column roles from dtype and naming conventions:
          - object/category → dimension
          - numeric → metric
          - datetime → timestamp
          - trend_column → trend
        """
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)

            if col == timestamp_column:
                role, cdtype = "timestamp", "datetime"
            elif col == trend_column:
                role, cdtype = "trend", "numeric"
            elif any(x in dtype for x in ("object", "category", "bool", "str", "string")):
                role, cdtype = "dimension", "categorical"
            elif "int" in dtype or "float" in dtype:
                role, cdtype = "metric", "numeric"
            elif "datetime" in dtype:
                role, cdtype = "timestamp", "datetime"
            else:
                role, cdtype = "meta", "categorical"

            is_primary = (
                primary_dimensions and col in primary_dimensions
            ) or (
                primary_dimensions is None and role == "dimension"
            )

            valid_vals = None
            if role == "dimension":
                uniq = df[col].dropna().unique()
                if len(uniq) <= 30:  # Only cache if cardinality is manageable
                    valid_vals = sorted(str(v) for v in uniq)

            columns.append(ColumnSpec(
                name=col, role=role, dtype=cdtype,
                description=f"Auto-detected {role} ({dtype})",
                valid_values=valid_vals,
                is_primary=is_primary,
            ))

        config = SignalConfig(
            signal_id=signal_id,
            signal_name=signal_name,
            description=description or f"Auto-registered signal: {signal_name}",
            columns=columns,
            trend_column=trend_column,
            timestamp_column=timestamp_column,
            cohort_size_column=cohort_size_column,
        )
        self.register(config)
        return config


# ═════════════════════════════════════════════════════════════════════
#  GENERIC CONTEXT ENGINE — Works with any SignalConfig
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TokenBudget:
    zone_a: int = 0; zone_b: int = 0; zone_c: int = 0
    total_limit: int = 120_000

    @property
    def total(self): return self.zone_a + self.zone_b + self.zone_c
    @property
    def cache_ratio(self): return (self.zone_a + self.zone_c) / max(1, self.total)
    def ok(self): return self.total <= self.total_limit
    def to_dict(self):
        return {"zone_a": self.zone_a, "zone_b": self.zone_b, "zone_c": self.zone_c,
                "total": self.total, "limit": self.total_limit,
                "cache_ratio": round(self.cache_ratio, 3), "ok": self.ok()}


def _est_tokens(text: str) -> int:
    return len(text) // 4


class GenericContextEngine:
    """
    Schema-agnostic prompt builder.

    Reads a SignalConfig and DataFrame, then generates a
    Lost-in-the-Middle optimised prompt with:
      Zone A: Role + auto-generated column defs + output schema  (cacheable)
      Zone B: Pre-conditioned data table                          (dynamic)
      Zone C: Task restatement + grounding rules                  (cacheable)
    """

    # ── Zone A: Stable Prefix Builder ───────────────────────────

    def build_zone_a(self, config: SignalConfig, task: str = "trend") -> str:
        """
        Generate the stable prefix from signal config.
        This is IDENTICAL across calls for the same signal_id,
        so it maximises KV-cache hit rate.
        """
        role_map = {
            "trend": "Senior Data Analyst specialising in behavioural trend detection",
            "recommendation": "CRM Campaign Strategist designing targeted campaigns",
            "anomaly": "Anomaly Detection Specialist identifying statistical outliers",
        }
        role = role_map.get(task, "Data Analyst")

        col_defs = config.generate_column_definitions()
        trend_logic = config.generate_trend_flag_logic()
        schema = config.generate_yaml_schema()

        return f"""\
<ROLE>
You are a {role}.
Analyse the signal "{config.signal_name}" and identify the TOP 5
most actionable trends from the data below.
Signal: {config.description}
</ROLE>

{col_defs}

<TREND_LOGIC>
{trend_logic}
</TREND_LOGIC>

<GROUNDING_RULES>
1. Every insight MUST reference specific dimension values from the data.
2. Every numeric claim MUST trace to a value in the table.
3. Do NOT infer trends absent from the data.
4. Weak signals (|{config.trend_column}| < {abs(config.rising_threshold) / 2:.0%}) are NOT top trends.
5. Flag confidence: HIGH (large cohort + strong signal), MEDIUM, LOW.
6. Return YAML only. No fences. No commentary.
</GROUNDING_RULES>

<OUTPUT_FORMAT>
{schema}
</OUTPUT_FORMAT>"""

    # ── Zone B: Dynamic Data Builder ────────────────────────────

    def build_zone_b(
        self,
        df: pd.DataFrame,
        config: SignalConfig,
        max_rows: Optional[int] = None,
    ) -> str:
        """
        Pre-condition and serialise data for Zone B.

        Uses confidence-weighted trimming:
          score = |trend_column| × log(cohort_size + 1)

        Column ordering follows Lost-in-the-Middle:
          TOP:    dimensions + trend_column + trend_flag
          MIDDLE: supporting metrics
          BOTTOM: derived ratios / benchmarks
        """
        max_rows = max_rows or config.max_context_rows
        df = df.copy()

        # Filter by minimum cohort size
        if config.cohort_size_column in df.columns:
            df = df[df[config.cohort_size_column] >= config.min_cohort_size]

        # Latest period only
        if config.timestamp_column in df.columns:
            latest = df[config.timestamp_column].max()
            df = df[df[config.timestamp_column] == latest]

        # Add trend_flag if not present
        if "trend_flag" not in df.columns and config.trend_column in df.columns:
            df["trend_flag"] = np.where(
                df[config.trend_column] > config.rising_threshold, "RISING",
                np.where(df[config.trend_column] < config.falling_threshold,
                         "FALLING", "STABLE")
            )

        # Confidence-weighted trimming
        if config.trend_column in df.columns and config.cohort_size_column in df.columns:
            df["_score"] = (
                df[config.trend_column].abs()
                * np.log1p(df[config.cohort_size_column])
            )
            group_cols = config.primary_dimensions or config.all_dimension_names
            # Use only columns that exist in df
            group_cols = [c for c in group_cols if c in df.columns]
            if group_cols:
                df = (
                    df.sort_values("_score", ascending=False)
                    .groupby(group_cols)
                    .head(config.top_n_per_group)
                )
            df = df.drop(columns=["_score"], errors="ignore")

        df = df.head(max_rows)

        # Column ordering: Lost-in-the-Middle
        dim_names = [c for c in config.all_dimension_names if c in df.columns]
        trend_cols = [config.trend_column, "trend_flag"]
        trend_cols = [c for c in trend_cols if c in df.columns]
        metric_names = [c for c in config.all_metric_names
                        if c in df.columns and c not in trend_cols]
        ts_cols = [config.timestamp_column] if config.timestamp_column in df.columns else []

        ordered = (
            dim_names           # TOP: dimensions (high attention)
            + trend_cols        # TOP: trend signal
            + ts_cols           # MIDDLE: timestamp
            + metric_names      # MIDDLE: supporting metrics (lower attention)
        )
        # Remove dupes while preserving order
        seen = set()
        ordered = [c for c in ordered if not (c in seen or seen.add(c))]

        # Add any remaining columns at the bottom
        remaining = [c for c in df.columns if c not in seen
                     and not c.startswith("_")]
        ordered += remaining

        serialised = df[ordered].to_markdown(index=False)

        # National/global summary (BOTTOM ZONE)
        if group_cols and config.trend_column in df.columns:
            summary_cols = [config.trend_column]
            if config.cohort_size_column in df.columns:
                summary_cols.append(config.cohort_size_column)
            agg_dict = {}
            for col in summary_cols:
                if col in df.columns:
                    agg_dict[col] = "mean"
            if group_cols and agg_dict:
                # Summarise by first primary dimension
                summary = df.groupby(group_cols[0]).agg(agg_dict).reset_index()
                summary_text = summary.round(4).to_markdown(index=False)
            else:
                summary_text = ""
        else:
            summary_text = ""

        latest_label = latest if config.timestamp_column in df.columns else "current"

        text = f"\n---\n## DATA — {config.signal_name} (Period: {latest_label})\n\n{serialised}"
        if summary_text:
            text += f"\n\n---\n### Benchmark Summary\n\n{summary_text}"
        text += "\n---"

        return text

    # ── Zone C: Stable Suffix Builder ───────────────────────────

    def build_zone_c(self, config: SignalConfig) -> str:
        """
        Task restatement + grounding rules.
        IDENTICAL across calls → cacheable.
        """
        dims = ", ".join(config.all_dimension_names[:4])
        return f"""\
<TASK>
Identify TOP 5 trends in "{config.signal_name}".
Prioritise:
  (a) Large cohort ({config.cohort_size_column} > {config.min_cohort_size * 5})
  (b) Strong signal (|{config.trend_column}| > {config.rising_threshold * 1.5:.0%})
  (c) Actionable for targeting by {dims}
Rank by impact = signal_strength × cohort_size.
Every claim must cite a data value. No hallucinations.
Return YAML only.
</TASK>"""

    # ── Full Prompt Assembly ────────────────────────────────────

    def build_prompt(
        self,
        df: pd.DataFrame,
        config: SignalConfig,
        task: str = "trend",
    ) -> Tuple[str, TokenBudget]:
        """
        Assemble full prompt: Zone A + Zone B + Zone C.

        Auto-compresses Zone B if token budget is exceeded.
        """
        budget = TokenBudget()

        zone_a = self.build_zone_a(config, task=task)
        budget.zone_a = _est_tokens(zone_a)

        zone_c = self.build_zone_c(config)
        budget.zone_c = _est_tokens(zone_c)

        # Build Zone B with auto-compression
        for max_rows in [config.max_context_rows, 80, 60, 40, 20]:
            zone_b = self.build_zone_b(df, config, max_rows=max_rows)
            budget.zone_b = _est_tokens(zone_b)
            if budget.ok():
                break

        prompt = f"{zone_a}\n{zone_b}\n{zone_c}"
        return prompt, budget


# ═════════════════════════════════════════════════════════════════════
#  GENERIC MEMORY — Playbook that works with any signal
# ═════════════════════════════════════════════════════════════════════

@dataclass
class GenericPlaybookEntry:
    """Signal-agnostic trend entry for inter-call memory."""
    trend_id:    str
    signal_id:   str
    title:       str
    segment:     Dict[str, str]      # dimension_name → value
    direction:   str
    metric_name: str
    metric_value: float
    benchmark:   str
    confidence:  str
    evidence:    str

    def to_block(self) -> str:
        seg_str = " / ".join(f"{k}={v}" for k, v in self.segment.items())
        return f"""[{self.trend_id}]
  Signal     : {self.signal_id}
  Title      : {self.title}
  Segment    : {seg_str}
  Direction  : {self.direction}
  Value      : {self.metric_name} = {self.metric_value}
  Benchmark  : {self.benchmark}
  Confidence : {self.confidence}
  Evidence   : {self.evidence}"""


def format_playbook(entries: List[GenericPlaybookEntry], notes: str = "") -> str:
    blocks = [e.to_block() for e in entries]
    if notes:
        blocks.append(f"\n[DATA-QUALITY-NOTES]\n  {notes}")
    return "\n\n".join(blocks)


def build_playbook_from_yaml(trend_yaml: dict, signal_id: str) -> List[GenericPlaybookEntry]:
    """Convert any trend YAML output into playbook entries."""
    if isinstance(trend_yaml, str):
        trend_yaml = yaml.safe_load(trend_yaml)
    entries = []
    for t in trend_yaml.get("trends", []):
        seg = t.get("segment", {})
        if isinstance(seg, str):
            seg = {"segment": seg}
        entries.append(GenericPlaybookEntry(
            trend_id=f"TREND-{t['rank']}",
            signal_id=signal_id,
            title=t.get("trend_title", ""),
            segment=seg,
            direction=t.get("direction", ""),
            metric_name=t.get("metric_name", ""),
            metric_value=t.get("metric_value", 0),
            benchmark=t.get("benchmark_comparison", ""),
            confidence=t.get("confidence", ""),
            evidence=t.get("evidence_summary", ""),
        ))
    return entries


# ═════════════════════════════════════════════════════════════════════
#  GENERIC VALIDATOR — Schema-driven, works with any signal
# ═════════════════════════════════════════════════════════════════════

class Severity(Enum):
    CRITICAL = "CRITICAL"; WARNING = "WARNING"; INFO = "INFO"

@dataclass
class Issue:
    layer: str; severity: Severity; path: str; msg: str
    expected: Any = None; actual: Any = None

@dataclass
class ValidationReport:
    is_valid: bool = True; checks: int = 0; passed: int = 0
    issues: List[Issue] = field(default_factory=list)

    def add_issue(self, i: Issue):
        self.issues.append(i)
        if i.severity == Severity.CRITICAL: self.is_valid = False

    def add_pass(self): self.passed += 1

    def summary(self) -> str:
        cr = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        wr = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        lines = ["=" * 65, "GROUNDING VALIDATION REPORT", "=" * 65,
                 f"  Status: {'PASSED' if self.is_valid else 'FAILED'}  |  Checks: {self.checks}  |  Passed: {self.passed}",
                 f"  Critical: {cr}  |  Warnings: {wr}", "-" * 65]
        for i in self.issues:
            ic = {"CRITICAL": "[!!]", "WARNING": "[! ]", "INFO": "[i ]"}[i.severity.value]
            lines.append(f"  {ic} {i.layer}: {i.path} — {i.msg}")
            if i.expected: lines.append(f"      Expected: {i.expected}  |  Actual: {i.actual}")
        if not self.issues: lines.append("  All checks passed.")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_yaml(self) -> str:
        return yaml.dump({"validation": {
            "status": "PASSED" if self.is_valid else "FAILED",
            "checks": self.checks, "passed": self.passed,
            "issues": [{"layer": i.layer, "severity": i.severity.value,
                         "path": i.path, "msg": i.msg} for i in self.issues]
        }}, default_flow_style=False)


class GenericValidator:
    """
    Schema-driven validator. Reads a SignalConfig to know
    what dimensions/values are valid. No hardcoded rules.

    Layers:
      L1: Structure — required fields present?
      L2: Citation  — TREND-IDs exist?
      L3: Segment   — dimension values in data?
      L4: Metric    — numeric claims match data?
      L5: Direction — RISING/FALLING consistent?
      L6: Entity    — no hallucinated dimension values?
    """

    def validate(
        self,
        output_yaml: dict,
        config: SignalConfig,
        source_df: pd.DataFrame,
        cited_trend_ids: Optional[Set[str]] = None,
    ) -> ValidationReport:

        rp = ValidationReport()
        trends = output_yaml.get("trends", [])

        if not trends:
            rp.add_issue(Issue("STRUCTURE", Severity.CRITICAL, "trends", "Empty"))
            return rp

        # Build valid value sets from DATA (not config, which may be stale)
        valid_sets = {}
        for c in config.dimensions:
            if c.name in source_df.columns:
                vals = set(str(v) for v in source_df[c.name].dropna().unique())
                vals.add("All")
                valid_sets[c.name] = vals

        # Valid trend IDs (if checking recommendations)
        if cited_trend_ids is None:
            cited_trend_ids = {f"TREND-{t.get('rank', i+1)}" for i, t in enumerate(trends)}

        for idx, t in enumerate(trends):
            p = f"trend[{idx}]"

            # L1: Structure
            rp.checks += 1
            required = {"rank", "trend_title", "direction", "metric_value",
                        "metric_name", "confidence", "evidence_summary"}
            missing = required - set(t.keys())
            if missing:
                rp.add_issue(Issue("STRUCTURE", Severity.CRITICAL, p, f"Missing: {missing}"))
            else:
                rp.add_pass()

            # L2: Direction validity
            rp.checks += 1
            d = t.get("direction", "")
            if d not in ("RISING", "FALLING", "STABLE"):
                rp.add_issue(Issue("STRUCTURE", Severity.WARNING, f"{p}.direction",
                                   f"Invalid: {d}", {"RISING","FALLING","STABLE"}, d))
            else:
                rp.add_pass()

            # L3: Segment grounding
            segment = t.get("segment", {})
            if isinstance(segment, str):
                segment = {"segment": segment}
            for dim_name, valid_vals in valid_sets.items():
                val = segment.get(dim_name, "All")
                if val == "All":
                    continue
                rp.checks += 1
                if str(val) not in valid_vals:
                    rp.add_issue(Issue("SEGMENT", Severity.CRITICAL,
                                       f"{p}.segment.{dim_name}",
                                       f"'{val}' not in data",
                                       f"{len(valid_vals)} valid values", val))
                else:
                    rp.add_pass()

            # L4: Metric name exists in config
            rp.checks += 1
            mn = t.get("metric_name", "")
            valid_metrics = {c.name for c in config.metrics} | {config.trend_column, "trend_flag"}
            if mn and mn not in valid_metrics and mn not in source_df.columns:
                rp.add_issue(Issue("METRIC", Severity.WARNING, f"{p}.metric_name",
                                   f"'{mn}' not a known metric", valid_metrics, mn))
            else:
                rp.add_pass()

            # L5: Confidence validity
            rp.checks += 1
            conf = t.get("confidence", "")
            if conf not in ("HIGH", "MEDIUM", "LOW"):
                rp.add_issue(Issue("STRUCTURE", Severity.WARNING, f"{p}.confidence",
                                   f"Invalid: {conf}"))
            else:
                rp.add_pass()

            # L6: Evidence should not be empty
            rp.checks += 1
            ev = t.get("evidence_summary", "")
            if len(ev) < 10:
                rp.add_issue(Issue("EVIDENCE", Severity.WARNING, f"{p}.evidence",
                                   "Evidence too short — likely not grounded"))
            else:
                rp.add_pass()

        return rp


# ═════════════════════════════════════════════════════════════════════
#  DATA QUALITY — Context Poisoning Prevention
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    rows_before: int = 0; rows_after: int = 0; issues: List[str] = field(default_factory=list)
    @property
    def rows_dropped(self): return self.rows_before - self.rows_after


def validate_dataframe(df: pd.DataFrame, config: SignalConfig) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Config-driven data cleaning. Uses the SignalConfig to know
    which columns are critical, what categoricals are valid, etc.
    """
    rp = DataQualityReport(rows_before=len(df))
    df = df.copy()

    # Drop NaN in trend column
    if config.trend_column in df.columns:
        n = df[config.trend_column].isna().sum()
        if n > 0:
            rp.issues.append(f"Dropped {n} NaN in {config.trend_column}")
            df = df.dropna(subset=[config.trend_column])

    # Drop NaN in cohort size column
    if config.cohort_size_column in df.columns:
        n = df[config.cohort_size_column].isna().sum()
        if n > 0:
            rp.issues.append(f"Dropped {n} NaN in {config.cohort_size_column}")
            df = df.dropna(subset=[config.cohort_size_column])
        neg = (df[config.cohort_size_column] < 0).sum()
        if neg > 0:
            rp.issues.append(f"Dropped {neg} negative {config.cohort_size_column}")
            df = df[df[config.cohort_size_column] >= 0]

    # Cap extreme outliers in trend column
    if config.trend_column in df.columns:
        cap = 5.0
        n = (df[config.trend_column].abs() > cap).sum()
        if n > 0:
            rp.issues.append(f"Capped {n} outliers in {config.trend_column} at ±{cap}")
            df[config.trend_column] = df[config.trend_column].clip(-cap, cap)

    # Validate categoricals
    for dim_name, valid_vals in config.all_valid_categoricals.items():
        if dim_name in df.columns:
            inv = ~df[dim_name].isin(valid_vals)
            n = inv.sum()
            if n > 0:
                rp.issues.append(f"Dropped {n} invalid {dim_name} values")
                df = df[~inv]

    df = df.drop_duplicates()
    rp.rows_after = len(df)
    return df, rp


# ═════════════════════════════════════════════════════════════════════
#  COMPACTION — Anchored State for Next Run
# ═════════════════════════════════════════════════════════════════════

def compact_state(signal_id: str, trend_yaml: dict, status: str) -> str:
    trends = trend_yaml.get("trends", [])
    return yaml.dump({"compacted_state": {
        "signal_id": signal_id,
        "generated_at": datetime.now().isoformat(),
        "trends": [
            {"id": f"TREND-{t['rank']}", "title": t.get("trend_title", ""),
             "direction": t.get("direction", ""), "value": t.get("metric_value", 0),
             "confidence": t.get("confidence", "")}
            for t in trends
        ],
        "validation": status,
        "notes": trend_yaml.get("data_quality_notes", ""),
    }}, default_flow_style=False)


# ═════════════════════════════════════════════════════════════════════
#  TOOLS — YAML Parser
# ═════════════════════════════════════════════════════════════════════

def parse_yaml_output(raw: str) -> dict:
    c = re.sub(r'^```ya?ml\s*', '', raw.strip(), flags=re.MULTILINE)
    c = re.sub(r'^```\s*$', '', c, flags=re.MULTILINE).strip()
    try: return yaml.safe_load(c)
    except: return {"error": "Parse failed", "raw": raw[:500]}


# ═════════════════════════════════════════════════════════════════════
#  DEMO — 5 Heterogeneous Signals with Different Schemas
# ═════════════════════════════════════════════════════════════════════

def create_demo_signals() -> Tuple[SignalRegistry, Dict[str, pd.DataFrame]]:
    """
    Create 5 signals with COMPLETELY DIFFERENT schemas.
    Demonstrates the framework handles any column structure.
    """
    registry = SignalRegistry()
    datasets = {}
    np.random.seed(42)
    weeks = pd.date_range(end=pd.Timestamp.today().normalize(), periods=4, freq="W-MON")

    # ── Signal 1: Purchase Velocity (age, product, city, gender) ──
    rows = []
    for w in weeks:
        for ag in ["Gen-Z", "Millennial", "Gen-X", "Boomer"]:
            for p in ["Product-X", "Product-Y", "Product-Z"]:
                for c in ["Bengaluru", "Mumbai", "Delhi"]:
                    au = np.random.randint(200, 2000)
                    vel = np.random.uniform(-0.5, 0.6)
                    if ag == "Gen-Z" and p == "Product-X": vel = abs(vel) + 0.2
                    rows.append({"age_group": ag, "product_category": p, "city": c,
                                 "week_start": w.strftime("%Y-%m-%d"),
                                 "active_users": au, "purchase_count": int(au * 0.08),
                                 "wow_velocity": round(vel, 4)})
    datasets["purchase_velocity"] = pd.DataFrame(rows)
    registry.auto_register_from_dataframe(
        "purchase_velocity", "Purchase Velocity", datasets["purchase_velocity"],
        trend_column="wow_velocity", primary_dimensions=["age_group", "product_category"])

    # ── Signal 2: Email Engagement (campaign_type, send_day, segment) ──
    rows = []
    for w in weeks:
        for ct in ["Promotional", "Transactional", "Newsletter", "Win-Back"]:
            for sd in ["Monday", "Wednesday", "Friday", "Sunday"]:
                for seg in ["High-Value", "Dormant", "New-User"]:
                    sent = np.random.randint(1000, 50000)
                    orc = np.random.uniform(-0.3, 0.4)
                    if ct == "Win-Back" and seg == "Dormant": orc += 0.25
                    rows.append({"campaign_type": ct, "send_day": sd, "user_segment": seg,
                                 "week_start": w.strftime("%Y-%m-%d"),
                                 "emails_sent": sent, "open_rate": round(np.random.uniform(0.1, 0.4), 4),
                                 "wow_open_rate_change": round(orc, 4)})
    datasets["email_engagement"] = pd.DataFrame(rows)
    registry.auto_register_from_dataframe(
        "email_engagement", "Email Open Rate Trend", datasets["email_engagement"],
        trend_column="wow_open_rate_change", cohort_size_column="emails_sent",
        primary_dimensions=["campaign_type", "user_segment"])

    # ── Signal 3: App Session Depth (os, feature, app_version) ──
    rows = []
    for w in weeks:
        for os_ in ["Android", "iOS"]:
            for feat in ["Home", "Search", "Cart", "Profile", "Wishlist"]:
                for ver in ["v8.1", "v8.2", "v9.0"]:
                    users = np.random.randint(500, 10000)
                    chg = np.random.uniform(-0.2, 0.3)
                    if feat == "Search" and ver == "v9.0": chg += 0.15
                    rows.append({"os": os_, "feature_used": feat, "app_version": ver,
                                 "week_start": w.strftime("%Y-%m-%d"),
                                 "active_users": users,
                                 "avg_session_depth": round(np.random.uniform(2, 12), 2),
                                 "wow_depth_change": round(chg, 4)})
    datasets["app_session_depth"] = pd.DataFrame(rows)
    registry.auto_register_from_dataframe(
        "app_session_depth", "App Session Depth Trend", datasets["app_session_depth"],
        trend_column="wow_depth_change", primary_dimensions=["feature_used", "os"])

    # ── Signal 4: Cart Abandonment (payment_method, cart_value, device) ──
    rows = []
    for w in weeks:
        for pm in ["UPI", "Credit Card", "Debit Card", "COD", "Wallet"]:
            for cv in ["<500", "500-2000", "2000-5000", ">5000"]:
                for dev in ["Mobile", "Desktop", "Tablet"]:
                    users = np.random.randint(100, 5000)
                    chg = np.random.uniform(-0.3, 0.2)
                    if pm == "COD" and cv == ">5000": chg -= 0.2
                    rows.append({"payment_method": pm, "cart_value_tier": cv, "device": dev,
                                 "week_start": w.strftime("%Y-%m-%d"),
                                 "active_users": users,
                                 "abandonment_rate": round(np.random.uniform(0.3, 0.8), 4),
                                 "wow_abandon_change": round(chg, 4)})
    datasets["cart_abandonment"] = pd.DataFrame(rows)
    registry.auto_register_from_dataframe(
        "cart_abandonment", "Cart Abandonment Rate Trend", datasets["cart_abandonment"],
        trend_column="wow_abandon_change", primary_dimensions=["payment_method", "cart_value_tier"])

    # ── Signal 5: Web Page Performance (page_category, traffic_source, region) ──
    rows = []
    for w in weeks:
        for pc in ["PDP", "PLP", "Homepage", "Checkout", "Blog"]:
            for ts in ["Organic", "Paid", "Social", "Direct", "Referral"]:
                for rg in ["North", "South", "East", "West"]:
                    views = np.random.randint(500, 50000)
                    chg = np.random.uniform(-0.25, 0.35)
                    if pc == "PDP" and ts == "Paid": chg += 0.15
                    rows.append({"page_category": pc, "traffic_source": ts, "region": rg,
                                 "week_start": w.strftime("%Y-%m-%d"),
                                 "page_views": views,
                                 "avg_time_on_page": round(np.random.uniform(10, 180), 1),
                                 "bounce_rate": round(np.random.uniform(0.2, 0.7), 4),
                                 "wow_views_change": round(chg, 4)})
    datasets["web_page_perf"] = pd.DataFrame(rows)
    registry.auto_register_from_dataframe(
        "web_page_perf", "Web Page Views Trend", datasets["web_page_perf"],
        trend_column="wow_views_change", cohort_size_column="page_views",
        primary_dimensions=["page_category", "traffic_source"])

    return registry, datasets


# ═════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — Run any signal through the generic pipeline
# ═════════════════════════════════════════════════════════════════════

def run_signal_pipeline(
    signal_id: str,
    df: pd.DataFrame,
    registry: SignalRegistry,
    simulated_output: Optional[dict] = None,
) -> dict:
    """
    Generic pipeline: works with ANY registered signal.

    Steps:
      1. Load config from registry
      2. Validate + clean data (config-driven)
      3. Build prompt (auto-generated zones)
      4. [Simulated] LLM call
      5. Validate output (config-driven)
      6. Compact state
    """
    config = registry.get(signal_id)
    engine = GenericContextEngine()
    validator = GenericValidator()

    t0 = time.time()
    print(f"\n{'─' * 65}")
    print(f"  Signal: {config.signal_name} ({signal_id})")
    print(f"  Dimensions: {config.all_dimension_names}")
    print(f"  Trend col:  {config.trend_column}")
    print(f"{'─' * 65}")

    # Step 1: Clean
    df_clean, dq = validate_dataframe(df, config)
    print(f"  [1] Data: {dq.rows_before} → {dq.rows_after} rows", end="")
    if dq.issues: print(f"  ({'; '.join(dq.issues)})")
    else: print()

    # Step 2: Build prompt
    prompt, budget = engine.build_prompt(df_clean, config)
    prefix_hash = hashlib.sha256(
        engine.build_zone_a(config).encode()
    ).hexdigest()[:12]
    print(f"  [2] Prompt: {budget.total:,} tokens  |  Cache: {budget.cache_ratio:.0%}  |  Prefix: {prefix_hash}")

    # Step 3: Simulated LLM output
    if simulated_output is None:
        simulated_output = _generate_dummy_output(df_clean, config)
    print(f"  [3] Trends: {len(simulated_output.get('trends', []))}")

    # Step 4: Validate
    report = validator.validate(simulated_output, config, df_clean)
    status = "PASSED" if report.is_valid else "FAILED"
    print(f"  [4] Validation: {status}  |  {report.checks} checks, {report.passed} passed")
    if report.issues:
        for i in report.issues[:3]:
            print(f"      [{i.severity.value}] {i.layer}: {i.msg}")

    # Step 5: Compact
    cs = compact_state(signal_id, simulated_output, status)

    latency = round((time.time() - t0) * 1000, 1)
    print(f"  [5] Compacted: {_est_tokens(cs)} tokens  |  Latency: {latency}ms")

    return {
        "signal_id": signal_id,
        "config": config,
        "prompt": prompt,
        "budget": budget,
        "output": simulated_output,
        "validation": report,
        "compacted": cs,
        "latency_ms": latency,
    }


def _generate_dummy_output(df: pd.DataFrame, config: SignalConfig) -> dict:
    """Generate a realistic dummy trend output from actual data."""
    if config.trend_column not in df.columns:
        return {"trends": [], "data_quality_notes": "No trend column found"}

    latest = df[config.timestamp_column].max() if config.timestamp_column in df.columns else None
    df_l = df[df[config.timestamp_column] == latest] if latest else df

    # Sort by absolute trend value
    df_sorted = df_l.reindex(
        df_l[config.trend_column].abs().sort_values(ascending=False).index
    ).head(5)

    trends = []
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        seg = {}
        for d in config.dimensions:
            if d.name in row.index:
                seg[d.name] = str(row[d.name])

        val = row[config.trend_column]
        direction = "RISING" if val > config.rising_threshold else (
            "FALLING" if val < config.falling_threshold else "STABLE")
        cohort = row.get(config.cohort_size_column, 0)
        conf = "HIGH" if abs(val) > 0.3 and cohort > 500 else (
            "MEDIUM" if abs(val) > 0.15 else "LOW")

        seg_desc = ", ".join(f"{k}={v}" for k, v in seg.items())
        trends.append({
            "rank": i + 1,
            "trend_title": f"{direction} {config.trend_column} for {seg_desc}",
            "segment": seg,
            "signal_type": config.signal_id,
            "direction": direction,
            "metric_value": round(float(val), 4),
            "metric_name": config.trend_column,
            "benchmark_comparison": f"{abs(val):.0%} {'above' if val > 0 else 'below'} baseline",
            "confidence": conf,
            "evidence_summary": f"{seg_desc} shows {config.trend_column}={val:.4f} with {cohort} users.",
        })

    return {
        "trends": trends,
        "data_quality_notes": f"Auto-generated from {len(df_l)} rows, latest period {latest}."
    }


# ═════════════════════════════════════════════════════════════════════
#  MAIN — Run all 5 heterogeneous signals through the generic pipeline
# ═════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("╔" + "═" * 63 + "╗")
    print("║  GENERIC CONTEXT ENGINEERING v4.0                             ║")
    print("║  Schema-Agnostic · Registry-Driven · Any Signal              ║")
    print("╚" + "═" * 63 + "╝")

    os.makedirs("outputs", exist_ok=True)

    # Create registry with 5 heterogeneous signals
    registry, datasets = create_demo_signals()

    print(f"\nRegistered signals: {registry.list_signals()}")
    print(f"Each has DIFFERENT columns — one framework handles all.\n")

    results = {}
    for sig_id in registry.list_signals():
        r = run_signal_pipeline(sig_id, datasets[sig_id], registry)
        results[sig_id] = r

        # Save artifacts
        with open(f"outputs/{sig_id}_prompt.txt", "w") as f:
            f.write(r["prompt"])
        with open(f"outputs/{sig_id}_output.yml", "w") as f:
            yaml.dump(r["output"], f, default_flow_style=False)
        with open(f"outputs/{sig_id}_validation.yml", "w") as f:
            f.write(r["validation"].to_yaml())
        with open(f"outputs/{sig_id}_config.yml", "w") as f:
            f.write(r["config"].to_yaml())

    # Summary
    print("\n" + "═" * 65)
    print("PIPELINE SUMMARY — All Signals")
    print("═" * 65)
    print(f"  {'Signal':<25s} {'Dims':<8s} {'Tokens':<10s} {'Cache':<8s} {'Valid':<8s} {'ms':<6s}")
    print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")
    for sid, r in results.items():
        ndims = len(r["config"].all_dimension_names)
        print(f"  {sid:<25s} {ndims:<8d} {r['budget'].total:<10,d} "
              f"{r['budget'].cache_ratio:<8.0%} "
              f"{'PASS' if r['validation'].is_valid else 'FAIL':<8s} "
              f"{r['latency_ms']:<6.0f}")

    print(f"\n  Total signals: {len(results)}")
    print(f"  All passed:    {all(r['validation'].is_valid for r in results.values())}")

    # Save registry
    with open("outputs/signal_registry.yml", "w") as f:
        for sid in registry.list_signals():
            f.write(f"---\n# {sid}\n")
            f.write(registry.get(sid).to_yaml())
            f.write("\n")

    return results


if __name__ == "__main__":
    run_pipeline()

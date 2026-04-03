"""
Context Engineering - Lean (Zone A/B/C + Inversion + Validation)
No token budgets. No cache tracking. No observability overhead.

  Zone A (top)    : Role + schema + rules        -> LLM sees first
  Zone B (middle) : Inverted DataFrame           -> data payload
  Zone C (bottom) : Task + example + guardrails  -> LLM sees last
"""

import pandas as pd, numpy as np, yaml, re, os, json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ─── SIGNAL CONFIG ───────────────────────────────────────────────

@dataclass
class SignalConfig:
    signal_name: str
    dimensions: List[str]
    trend_column: str
    cohort_size_column: str
    metrics: Optional[List[str]] = None
    timestamp_column: str = "week_start"
    rising_threshold: float = 0.10
    falling_threshold: float = -0.10
    min_cohort_size: int = 50
    top_n_cards: int = 5


# ─── HELPERS ─────────────────────────────────────────────────────

def _resolve_metrics(df, cfg):
    if cfg.metrics: return [c for c in cfg.metrics if c in df.columns]
    exclude = {cfg.trend_column, cfg.cohort_size_column, cfg.timestamp_column, "trend_flag", "_score"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

def _cohort_label(row, dims):
    return " x ".join(str(row.get(d, "?")) for d in dims if d in row.index)

def _fmt(val, name):
    if pd.isna(val): return "-"
    if isinstance(val, float):
        if any(x in name.lower() for x in ("rate","pct","velocity","change")): return f"{val:+.1%}" if abs(val)<10 else f"{val:+.2f}"
        if any(x in name.lower() for x in ("revenue","spend")):
            if abs(val)>=1e6: return f"INR {val/1e6:.1f}M"
            if abs(val)>=1e3: return f"INR {val/1e3:.1f}K"
            return f"INR {val:,.0f}"
        return f"{val:,.2f}" if abs(val)<100 else f"{val:,.0f}"
    return str(val)


# ─── PRECONDITION ────────────────────────────────────────────────

def precondition(df, cfg):
    df = df.copy()
    if cfg.timestamp_column in df.columns:
        df = df[df[cfg.timestamp_column] == df[cfg.timestamp_column].max()]
    if cfg.cohort_size_column in df.columns:
        df = df[df[cfg.cohort_size_column] >= cfg.min_cohort_size]
    if cfg.trend_column in df.columns:
        df = df.dropna(subset=[cfg.trend_column])
        df[cfg.trend_column] = df[cfg.trend_column].clip(-5.0, 5.0)
    if cfg.trend_column in df.columns:
        df["trend_flag"] = np.where(
            df[cfg.trend_column] > cfg.rising_threshold, "RISING",
            np.where(df[cfg.trend_column] < cfg.falling_threshold, "FALLING", "STABLE"))
    if cfg.trend_column in df.columns and cfg.cohort_size_column in df.columns:
        df["_score"] = df[cfg.trend_column].abs() * np.log1p(df[cfg.cohort_size_column])
        df = df.sort_values("_score", ascending=False).drop(columns=["_score"])
    return df


# ─── ZONE B: DATAFRAME INVERSION (HYBRID) ───────────────────────

def invert_dataframe(df, cfg):
    df = precondition(df, cfg)
    metrics = _resolve_metrics(df, cfg)
    df_top = df.head(cfg.top_n_cards)
    df_rest = df.iloc[cfg.top_n_cards:].head(30)
    sections = []

    # TOP: Signal Cards
    sections.append("### Top signals (detailed)\n")
    for i, (_, row) in enumerate(df_top.iterrows()):
        cohort = _cohort_label(row, cfg.dimensions)
        tv = row.get(cfg.trend_column, 0)
        flag = row.get("trend_flag", "STABLE")
        arrow = {"RISING": "^", "FALLING": "v"}.get(flag, "-")
        cs = row.get(cfg.cohort_size_column, 0)
        conf = "HIGH" if abs(tv) > 0.3 and cs > 500 else ("MEDIUM" if abs(tv) > 0.15 or cs > 200 else "LOW")
        lines = [f"[{i+1}] {arrow} {flag} -- {cohort}"]
        lines.append(f"  {cfg.trend_column:<24s}: {_fmt(tv, cfg.trend_column)}")
        lines.append(f"  {cfg.cohort_size_column:<24s}: {int(cs):,}")
        for m in metrics:
            if m in row.index and pd.notna(row[m]):
                lines.append(f"  {m:<24s}: {_fmt(row[m], m)}")
        lines.append(f"  {'confidence':<24s}: {conf}")
        sections.append("\n".join(lines))
        sections.append("")

    # MIDDLE: Compact Table
    if len(df_rest) > 0:
        sections.append("### Supporting signals (compact)\n")
        df_c = df_rest.copy()
        df_c["cohort"] = df_c.apply(lambda r: _cohort_label(r, cfg.dimensions), axis=1)
        compact_m = metrics[:2] if len(metrics) > 2 else metrics
        cols = ["cohort", cfg.trend_column, "trend_flag", cfg.cohort_size_column] + compact_m
        cols = list(dict.fromkeys(c for c in cols if c in df_c.columns))
        sections.append(df_c[cols].to_markdown(index=False))
        sections.append("")

    # BOTTOM: Benchmark
    if cfg.dimensions and cfg.trend_column in df.columns:
        primary = cfg.dimensions[0]
        if primary in df.columns:
            agg = {cfg.trend_column: "mean", cfg.cohort_size_column: "sum"}
            for m in metrics[:2]:
                if m in df.columns: agg[m] = "mean"
            agg = {k: v for k, v in agg.items() if k in df.columns}
            if agg:
                bench = df.groupby(primary).agg(agg).round(4).reset_index()
                sections.append(f"### National benchmark (per {primary})\n")
                sections.append(bench.to_markdown(index=False))

    return "\n".join(sections)


# ─── ZONE A: SYSTEM PROMPT ──────────────────────────────────────

def build_zone_a(cfg):
    dims = ", ".join(cfg.dimensions)
    trend_logic = (f"RISING = {cfg.trend_column} > {cfg.rising_threshold:+.0%}, "
                   f"FALLING = {cfg.trend_column} < {cfg.falling_threshold:+.0%}")
    dim_schema = "\n".join(f'      {d}: "value or All"' for d in cfg.dimensions)
    return f"""\
<ROLE>
You are a Senior CRM Analytics Strategist. Identify the TOP 5 most
actionable trends from the "{cfg.signal_name}" data below.
</ROLE>

<SCHEMA>
Dimensions  : {dims}
Trend       : {cfg.trend_column} ({trend_logic})
Cohort size : {cfg.cohort_size_column}
</SCHEMA>

<GROUNDING_RULES>
1. Every insight MUST reference specific dimension values from the data.
2. Every numeric claim MUST match a value in the data below.
3. Do NOT infer trends absent from the data.
4. Flag confidence: HIGH / MEDIUM / LOW.
5. Return YAML only. No markdown fences. No commentary outside YAML.
</GROUNDING_RULES>

<OUTPUT_FORMAT>
Return EXACTLY this YAML structure (no extra fields, no missing fields):

trends:
  - rank: 1
    trend_title: "descriptive title"
    segment:
{dim_schema}
    direction: "RISING | FALLING"
    metric_value: 0.0
    metric_name: "{cfg.trend_column}"
    benchmark_comparison: "X% above/below"
    confidence: "HIGH | MEDIUM | LOW"
    evidence_summary: "grounded explanation citing data"
  - rank: 2
    trend_title: "..."
    ...
data_quality_notes: "caveats"
</OUTPUT_FORMAT>

<WRONG_EXAMPLES>
Do NOT return any of these — they are common mistakes:

  WRONG: Adding fields not in the schema
    recommendation: "Run a campaign"        ← not in schema, do not add

  WRONG: Flat string instead of dict for segment
    segment: "Gen-Z x Product-X"           ← WRONG
    segment:                                ← RIGHT
      age_group: "Gen-Z"
      product_category: "Product-X"

  WRONG: Unquoted colons in string values (causes YAML parse failure)
    evidence_summary: Gen-Z: velocity +54%  ← BREAKS YAML
    evidence_summary: "Gen-Z: velocity +54%" ← CORRECT

  WRONG: metric_value as string
    metric_value: "0.54"                    ← WRONG (string)
    metric_value: 0.54                      ← CORRECT (number)

  WRONG: Markdown fences around output
    ```yaml                                 ← do not include
    trends: ...
    ```                                     ← do not include

  WRONG: Commentary before or after YAML
    Here are the trends:                    ← do not include
    trends: ...
    These trends show...                    ← do not include
</WRONG_EXAMPLES>"""


# ─── ZONE C: TASK RESTATEMENT ────────────────────────────────────

def build_zone_c(cfg):
    dims = ", ".join(cfg.dimensions[:3])
    d0 = cfg.dimensions[0] if cfg.dimensions else "segment"
    d1 = cfg.dimensions[1] if len(cfg.dimensions) > 1 else d0
    return f"""\
<FILLED_EXAMPLE>
Below is an example of CORRECT output with 2 entries.
Your output must follow this exact structure for all 5 trends.

trends:
  - rank: 1
    trend_title: "RISING {cfg.trend_column} for {d0}=GroupA x {d1}=ItemX"
    segment:
      {d0}: "GroupA"
      {d1}: "ItemX"
    direction: "RISING"
    metric_value: 0.35
    metric_name: "{cfg.trend_column}"
    benchmark_comparison: "35% above previous week"
    confidence: "HIGH"
    evidence_summary: "GroupA x ItemX shows {cfg.trend_column}=+0.35 with 2,500 {cfg.cohort_size_column}."
  - rank: 2
    trend_title: "FALLING {cfg.trend_column} for {d0}=GroupB x {d1}=ItemY"
    segment:
      {d0}: "GroupB"
      {d1}: "ItemY"
    direction: "FALLING"
    metric_value: -0.28
    metric_name: "{cfg.trend_column}"
    benchmark_comparison: "28% below previous week"
    confidence: "HIGH"
    evidence_summary: "GroupB x ItemY shows {cfg.trend_column}=-0.28 with 1,800 {cfg.cohort_size_column}."
data_quality_notes: "Small cohorts (<100 users) excluded."
</FILLED_EXAMPLE>

<TASK>
Now analyse the ACTUAL data above and return TOP 5 trends.
Prioritise:
  (a) Large cohort ({cfg.cohort_size_column} > {cfg.min_cohort_size * 5})
  (b) Strong signal (|{cfg.trend_column}| > {cfg.rising_threshold * 1.5:.0%})
  (c) Actionable for targeting by {dims}
Rank by impact = signal_strength x cohort_size.
Every claim must cite a specific value from the data.

CRITICAL REMINDERS:
  - Return YAML only, no fences, no commentary
  - Quote all strings containing colons or special characters
  - metric_value must be a number, not a string
  - segment must be a dict, not a flat string
  - Exactly 5 trends, ranked 1-5
</TASK>"""


# ─── FULL PROMPT BUILDER ────────────────────────────────────────

def build_prompt(df, cfg):
    """One function. Any DataFrame. Any config. Complete prompt."""
    return f"""{build_zone_a(cfg)}

---
## {cfg.signal_name} -- Aggregated Data

{invert_dataframe(df, cfg)}

---

{build_zone_c(cfg)}"""


# ─── GROUNDING VALIDATOR ────────────────────────────────────────

class Severity(Enum):
    CRITICAL = "CRITICAL"; WARNING = "WARNING"

@dataclass
class Issue:
    layer: str; severity: Severity; path: str; msg: str

@dataclass
class ValidationReport:
    is_valid: bool = True; checks: int = 0; passed: int = 0
    issues: List[Issue] = field(default_factory=list)
    def fail(self, layer, path, msg):
        self.issues.append(Issue(layer, Severity.CRITICAL, path, msg)); self.is_valid = False
    def warn(self, layer, path, msg):
        self.issues.append(Issue(layer, Severity.WARNING, path, msg))
    def ok(self): self.passed += 1
    def summary(self):
        cr = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        wr = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        lines = ["=" * 60,
                 f"  VALIDATION: {'PASSED' if self.is_valid else 'FAILED'}  "
                 f"| Checks: {self.checks} | Passed: {self.passed} | Critical: {cr} | Warnings: {wr}",
                 "-" * 60]
        for i in self.issues:
            ic = "[!!]" if i.severity == Severity.CRITICAL else "[! ]"
            lines.append(f"  {ic} {i.layer}: {i.path} -- {i.msg}")
        if not self.issues: lines.append("  All checks passed.")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── ROBUST YAML PARSER (6 repair layers) ────────────────────
#
# LLMs generate broken YAML in predictable ways. This parser
# handles each failure mode with a specific repair step.
#
# Common LLM YAML failures:
#   1. Markdown fences:     ```yaml ... ```
#   2. Preamble text:       "Here are the trends:\n" before YAML
#   3. Unquoted colons:     evidence: Gen-Z: velocity +54%
#   4. Inconsistent indent: mixing 2-space and 4-space
#   5. Special characters:  ×, —, ₹, (), % inside values
#   6. Trailing commas:     values: [a, b, c,]
#

@dataclass
class ParseResult:
    """Result of YAML parsing attempt."""
    success: bool
    data: Optional[dict] = None
    raw: str = ""
    repair_steps: List[str] = field(default_factory=list)
    error: str = ""


def parse_llm_yaml(raw_text: str) -> ParseResult:
    """
    Multi-layer YAML parser for LLM output.

    Tries progressively aggressive repairs:
      Layer 1: Strip markdown fences + preamble
      Layer 2: Fix unquoted colons in values
      Layer 3: Normalise indentation
      Layer 4: Quote all string values
      Layer 5: Extract YAML from mixed content
      Layer 6: Fall back to JSON extraction

    Usage:
        result = parse_llm_yaml(raw_llm_response)
        if result.success:
            trends = result.data  # dict
        else:
            print(result.error)
    """
    result = ParseResult(success=False, raw=raw_text)

    if not raw_text or not raw_text.strip():
        result.error = "Empty input"
        return result

    # ── Layer 1: Strip fences + preamble ───────────────────────
    cleaned = _strip_fences_and_preamble(raw_text)
    result.repair_steps.append("L1:strip_fences")

    parsed = _try_yaml_parse(cleaned)
    if parsed is not None:
        result.success = True
        result.data = parsed
        return result

    # ── Layer 2: Fix unquoted colons in values ─────────────────
    # The #1 cause of "scanning a simple key" error.
    # "evidence: Gen-Z: velocity +54%" → 'evidence: "Gen-Z: velocity +54%"'
    fixed = _fix_unquoted_colons(cleaned)
    result.repair_steps.append("L2:fix_colons")

    parsed = _try_yaml_parse(fixed)
    if parsed is not None:
        result.success = True
        result.data = parsed
        return result

    # ── Layer 3: Normalise indentation to 2-space ──────────────
    normalised = _normalise_indentation(fixed)
    result.repair_steps.append("L3:normalise_indent")

    parsed = _try_yaml_parse(normalised)
    if parsed is not None:
        result.success = True
        result.data = parsed
        return result

    # ── Layer 4: Force-quote all string values ─────────────────
    quoted = _force_quote_values(normalised)
    result.repair_steps.append("L4:force_quote")

    parsed = _try_yaml_parse(quoted)
    if parsed is not None:
        result.success = True
        result.data = parsed
        return result

    # ── Layer 5: Extract YAML block from mixed content ─────────
    extracted = _extract_yaml_block(raw_text)
    result.repair_steps.append("L5:extract_block")

    if extracted:
        parsed = _try_yaml_parse(extracted)
        if parsed is not None:
            result.success = True
            result.data = parsed
            return result

        # Try layers 2-4 on extracted block too
        fixed_ex = _force_quote_values(_normalise_indentation(_fix_unquoted_colons(extracted)))
        parsed = _try_yaml_parse(fixed_ex)
        if parsed is not None:
            result.success = True
            result.data = parsed
            result.repair_steps.append("L5:extract+repair")
            return result

    # ── Layer 6: JSON fallback ─────────────────────────────────
    # Some LLMs return JSON even when asked for YAML
    json_data = _try_json_parse(raw_text)
    result.repair_steps.append("L6:json_fallback")

    if json_data is not None:
        result.success = True
        result.data = json_data
        return result

    # ── Layer 7: Regex extraction (last resort) ────────────────
    regex_data = _regex_extract_trends(raw_text)
    result.repair_steps.append("L7:regex_extract")

    if regex_data:
        result.success = True
        result.data = regex_data
        return result

    result.error = (
        f"All 7 parse layers failed. "
        f"Repairs attempted: {', '.join(result.repair_steps)}. "
        f"First 200 chars: {raw_text[:200]}"
    )
    return result


# ── Internal repair functions ──────────────────────────────────

def _try_yaml_parse(text):
    """Attempt yaml.safe_load, return None on failure."""
    try:
        result = yaml.safe_load(text)
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        return None


def _try_json_parse(text):
    """Extract and parse JSON from text."""
    # Strip fences
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    # Try to find JSON object
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _strip_fences_and_preamble(text):
    """Remove ```yaml fences and any text before the first YAML key."""
    # Remove fences
    text = re.sub(r'^```ya?ml\s*\n?', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Remove preamble (text before first YAML key)
    # Look for first line that looks like a YAML key (word followed by colon)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (stripped.startswith('- ') or
                        re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:', stripped)):
            return '\n'.join(lines[i:])

    return text


def _fix_unquoted_colons(text):
    """
    Fix unquoted colons inside YAML values.

    The pattern: after 'key: ', if the value contains another colon
    that's NOT part of a URL (http:// or https://), wrap the value in quotes.

    Before: evidence_summary: Gen-Z: velocity +54% in Bengaluru
    After:  evidence_summary: "Gen-Z: velocity +54% in Bengaluru"
    """
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.lstrip()

        # Skip comments, list items starting with -, empty lines
        if not stripped or stripped.startswith('#'):
            fixed_lines.append(line)
            continue

        # Match key: value pattern
        match = re.match(r'^(\s*)(- )?([\w_]+)\s*:\s*(.+)$', line)
        if match:
            indent = match.group(1)
            list_prefix = match.group(2) or ""
            key = match.group(3)
            value = match.group(4).strip()

            # Check if value already quoted
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                fixed_lines.append(line)
                continue

            # Check if value is a YAML structure (list, nested key)
            if value.startswith('[') or value.startswith('{') or value == '':
                fixed_lines.append(line)
                continue

            # Check if value is a number or boolean
            if re.match(r'^-?\d+(\.\d+)?$', value) or value.lower() in ('true', 'false', 'null', 'none'):
                fixed_lines.append(line)
                continue

            # Check if value contains a problematic colon (not in URL)
            has_problem = False
            if ':' in value and not re.search(r'https?://', value):
                has_problem = True
            # Check for other YAML-special characters
            if any(ch in value for ch in ['#', '&', '*', '!', '|', '>', '{', '}', '[', ']', '%']):
                has_problem = True
            # Check for leading/trailing special chars
            if value and value[0] in ('"', "'", '@', '`', ','):
                has_problem = True

            if has_problem:
                # Escape any existing double quotes inside the value
                value_escaped = value.replace('"', '\\"')
                fixed_lines.append(f'{indent}{list_prefix}{key}: "{value_escaped}"')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def _normalise_indentation(text):
    """
    Normalise mixed indentation to consistent 2-space indent.

    LLMs sometimes mix 2-space and 4-space, or use tabs.
    YAML is strict about consistent indentation within a block.
    """
    lines = text.split('\n')
    fixed = []

    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue

        # Replace tabs with 2 spaces
        line = line.replace('\t', '  ')

        # Count leading spaces
        stripped = line.lstrip()
        n_spaces = len(line) - len(stripped)

        # Normalise: if indent is odd, round down to even
        if n_spaces % 2 != 0:
            n_spaces = n_spaces - 1

        # Cap excessive indentation (>8 is almost certainly wrong)
        if n_spaces > 8:
            n_spaces = 8

        fixed.append(' ' * n_spaces + stripped)

    return '\n'.join(fixed)


def _force_quote_values(text):
    """
    Nuclear option: wrap ALL string values in double quotes.
    Preserves numbers, booleans, lists, and nested keys.
    """
    lines = text.split('\n')
    fixed = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            fixed.append(line)
            continue

        match = re.match(r'^(\s*)(- )?([\w_]+)\s*:\s*(.+)$', line)
        if match:
            indent = match.group(1)
            list_prefix = match.group(2) or ""
            key = match.group(3)
            value = match.group(4).strip()

            # Already quoted
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                fixed.append(line)
                continue

            # YAML structures — leave alone
            if value.startswith('[') or value.startswith('{') or value == '':
                fixed.append(line)
                continue

            # Pure numbers — leave alone
            if re.match(r'^-?\d+(\.\d+)?$', value):
                fixed.append(line)
                continue

            # Booleans/null — leave alone
            if value.lower() in ('true', 'false', 'null', 'none'):
                fixed.append(line)
                continue

            # Everything else: force quote
            value_escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            fixed.append(f'{indent}{list_prefix}{key}: "{value_escaped}"')
        else:
            fixed.append(line)

    return '\n'.join(fixed)


def _extract_yaml_block(text):
    """
    Extract the YAML portion from mixed content.
    Looks for 'trends:' as the anchor point.
    """
    # Find 'trends:' line
    match = re.search(r'^(trends\s*:.*$)', text, re.MULTILINE)
    if not match:
        return None

    start = match.start()
    # Take everything from 'trends:' to end, or to next markdown fence
    rest = text[start:]
    fence_match = re.search(r'^```', rest, re.MULTILINE)
    if fence_match:
        rest = rest[:fence_match.start()]

    return rest.strip()


def _regex_extract_trends(text):
    """
    Last resort: extract trend data using regex patterns.
    Builds a dict from whatever structure it can find.
    """
    trends = []
    # Look for rank patterns
    blocks = re.split(r'(?:^|\n)\s*-?\s*rank\s*:', text)

    for i, block in enumerate(blocks[1:], 1):  # Skip first empty split
        trend = {"rank": i}
        # Extract key-value pairs
        for match in re.finditer(r'([\w_]+)\s*:\s*"?([^"\n]+)"?', block):
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key == "rank":
                continue
            # Try to convert numbers
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except (ValueError, TypeError):
                pass
            trend[key] = value

        if len(trend) > 2:  # More than just rank
            trends.append(trend)

        if len(trends) >= 5:
            break

    if trends:
        return {"trends": trends, "data_quality_notes": "Recovered via regex extraction."}
    return None


def validate_output(output_yaml, df, cfg):
    rp = ValidationReport()
    trends = output_yaml.get("trends", [])
    if not trends: rp.fail("STRUCTURE", "trends", "Empty"); return rp
    valid_sets = {}
    for dim in cfg.dimensions:
        if dim in df.columns:
            valid_sets[dim] = set(str(v) for v in df[dim].dropna().unique()) | {"All"}
    for idx, t in enumerate(trends):
        p = f"trend[{idx}]"
        rp.checks += 1
        missing = {"rank","trend_title","direction","metric_value","metric_name","confidence","evidence_summary"} - set(t.keys())
        if missing: rp.fail("STRUCTURE", p, f"Missing: {missing}")
        else: rp.ok()
        seg = t.get("segment", {})
        if isinstance(seg, str): seg = {"segment": seg}
        for dim_name, valid_vals in valid_sets.items():
            val = seg.get(dim_name, "All")
            if val == "All": continue
            rp.checks += 1
            if str(val) not in valid_vals: rp.fail("SEGMENT", f"{p}.{dim_name}", f"'{val}' not in data")
            else: rp.ok()
        rp.checks += 1
        if t.get("direction","") not in ("RISING","FALLING","STABLE"): rp.warn("DIRECTION", f"{p}.direction", f"Invalid: {t.get('direction','')}")
        else: rp.ok()
        rp.checks += 1
        if len(str(t.get("evidence_summary",""))) < 10: rp.warn("EVIDENCE", f"{p}.evidence", "Too short")
        else: rp.ok()
    return rp


# ─── CONVENIENCE ─────────────────────────────────────────────────

def run_signal(df, cfg, llm_response=None, simulated_output=None):
    """
    Full pipeline for any signal.

    Args:
        df:  Your aggregated DataFrame
        cfg: SignalConfig for this signal
        llm_response:     Raw string from LLM (will be parsed)
        simulated_output: Pre-parsed dict (skips parsing)

    Returns dict with: prompt, output, validation, parse_result
    """
    prompt = build_prompt(df, cfg)

    parse_result = None

    if llm_response is not None:
        # Parse raw LLM response
        parse_result = parse_llm_yaml(llm_response)
        if parse_result.success:
            output = parse_result.data
        else:
            return {
                "prompt": prompt,
                "output": None,
                "validation": None,
                "parse_result": parse_result,
                "error": parse_result.error,
            }
    elif simulated_output is not None:
        output = simulated_output
    else:
        output = _auto_output(df, cfg)

    report = validate_output(output, df, cfg)
    return {
        "prompt": prompt,
        "output": output,
        "validation": report,
        "parse_result": parse_result,
    }

def _auto_output(df, cfg):
    df = precondition(df, cfg)
    metrics = _resolve_metrics(df, cfg)
    trends = []
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        seg = {d: str(row.get(d, "All")) for d in cfg.dimensions if d in row.index}
        tv = row.get(cfg.trend_column, 0); cs = row.get(cfg.cohort_size_column, 0)
        flag = "RISING" if tv > cfg.rising_threshold else ("FALLING" if tv < cfg.falling_threshold else "STABLE")
        conf = "HIGH" if abs(tv)>0.3 and cs>500 else "MEDIUM"
        sd = ", ".join(f"{k}={v}" for k,v in seg.items())
        trends.append({"rank": i+1, "trend_title": f"{flag} {cfg.trend_column} for {sd}",
            "segment": seg, "direction": flag, "metric_value": round(float(tv),4),
            "metric_name": cfg.trend_column,
            "benchmark_comparison": f"{abs(tv):.0%} {'above' if tv>0 else 'below'} baseline",
            "confidence": conf,
            "evidence_summary": f"{sd} shows {cfg.trend_column}={tv:.4f} with {int(cs):,} {cfg.cohort_size_column}."})
    return {"trends": trends, "data_quality_notes": f"From {len(df)} rows."}


# ─── DEMO ────────────────────────────────────────────────────────

def demo():
    np.random.seed(42)
    print("=" * 60)
    print("  Context Engineering -- Lean (Zone A/B/C)")
    print("=" * 60)
    os.makedirs("outputs", exist_ok=True)

    # 3 signals with DIFFERENT schemas
    signals = []

    # Signal 1: Purchase Velocity
    r1 = [{"age_group": ag, "product_category": p, "city": c, "week_start": "2026-03-30",
           "active_users": np.random.randint(200,3000),
           "purchase_rate": round(np.random.uniform(0.02,0.12),4),
           "total_revenue": round(np.random.uniform(5000,500000),2),
           "wow_purchase_velocity_pct": round(
               np.random.uniform(-0.5,0.6) + (0.3 if ag=="Gen-Z" and p=="Product-X" else 0)
               + (-0.3 if ag=="Millennial" and p=="Product-Y" else 0), 4)}
          for ag in ["Gen-Z","Millennial","Gen-X","Boomer"]
          for p in ["Product-X","Product-Y","Product-Z"]
          for c in ["Bengaluru","Mumbai","Delhi"]]
    signals.append(("purchase_velocity", pd.DataFrame(r1),
        SignalConfig("Purchase Velocity", ["age_group","product_category","city"],
                     "wow_purchase_velocity_pct", "active_users",
                     ["purchase_rate","total_revenue"])))

    # Signal 2: Email Engagement
    r2 = [{"campaign_type": ct, "user_segment": seg, "send_day": sd, "week_start": "2026-03-30",
           "emails_sent": np.random.randint(1000,50000),
           "open_rate": round(np.random.uniform(0.1,0.4),4),
           "wow_open_rate_change": round(
               np.random.uniform(-0.3,0.4) + (0.25 if ct=="Win-Back" and seg=="Dormant" else 0), 4)}
          for ct in ["Promotional","Win-Back","Newsletter"]
          for seg in ["High-Value","Dormant","New-User"]
          for sd in ["Monday","Wednesday","Friday"]]
    signals.append(("email_engagement", pd.DataFrame(r2),
        SignalConfig("Email Open Rate", ["campaign_type","user_segment","send_day"],
                     "wow_open_rate_change", "emails_sent", ["open_rate"])))

    # Signal 3: App Sessions
    r3 = [{"os": os_, "feature": feat, "app_version": ver, "week_start": "2026-03-30",
           "active_users": np.random.randint(500,10000),
           "avg_depth": round(np.random.uniform(2,12),2),
           "wow_depth_change": round(
               np.random.uniform(-0.2,0.3) + (0.15 if feat=="Search" and ver=="v9.0" else 0), 4)}
          for os_ in ["Android","iOS"]
          for feat in ["Home","Search","Cart","Profile"]
          for ver in ["v8.1","v9.0"]]
    signals.append(("app_sessions", pd.DataFrame(r3),
        SignalConfig("App Session Depth", ["os","feature","app_version"],
                     "wow_depth_change", "active_users", ["avg_depth"])))

    for name, df, cfg in signals:
        print(f"\n--- {cfg.signal_name} ({name}) ---")
        print(f"  Schema : {cfg.dimensions} + {cfg.trend_column}")
        print(f"  Rows   : {len(df)}")

        result = run_signal(df, cfg)
        with open(f"outputs/{name}_prompt.txt", "w") as f: f.write(result["prompt"])
        with open(f"outputs/{name}_output.yml", "w") as f: yaml.dump(result["output"], f, default_flow_style=False)

        chars = len(result["prompt"])
        print(f"  Prompt : {chars:,} chars (~{chars//4:,} tokens)")
        print(f"  Trends : {len(result['output']['trends'])}")
        print(result["validation"].summary())

    # Show what the inverted data looks like
    print("\n" + "=" * 60)
    print("  INVERTED DATA (Purchase Velocity, first 25 lines)")
    print("=" * 60)
    inv = invert_dataframe(pd.DataFrame(r1), signals[0][2])
    for line in inv.split("\n")[:25]:
        print(f"  {line}")

    # ── PARSER STRESS TESTS ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  YAML PARSER STRESS TESTS")
    print("=" * 60)

    test_cases = {
        "Clean YAML": """\
trends:
  - rank: 1
    trend_title: "Gen-Z surge on Product-X"
    direction: RISING
    metric_value: 0.54
    metric_name: wow_velocity
    confidence: HIGH
    evidence_summary: "Gen-Z shows +54% WoW velocity"
data_quality_notes: "None"
""",

        "Unquoted colons (common failure)": """\
trends:
  - rank: 1
    trend_title: Gen-Z: surge on Product-X
    direction: RISING
    metric_value: 0.54
    metric_name: wow_velocity
    confidence: HIGH
    evidence_summary: Gen-Z: velocity +54% in Bengaluru: top city
data_quality_notes: Small cohort warning: Boomers < 100 users
""",

        "Markdown fences + preamble": """\
Here are the top 5 trends I identified:

```yaml
trends:
  - rank: 1
    trend_title: "Gen-Z surge"
    direction: RISING
    metric_value: 0.54
    metric_name: wow_velocity
    confidence: HIGH
    evidence_summary: "Gen-Z shows +54%"
data_quality_notes: "None"
```
""",

        "Mixed indentation (2+4 space)": """\
trends:
  - rank: 1
    trend_title: "Gen-Z surge"
    direction: RISING
        metric_value: 0.54
    metric_name: wow_velocity
      confidence: HIGH
    evidence_summary: "Gen-Z +54%"
data_quality_notes: "None"
""",

        "Special chars (%, x, parentheses)": """\
trends:
  - rank: 1
    trend_title: Gen-Z x Product-X (+54% WoW)
    direction: RISING
    metric_value: 0.54
    metric_name: wow_velocity
    confidence: HIGH
    evidence_summary: 3,200 users (top 5%) in Bengaluru & Mumbai
data_quality_notes: Boomer cohort (n=83) excluded
""",

        "JSON instead of YAML": """\
```json
{
  "trends": [
    {
      "rank": 1,
      "trend_title": "Gen-Z surge",
      "direction": "RISING",
      "metric_value": 0.54,
      "metric_name": "wow_velocity",
      "confidence": "HIGH",
      "evidence_summary": "Gen-Z +54%"
    }
  ],
  "data_quality_notes": "None"
}
```
""",

        "Worst case (all problems)": """\
Based on my analysis of the data, here are the trends:

```yaml
trends:
  - rank: 1
    trend_title: Gen-Z: Product-X surge (+54% WoW)
    direction: RISING
        metric_value: 0.54
    metric_name: wow_velocity
    confidence: HIGH
    evidence_summary: Gen-Z x Product-X: velocity is +54% in Bengaluru & Mumbai (3,200 users)
  - rank: 2
    trend_title: Millennial: Product-Y decline
    direction: FALLING
    metric_value: -0.45
    metric_name: wow_velocity
    confidence: HIGH
    evidence_summary: Millennials show -45% WoW: churn risk detected
data_quality_notes: Boomer cohort (n<100): excluded from analysis
```

These trends show clear demographic patterns.
""",
    }

    for name, raw in test_cases.items():
        result = parse_llm_yaml(raw)
        status = "OK" if result.success else "FAIL"
        n_trends = len(result.data.get("trends", [])) if result.data else 0
        steps = " -> ".join(result.repair_steps[:3])
        print(f"\n  [{status}] {name}")
        print(f"       Trends: {n_trends}  |  Repairs: {steps}")
        if not result.success:
            print(f"       Error: {result.error[:80]}")


if __name__ == "__main__":
    demo()

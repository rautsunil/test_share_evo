"""
╔══════════════════════════════════════════════════════════════════════╗
║  Zone A + Zone C Schema Design                                       ║
║  CRM Micro-Signal Insights with Row-Level Citation Tracing           ║
║                                                                      ║
║  Output: JSON (not YAML)                                             ║
║                                                                      ║
║  Why JSON over YAML for this schema:                                 ║
║    1. Citations are nested arrays-of-objects — JSON handles this     ║
║       unambiguously with [] and {}. YAML requires perfect indent     ║
║       across 4 nesting levels and LLMs frequently get it wrong.      ║
║    2. json.loads() is deterministic — it either parses or doesn't.   ║
║       yaml.safe_load() can silently produce wrong nesting.           ║
║    3. metric_value must be a number. JSON enforces 0.54 ≠ "0.54".   ║
║       YAML treats them identically in many edge cases.               ║
║    4. Colons in insight text (your parse error) are not special      ║
║       in JSON strings — only in YAML.                                ║
║    5. Every downstream system (BigQuery, APIs, dashboards)           ║
║       natively consumes JSON. YAML needs conversion.                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═════════════════════════════════════════════════════════════════════
#  ZONE A — System Prompt (top of context window, high LLM attention)
# ═════════════════════════════════════════════════════════════════════
#
#  Contains:
#    1. Role definition
#    2. Column definitions from the input DataFrame
#    3. Output schema (JSON)
#    4. Grounding rules
#    5. Wrong examples (what NOT to do)
#
#  Everything here is STABLE across calls → maximises KV-cache hits
# ═════════════════════════════════════════════════════════════════════

ZONE_A_TEMPLATE = """\
<ROLE>
You are a Senior CRM Insights Analyst at Samsung. Your job is to extract
actionable micro-signal insights from aggregated behavioural data and
provide row-level citations for every claim.
</ROLE>

<INPUT_SCHEMA>
The data table below contains these columns:

  DIMENSIONS (categorical — use these to define user segments):
{dimension_definitions}

  METRICS (numeric — these are the signals to analyse):
{metric_definitions}

  Each row is uniquely identified by its ROW number (1-indexed from top).
  The source file is: {source_file}
</INPUT_SCHEMA>

<OUTPUT_SCHEMA>
Return a JSON array of exactly 5 insight objects.
Each insight MUST have row-level citations tracing every claim to the source data.

Return ONLY the JSON. No markdown fences. No commentary before or after.

{{
  "insights": [
    {{
      "insight_id": "NS_001",
      "insight": "A clear, specific, data-grounded statement about a behavioural trend or anomaly detected in the data. Must reference specific dimension values and metric numbers.",
      "why_interesting": "Marketing rationale explaining why this matters for campaign targeting, revenue impact, or customer lifecycle. Connect the data pattern to a business action.",
      "user_group": "Specific audience segment (e.g., Tech Enthusiasts, Gen-Z Early Adopters, Premium Upgraders, Budget-Conscious Families)",
      "product_category": "Specific product category from the data (e.g., Smartphones, TV, Wearables, Tablets)",
      "product_lob": "Line of Business code: MB | CE | TV | HA",
      "citations": [
        {{
          "file": "{source_file}",
          "row_id": 1,
          "metric_value": 0.54,
          "metric_name": "exact_column_name_from_data",
          "dimensions": {{
            "dimension_name_1": "exact_value_from_that_row",
            "dimension_name_2": "exact_value_from_that_row"
          }}
        }}
      ]
    }}
  ]
}}

FIELD RULES:
  insight_id    : Sequential "NS_001" to "NS_005"
  insight       : Must contain at least one specific number from the data
  why_interesting: Must connect the data pattern to a marketing action
  user_group    : Must be a meaningful audience segment, not a raw dimension value
  product_category: Must match a value in the product/category column of the data
  product_lob   : Must be one of: MB, CE, TV, HA
  citations     : 1 to 5 source rows. Every number in "insight" must trace to a citation.
    file        : Exact source filename (given above)
    row_id      : Integer row number (1-indexed) from the data table
    metric_value: Exact numeric value from that cell (number, NOT string)
    metric_name : Exact column name from the data table
    dimensions  : Dict of other column values from that same row
</OUTPUT_SCHEMA>

<GROUNDING_RULES>
1. Every number in "insight" MUST appear in at least one citation's metric_value.
2. Every citation's metric_value MUST match the actual value in the source data at that row_id.
3. Every citation's metric_name MUST be an actual column name from the data.
4. Every citation's row_id MUST be a valid row number from the data (1 to N).
5. Do NOT invent numbers. If a value is not in the data, do not cite it.
6. Do NOT hallucinate dimension values. Every value in citations.dimensions
   must exist in that row of the source data.
7. Each insight must cite DIFFERENT rows — do not reuse the same row_id
   across multiple insights unless the row genuinely supports both claims.
8. user_group should be a marketing-friendly segment name, not a raw
   dimension value. "Gen-Z Tech Enthusiasts" is good. "Gen-Z" alone is too vague.
</GROUNDING_RULES>

<WRONG_EXAMPLES>
Do NOT make these mistakes:

  WRONG: Number in insight without citation
    "insight": "Gen-Z shows 54% increase"
    "citations": []                          ← WHERE IS 54% FROM?

  WRONG: Citation metric_value doesn't match the data
    "metric_value": 0.62                     ← but the actual cell says 0.54
    (validator will catch this and reject the insight)

  WRONG: metric_value as string
    "metric_value": "0.54"                   ← WRONG (string)
    "metric_value": 0.54                     ← CORRECT (number)

  WRONG: row_id out of range
    "row_id": 150                            ← but data only has 48 rows

  WRONG: Vague user_group
    "user_group": "Gen-Z"                    ← too generic
    "user_group": "Gen-Z Mobile-First Shoppers" ← GOOD: specific + actionable

  WRONG: Markdown fences
    ```json                                  ← do not include
    {{"insights": [...]}}
    ```                                      ← do not include

  WRONG: Commentary outside JSON
    Here are the insights:                   ← do not include
    {{"insights": [...]}}
    Based on the above...                    ← do not include

  WRONG: Flat string for dimensions in citation
    "dimensions": "Gen-Z, Product-X, Mumbai" ← WRONG (string)
    "dimensions": {{"age_group": "Gen-Z", "product": "Product-X"}} ← CORRECT (dict)
</WRONG_EXAMPLES>"""


# ═════════════════════════════════════════════════════════════════════
#  ZONE C — Filled Example + Task (bottom, LLM attention recovers)
# ═════════════════════════════════════════════════════════════════════
#
#  Contains:
#    1. Two fully filled example insights with realistic citations
#    2. Task restatement with critical reminders
#
#  The filled example is the SINGLE most effective technique for
#  getting consistent JSON structure from an LLM.
# ═════════════════════════════════════════════════════════════════════

ZONE_C_TEMPLATE = """\
<FILLED_EXAMPLE>
Below are 2 CORRECT example insights. Your output must follow this
exact JSON structure for all 5 insights. Pay attention to:
  - citation row_ids are integers (not strings)
  - metric_values are numbers (not strings)
  - dimensions is a dict (not a string)
  - every number in "insight" text has a matching citation

{{
  "insights": [
    {{
      "insight_id": "NS_001",
      "insight": "Gen-Z users in Bengaluru show a 47% week-over-week increase in smartphone purchase velocity, with 3,200 active users — significantly above the national average of 12% for this age group.",
      "why_interesting": "This surge indicates a time-sensitive acquisition window. Gen-Z in Bengaluru are actively in-market for smartphones, likely driven by new product launches or seasonal factors. A targeted push notification campaign within the next 7 days could capture this momentum before it normalises.",
      "user_group": "Gen-Z Mobile-First Shoppers (Bengaluru)",
      "product_category": "Smartphones",
      "product_lob": "MB",
      "citations": [
        {{
          "file": "{source_file}",
          "row_id": 3,
          "metric_value": 0.47,
          "metric_name": "wow_purchase_velocity_pct",
          "dimensions": {{
            "age_group": "Gen-Z",
            "city": "Bengaluru",
            "product_category": "Smartphones"
          }}
        }},
        {{
          "file": "{source_file}",
          "row_id": 3,
          "metric_value": 3200,
          "metric_name": "active_users",
          "dimensions": {{
            "age_group": "Gen-Z",
            "city": "Bengaluru",
            "product_category": "Smartphones"
          }}
        }},
        {{
          "file": "{source_file}",
          "row_id": 42,
          "metric_value": 0.12,
          "metric_name": "wow_purchase_velocity_pct",
          "dimensions": {{
            "age_group": "Gen-Z",
            "city": "National Avg",
            "product_category": "Smartphones"
          }}
        }}
      ]
    }},
    {{
      "insight_id": "NS_002",
      "insight": "Millennial women in Mumbai have a cart abandonment rate of 68% on premium TVs (>50K INR), compared to 34% for the same segment on mid-range TVs — a 2x gap suggesting price friction, not product disinterest.",
      "why_interesting": "High add-to-cart but low conversion signals strong intent blocked by a barrier. An EMI/financing-focused retention campaign for this segment could recover significant revenue. The 2x gap between premium and mid-range confirms the barrier is price, not product fit.",
      "user_group": "Millennial Women Premium TV Considerers (Mumbai)",
      "product_category": "Television",
      "product_lob": "CE",
      "citations": [
        {{
          "file": "{source_file}",
          "row_id": 17,
          "metric_value": 0.68,
          "metric_name": "cart_abandonment_rate",
          "dimensions": {{
            "age_group": "Millennial",
            "gender": "Female",
            "city": "Mumbai",
            "price_tier": "Premium >50K"
          }}
        }},
        {{
          "file": "{source_file}",
          "row_id": 18,
          "metric_value": 0.34,
          "metric_name": "cart_abandonment_rate",
          "dimensions": {{
            "age_group": "Millennial",
            "gender": "Female",
            "city": "Mumbai",
            "price_tier": "Mid-Range 20-50K"
          }}
        }}
      ]
    }}
  ]
}}

Notice how each example:
  ✓ Every number in "insight" (47%, 3,200, 12%, 68%, 34%) has a citation
  ✓ Citations point to specific row_ids with exact metric_values
  ✓ user_group is a rich marketing segment, not just a dimension value
  ✓ why_interesting connects data to a specific campaign action
  ✓ dimensions dict captures the full context of each cited row
</FILLED_EXAMPLE>

<TASK>
Now analyse the ACTUAL data between the --- markers above.
Return exactly 5 insights as a JSON object matching the structure shown.

CRITICAL CHECKLIST (verify before returning):
  ☐ Exactly 5 insights, insight_id NS_001 through NS_005
  ☐ Every number in every "insight" text has a matching citation
  ☐ Every citation.row_id is a valid row number from the data
  ☐ Every citation.metric_value is a NUMBER, not a string
  ☐ Every citation.metric_name is an actual column from the data
  ☐ Every citation.dimensions is a dict, not a string
  ☐ product_lob is one of: MB, CE, TV, HA
  ☐ user_group is specific + actionable (not just "Gen-Z")
  ☐ No markdown fences, no commentary — JSON only
  ☐ All strings containing colons are properly escaped
</TASK>"""


# ═════════════════════════════════════════════════════════════════════
#  ZONE A + ZONE C BUILDER — Plugs your actual DataFrame columns in
# ═════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class InsightConfig:
    """
    Configure the insight extraction for your specific signal.

    Example:
        cfg = InsightConfig(
            source_file="purchase_velocity_w13_2026.csv",
            dimensions=["age_group", "gender", "city", "product_category"],
            metrics=["wow_purchase_velocity_pct", "active_users",
                     "purchase_rate", "total_revenue", "cart_conversion_rate"],
            lob_column="product_lob",
            valid_lobs=["MB", "CE", "TV", "HA"],
        )
    """
    source_file:  str                    # Exact filename for citation tracing
    dimensions:   List[str]              # Categorical columns (group-by)
    metrics:      List[str]              # Numeric columns (signals)
    lob_column:   str = "product_lob"    # LOB column name
    valid_lobs:   List[str] = None       # Valid LOB codes

    def __post_init__(self):
        if self.valid_lobs is None:
            self.valid_lobs = ["MB", "CE", "TV", "HA"]


def build_zone_a(df: pd.DataFrame, cfg: InsightConfig) -> str:
    """
    Generate Zone A from your DataFrame + config.
    Auto-discovers column descriptions from actual data.
    """
    # Build dimension definitions with actual unique values
    dim_lines = []
    for d in cfg.dimensions:
        if d in df.columns:
            uniq = sorted(str(v) for v in df[d].dropna().unique())
            if len(uniq) <= 15:
                vals = ", ".join(uniq)
                dim_lines.append(f"    {d:<30s}: {vals}")
            else:
                dim_lines.append(f"    {d:<30s}: ({len(uniq)} unique values)")
        else:
            dim_lines.append(f"    {d:<30s}: (column not found)")

    # Build metric definitions with ranges
    met_lines = []
    for m in cfg.metrics:
        if m in df.columns:
            mn, mx = df[m].min(), df[m].max()
            avg = df[m].mean()
            met_lines.append(f"    {m:<30s}: range [{mn:.4f} to {mx:.4f}], avg={avg:.4f}")
        else:
            met_lines.append(f"    {m:<30s}: (column not found)")

    return ZONE_A_TEMPLATE.format(
        dimension_definitions="\n".join(dim_lines),
        metric_definitions="\n".join(met_lines),
        source_file=cfg.source_file,
    )


def build_zone_c(cfg: InsightConfig) -> str:
    """Generate Zone C with filled example."""
    return ZONE_C_TEMPLATE.format(source_file=cfg.source_file)


# ═════════════════════════════════════════════════════════════════════
#  ZONE B — Inverted DataFrame (hybrid: cards + table)
# ═════════════════════════════════════════════════════════════════════

def build_zone_b(
    df: pd.DataFrame,
    cfg: InsightConfig,
    trend_column: str,
    cohort_size_column: str = "active_users",
    top_n_cards: int = 5,
) -> str:
    """
    Invert the DataFrame for Zone B injection.

    CRITICAL: Each row gets a ROW_ID so the LLM can cite it.
    """
    df = df.copy()

    # Add 1-indexed row IDs for citation tracing
    df.insert(0, "ROW_ID", range(1, len(df) + 1))

    # Column ordering: ROW_ID first, then dims, trend, metrics
    col_order = ["ROW_ID"] + cfg.dimensions + [trend_column, cohort_size_column]
    col_order += [m for m in cfg.metrics if m not in col_order]
    if cfg.lob_column in df.columns and cfg.lob_column not in col_order:
        col_order.append(cfg.lob_column)
    col_order = [c for c in col_order if c in df.columns]
    seen = set()
    col_order = [c for c in col_order if not (c in seen or seen.add(c))]

    # Sort by signal strength for card selection
    if trend_column in df.columns and cohort_size_column in df.columns:
        df["_score"] = df[trend_column].abs() * np.log1p(df[cohort_size_column])
        df = df.sort_values("_score", ascending=False).drop(columns=["_score"])

    # Add trend_flag
    if trend_column in df.columns:
        df["trend_flag"] = np.where(df[trend_column] > 0.10, "RISING",
                                    np.where(df[trend_column] < -0.10, "FALLING", "STABLE"))

    sections = [f"Source file: {cfg.source_file}", f"Total rows: {len(df)}", ""]

    # Top-N as signal cards (with ROW_ID for citation)
    sections.append("### Top signals (with ROW_ID for citation)\n")
    for _, row in df.head(top_n_cards).iterrows():
        rid = int(row["ROW_ID"])
        dims = " x ".join(str(row.get(d, "?")) for d in cfg.dimensions if d in row.index)
        tv = row.get(trend_column, 0)
        flag = row.get("trend_flag", "STABLE")
        arrow = "^" if flag == "RISING" else ("v" if flag == "FALLING" else "-")
        lines = [f"[ROW {rid}] {arrow} {flag} -- {dims}"]
        for m in cfg.metrics:
            if m in row.index and pd.notna(row[m]):
                v = row[m]
                if isinstance(v, float):
                    lines.append(f"  {m:<30s}: {v:.4f}")
                else:
                    lines.append(f"  {m:<30s}: {v}")
        if cfg.lob_column in row.index:
            lines.append(f"  {cfg.lob_column:<30s}: {row[cfg.lob_column]}")
        sections.append("\n".join(lines))
        sections.append("")

    # Full data as table (with ROW_ID column for citation lookup)
    sections.append("### Complete data (ROW_ID = citation reference)\n")
    display_cols = col_order.copy()
    if "trend_flag" in df.columns and "trend_flag" not in display_cols:
        display_cols.insert(2, "trend_flag")
    display_cols = [c for c in display_cols if c in df.columns]
    sections.append(df[display_cols].to_markdown(index=False))

    return "\n".join(sections)


# ═════════════════════════════════════════════════════════════════════
#  FULL PROMPT BUILDER
# ═════════════════════════════════════════════════════════════════════

def build_prompt(
    df: pd.DataFrame,
    cfg: InsightConfig,
    trend_column: str,
    cohort_size_column: str = "active_users",
) -> str:
    """
    Complete prompt: Zone A + Zone B + Zone C.

    Usage:
        prompt = build_prompt(df, cfg, trend_column="wow_velocity")
    """
    zone_a = build_zone_a(df, cfg)
    zone_b = build_zone_b(df, cfg, trend_column, cohort_size_column)
    zone_c = build_zone_c(cfg)

    return f"""{zone_a}

---

{zone_b}

---

{zone_c}"""


# ═════════════════════════════════════════════════════════════════════
#  JSON PARSER — Robust, handles common LLM output quirks
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ParseResult:
    success: bool
    data: Optional[dict] = None
    repair_steps: List[str] = field(default_factory=list)
    error: str = ""


def parse_llm_json(raw: str) -> ParseResult:
    """
    Multi-layer JSON parser for LLM output.

    Layer 1: Strip fences + preamble
    Layer 2: Fix trailing commas
    Layer 3: Fix single quotes → double quotes
    Layer 4: Extract JSON object from mixed text
    Layer 5: Fix unescaped control characters
    """
    result = ParseResult(success=False)

    if not raw or not raw.strip():
        result.error = "Empty input"
        return result

    # Layer 1: Strip markdown fences and preamble
    cleaned = re.sub(r'```json\s*\n?', '', raw.strip())
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    # Remove text before first { and after last }
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace >= 0 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]
    result.repair_steps.append("L1:strip_fences")

    # Try direct parse
    parsed = _try_json(cleaned)
    if parsed:
        result.success = True; result.data = parsed; return result

    # Layer 2: Fix trailing commas (common LLM mistake)
    fixed = re.sub(r',\s*}', '}', cleaned)
    fixed = re.sub(r',\s*]', ']', fixed)
    result.repair_steps.append("L2:trailing_commas")

    parsed = _try_json(fixed)
    if parsed:
        result.success = True; result.data = parsed; return result

    # Layer 3: Single quotes → double quotes
    sq_fixed = fixed.replace("'", '"')
    result.repair_steps.append("L3:single_quotes")

    parsed = _try_json(sq_fixed)
    if parsed:
        result.success = True; result.data = parsed; return result

    # Layer 4: Try to find and extract JSON object
    match = re.search(r'\{[\s\S]*"insights"[\s\S]*\}', raw)
    result.repair_steps.append("L4:regex_extract")
    if match:
        extracted = match.group()
        extracted = re.sub(r',\s*}', '}', extracted)
        extracted = re.sub(r',\s*]', ']', extracted)
        parsed = _try_json(extracted)
        if parsed:
            result.success = True; result.data = parsed; return result

    # Layer 5: Fix unescaped newlines and control chars in strings
    escaped = re.sub(r'(?<!\\)\n', '\\n', fixed)
    escaped = re.sub(r'(?<!\\)\t', '\\t', escaped)
    result.repair_steps.append("L5:escape_control")

    parsed = _try_json(escaped)
    if parsed:
        result.success = True; result.data = parsed; return result

    result.error = f"All layers failed. First 200 chars: {raw[:200]}"
    return result


def _try_json(text):
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════
#  CITATION VALIDATOR — Checks every citation against source data
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ValidationIssue:
    severity: str  # "CRITICAL" | "WARNING"
    insight_id: str
    field: str
    message: str

@dataclass
class ValidationReport:
    is_valid: bool = True
    checks: int = 0
    passed: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def fail(self, iid, field, msg):
        self.issues.append(ValidationIssue("CRITICAL", iid, field, msg))
        self.is_valid = False

    def warn(self, iid, field, msg):
        self.issues.append(ValidationIssue("WARNING", iid, field, msg))

    def ok(self): self.passed += 1

    def summary(self):
        cr = sum(1 for i in self.issues if i.severity == "CRITICAL")
        wr = sum(1 for i in self.issues if i.severity == "WARNING")
        lines = ["=" * 65,
                 f"  VALIDATION: {'PASSED' if self.is_valid else 'FAILED'}  "
                 f"| Checks: {self.checks} | Passed: {self.passed} "
                 f"| Critical: {cr} | Warnings: {wr}",
                 "-" * 65]
        for i in self.issues:
            ic = "[!!]" if i.severity == "CRITICAL" else "[! ]"
            lines.append(f"  {ic} {i.insight_id} {i.field}: {i.message}")
        if not self.issues: lines.append("  All checks passed.")
        lines.append("=" * 65)
        return "\n".join(lines)


def validate_insights(
    output: dict,
    df: pd.DataFrame,
    cfg: InsightConfig,
    tolerance: float = 0.01,
) -> ValidationReport:
    """
    Validate every insight and citation against the source DataFrame.

    Checks:
      L1: Structure    — required fields present
      L2: LOB          — product_lob in valid list
      L3: Citation row — row_id exists in data
      L4: Citation value — metric_value matches actual cell (within tolerance)
      L5: Citation name — metric_name is an actual column
      L6: Citation dims — dimension values match that row
      L7: Grounding    — every number in insight text has a citation
    """
    rp = ValidationReport()
    insights = output.get("insights", [])

    if not insights:
        rp.fail("", "insights", "No insights found")
        return rp

    n_rows = len(df)

    for ins in insights:
        iid = ins.get("insight_id", "?")

        # L1: Required fields
        rp.checks += 1
        required = {"insight_id", "insight", "why_interesting", "user_group",
                     "product_category", "product_lob", "citations"}
        missing = required - set(ins.keys())
        if missing:
            rp.fail(iid, "structure", f"Missing fields: {missing}")
        else:
            rp.ok()

        # L2: LOB validity
        rp.checks += 1
        lob = ins.get("product_lob", "")
        if lob not in cfg.valid_lobs:
            rp.fail(iid, "product_lob", f"'{lob}' not in {cfg.valid_lobs}")
        else:
            rp.ok()

        # L3-L6: Citation validation
        citations = ins.get("citations", [])
        rp.checks += 1
        if not citations:
            rp.fail(iid, "citations", "No citations — insight is ungrounded")
            continue
        else:
            rp.ok()

        for ci, cit in enumerate(citations):
            cp = f"citations[{ci}]"

            # L3: Row ID exists
            rp.checks += 1
            rid = cit.get("row_id")
            if not isinstance(rid, int) or rid < 1 or rid > n_rows:
                rp.fail(iid, f"{cp}.row_id", f"row_id={rid} out of range [1, {n_rows}]")
                continue
            else:
                rp.ok()

            row = df.iloc[rid - 1]  # Convert 1-indexed to 0-indexed

            # L4: Metric value matches actual cell
            rp.checks += 1
            mn = cit.get("metric_name", "")
            mv = cit.get("metric_value")

            if mn not in df.columns:
                rp.fail(iid, f"{cp}.metric_name", f"'{mn}' is not a column in the data")
                continue
            else:
                rp.ok()

            rp.checks += 1
            actual = row[mn]
            if pd.notna(actual) and mv is not None:
                if isinstance(mv, str):
                    rp.warn(iid, f"{cp}.metric_value", f"String '{mv}' instead of number")
                    try:
                        mv = float(mv)
                    except ValueError:
                        rp.fail(iid, f"{cp}.metric_value", f"Cannot parse '{mv}' as number")
                        continue

                if abs(float(actual) - float(mv)) > tolerance:
                    rp.fail(iid, f"{cp}.metric_value",
                            f"Claimed {mv} but actual row {rid} has {actual:.4f} "
                            f"(diff={abs(float(actual) - float(mv)):.4f})")
                else:
                    rp.ok()
            else:
                rp.ok()

            # L6: Dimension values match that row
            dims = cit.get("dimensions", {})
            if isinstance(dims, dict):
                for dk, dv in dims.items():
                    if dk in df.columns:
                        rp.checks += 1
                        actual_dim = str(row.get(dk, ""))
                        if str(dv) != actual_dim:
                            rp.fail(iid, f"{cp}.dimensions.{dk}",
                                    f"Claimed '{dv}' but row {rid} has '{actual_dim}'")
                        else:
                            rp.ok()

        # L7: Numbers in insight text should have citations
        insight_text = ins.get("insight", "")
        numbers_in_text = re.findall(r'(\d+\.?\d*)\s*%', insight_text)
        cited_values = set()
        for cit in citations:
            mv = cit.get("metric_value")
            if mv is not None:
                cited_values.add(round(float(mv) * 100, 1))  # Convert to percentage
                cited_values.add(round(float(mv), 4))         # Raw value
                cited_values.add(int(float(mv)))               # Integer form

        for num_str in numbers_in_text:
            rp.checks += 1
            num = float(num_str)
            # Check if this number appears in any citation (as raw or percentage)
            if num in cited_values or round(num, 1) in cited_values:
                rp.ok()
            else:
                rp.warn(iid, "grounding",
                        f"'{num_str}%' in insight text but no matching citation value")

    return rp


# ═════════════════════════════════════════════════════════════════════
#  DEMO
# ═════════════════════════════════════════════════════════════════════

def demo():
    np.random.seed(42)
    print("=" * 65)
    print("  CRM Insight Extraction — JSON Schema with Citation Tracing")
    print("=" * 65)

    import os
    os.makedirs("outputs", exist_ok=True)

    # Create demo data
    rows = []
    for ag in ["Gen-Z", "Millennial", "Gen-X", "Boomer"]:
        for p in ["Smartphones", "Television", "Wearables", "Tablets"]:
            for c in ["Bengaluru", "Mumbai", "Delhi"]:
                lob = {"Smartphones": "MB", "Television": "CE",
                       "Wearables": "MB", "Tablets": "CE"}[p]
                au = np.random.randint(200, 5000)
                vel = round(np.random.uniform(-0.5, 0.6), 4)
                if ag == "Gen-Z" and p == "Smartphones": vel = round(abs(vel) + 0.3, 4)
                rows.append({
                    "age_group": ag, "product_category": p, "city": c,
                    "product_lob": lob,
                    "active_users": au,
                    "purchase_rate": round(np.random.uniform(0.02, 0.15), 4),
                    "total_revenue": round(au * np.random.uniform(100, 800), 2),
                    "wow_purchase_velocity_pct": vel,
                    "cart_conversion_rate": round(np.random.uniform(0.1, 0.5), 4),
                    "web_footprint_vs_national": round(np.random.uniform(0.5, 2.0), 4),
                })
    df = pd.DataFrame(rows)

    cfg = InsightConfig(
        source_file="purchase_velocity_w13_2026.csv",
        dimensions=["age_group", "product_category", "city", "product_lob"],
        metrics=["wow_purchase_velocity_pct", "active_users", "purchase_rate",
                 "total_revenue", "cart_conversion_rate", "web_footprint_vs_national"],
    )

    # Build prompt
    prompt = build_prompt(df, cfg, trend_column="wow_purchase_velocity_pct")
    with open("outputs/insight_prompt.txt", "w") as f:
        f.write(prompt)

    print(f"\n  DataFrame   : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Source file : {cfg.source_file}")
    print(f"  Dimensions  : {cfg.dimensions}")
    print(f"  Metrics     : {cfg.metrics}")
    print(f"  Prompt      : {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")

    # Show Zone A excerpt
    za = build_zone_a(df, cfg)
    print(f"\n  Zone A      : {len(za):,} chars")
    print(f"  Zone C      : {len(build_zone_c(cfg)):,} chars")

    # Test JSON parser
    print(f"\n{'─' * 65}")
    print("  JSON Parser Tests")
    print(f"{'─' * 65}")

    test_cases = {
        "Clean JSON": '{"insights": [{"insight_id": "NS_001", "insight": "test", "citations": []}]}',
        "With fences": '```json\n{"insights": [{"insight_id": "NS_001", "insight": "test", "citations": []}]}\n```',
        "Trailing commas": '{"insights": [{"insight_id": "NS_001", "insight": "test", "citations": [],},]}',
        "With preamble": 'Here are the insights:\n\n{"insights": [{"insight_id": "NS_001", "insight": "test", "citations": []}]}\n\nThese trends show...',
    }

    for name, raw in test_cases.items():
        r = parse_llm_json(raw)
        n = len(r.data.get("insights", [])) if r.data else 0
        print(f"  [{('OK' if r.success else 'FAIL')}] {name:<25s} insights={n}  repairs={' -> '.join(r.repair_steps[:3])}")

    print(f"\n  Full prompt saved to outputs/insight_prompt.txt")


if __name__ == "__main__":
    demo()

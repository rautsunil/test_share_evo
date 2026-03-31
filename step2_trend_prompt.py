"""
STEP 2 — Context Engineering: Trend Identification Prompt Builder
==================================================================
CORE PRINCIPLE — Lost in the Middle (Liu et al., 2023):
    LLMs attend strongly to the BEGINNING and END of their context window
    but performance degrades for information placed in the MIDDLE.

    This module structures the prompt so that:
    ┌──────────────────────────────────────────────────────────────┐
    │  ZONE A — SYSTEM FRAME (beginning)                          │
    │    • Role definition + output schema + grounding rules      │
    │    • CRITICAL: trend_flag column semantics explained here    │
    │                                                              │
    │  ZONE B — DATA PAYLOAD (middle)                              │
    │    • The serialised DataFrame                                │
    │    • Pre-filtered & pre-sorted to reduce noise               │
    │    • Column descriptions are NOT here (moved to Zone A/C)    │
    │                                                              │
    │  ZONE C — TASK RESTATEMENT + GUARDRAILS (end)                │
    │    • Re-anchors the exact task (top-5 trends)                │
    │    • Output format reminder                                  │
    │    • "Only cite data visible above" grounding instruction    │
    └──────────────────────────────────────────────────────────────┘

WHY THIS ORDERING MATTERS:
    If you bury the output schema or grounding rules inside the middle
    (between the data rows), the LLM is ~20-30% more likely to:
      (a) hallucinate metrics not in the table
      (b) ignore the requested output format
      (c) miss weak signals buried in the middle rows
"""

import pandas as pd
import json
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# 2A.  DATA PRE-CONDITIONING  (Shrink the middle, sharpen signal)
# ─────────────────────────────────────────────────────────────────────

def precondition_dataframe(
    df: pd.DataFrame,
    top_n_per_group: int = 5,
    min_active_users: int = 50
) -> pd.DataFrame:
    """
    Reduce the DataFrame to only the rows that carry signal,
    so the LLM's middle zone is as small as possible.

    Strategy:
    1. Filter out low-volume cohorts (noise).
    2. Keep only the latest week (trend_flag is already WoW).
    3. Within each (age_group × product_category), keep top-N
       by |wow_purchase_velocity_pct|  — strongest movers.
    4. Append a SUMMARY block: national averages per product.
    """
    df = df.copy()

    # 1. Remove noise
    df = df[df["active_users"] >= min_active_users]

    # 2. Latest week only (most recent signal)
    latest_week = df["week_start"].max()
    df_latest = df[df["week_start"] == latest_week].copy()

    # 3. Top movers per cohort
    df_latest["abs_velocity"] = df_latest["wow_purchase_velocity_pct"].abs()
    df_top = (
        df_latest
        .sort_values("abs_velocity", ascending=False)
        .groupby(["age_group", "product_category"])
        .head(top_n_per_group)
        .drop(columns=["abs_velocity"])
    )

    # 4. National summary row (appended at BOTTOM = high-attention zone)
    summary = (
        df_latest
        .groupby("product_category")
        .agg(
            national_avg_purchase_rate=("purchase_rate", "mean"),
            national_avg_velocity=("wow_purchase_velocity_pct", "mean"),
            national_avg_web_footprint=("web_footprint_vs_national", "mean"),
            total_active_users=("active_users", "sum"),
        )
        .reset_index()
    )

    return df_top, summary


# ─────────────────────────────────────────────────────────────────────
# 2B.  SERIALISE DATAFRAME → PROMPT-FRIENDLY FORMAT
# ─────────────────────────────────────────────────────────────────────

def serialise_for_prompt(
    df: pd.DataFrame,
    max_rows: int = 120,
    format: str = "markdown"    # "markdown" | "csv" | "json_records"
) -> str:
    """
    Convert DataFrame to a string that will sit in ZONE B of the prompt.

    Markdown tables are preferred because:
    - LLMs parse them accurately (trained on GitHub/docs).
    - Row boundaries are explicit (pipe delimiters).
    - Column headers stay visible even if the table is long.
    """
    df_trunc = df.head(max_rows)

    if format == "markdown":
        return df_trunc.to_markdown(index=False)
    elif format == "csv":
        return df_trunc.to_csv(index=False)
    elif format == "json_records":
        return json.dumps(
            df_trunc.to_dict(orient="records"),
            indent=2, default=str
        )
    else:
        raise ValueError(f"Unknown format: {format}")


# ─────────────────────────────────────────────────────────────────────
# 2C.  THE PROMPT TEMPLATE  (Lost-in-the-Middle optimised)
# ─────────────────────────────────────────────────────────────────────

TREND_IDENTIFICATION_PROMPT = """
{zone_a_system_frame}

---

## DATA TABLE — Aggregated Micro-Signals (Latest Week: {latest_week})

{zone_b_data_payload}

---

### National Benchmark Summary (per product category)

{zone_b_national_summary}

---

{zone_c_task_restatement}
"""

ZONE_A_SYSTEM_FRAME = """\
<ROLE>
You are a Senior CRM Analytics Strategist. Your job is to identify the
TOP 5 most actionable demographic-behavioural trends from the data table
provided below.
</ROLE>

<COLUMN_DEFINITIONS>
These are the columns you will see in the data table:

| Column                       | Meaning                                                              |
|------------------------------|----------------------------------------------------------------------|
| age_group                    | Generational cohort: Gen-Z, Millennial, Gen-X, Boomer               |
| product_category             | Product line                                                         |
| trend_flag                   | RISING (>+10% WoW), FALLING (<-10% WoW), STABLE (between ±10%)      |
| wow_purchase_velocity_pct    | Week-over-week % change in purchase count for this cohort            |
| gender                       | Male / Female                                                        |
| city                         | City of the user cohort                                              |
| active_users                 | Count of distinct active users in this cohort this week              |
| purchasers                   | Count of distinct users who made a purchase                          |
| purchase_count               | Total number of purchase events                                      |
| total_revenue                | Sum of revenue from purchases (INR)                                  |
| page_views                   | Total page views by this cohort                                      |
| sessions                     | Total sessions by this cohort                                        |
| add_to_cart_events           | Total add-to-cart events                                             |
| purchase_rate                | purchasers / active_users (conversion rate)                          |
| cart_conversion_rate         | purchase_count / add_to_cart_events                                  |
| pages_per_session            | Engagement depth: page_views / sessions                              |
| web_footprint_vs_national    | Ratio of this cohort's page_views to the national average (>1 = above avg) |
</COLUMN_DEFINITIONS>

<GROUNDING_RULES>
1. Every insight MUST reference a specific (age_group, product_category) pair.
2. Every numeric claim MUST be directly traceable to a value in the table.
3. Do NOT infer trends that are not visible in the data.
4. If a signal is weak (|velocity| < 5%), do NOT report it as a top trend.
5. Flag the CONFIDENCE LEVEL of each trend: HIGH (large cohort + strong signal),
   MEDIUM (moderate), LOW (small cohort or noisy).
</GROUNDING_RULES>

<OUTPUT_SCHEMA>
Return EXACTLY this JSON structure — no extra commentary outside the JSON:

```json
{{
  "trends": [
    {{
      "rank": 1,
      "trend_title": "Short descriptive title",
      "age_group": "...",
      "product_category": "...",
      "gender": "All | Male | Female",
      "city": "All | specific city",
      "signal_type": "purchase_velocity | web_footprint | cart_conversion | engagement",
      "direction": "RISING | FALLING",
      "metric_value": 0.0,
      "metric_name": "column name from the table",
      "benchmark_comparison": "X% above/below national average",
      "confidence": "HIGH | MEDIUM | LOW",
      "evidence_summary": "1-2 sentence grounded explanation"
    }}
  ],
  "data_quality_notes": "Any caveats about small sample sizes or missing data"
}}
```
</OUTPUT_SCHEMA>"""


ZONE_C_TASK_RESTATEMENT = """\
<TASK_REMINDER>
You have been given the aggregated micro-signal data above.

YOUR TASK: Identify the TOP 5 most significant and actionable trends.
Prioritise trends that are:
  (a) Statistically meaningful (large cohort size via active_users)
  (b) Strong signal (|wow_purchase_velocity_pct| > 15% OR web_footprint_vs_national > 1.3)
  (c) Actionable for a marketing campaign targeting specific demographics

REMEMBER:
- Return ONLY the JSON object defined in OUTPUT_SCHEMA above.
- Every claim must cite a specific value from the data table.
- DO NOT hallucinate metrics. If a value is not in the table, do not mention it.
- Rank trends by business impact (revenue × cohort size × signal strength).
</TASK_REMINDER>"""


# ─────────────────────────────────────────────────────────────────────
# 2D.  PROMPT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────

def build_trend_identification_prompt(
    df: pd.DataFrame,
    max_data_rows: int = 120,
    data_format: str = "markdown"
) -> str:
    """
    Assemble the full prompt with Lost-in-the-Middle ordering:
      Zone A (beginning) → Zone B (middle) → Zone C (end)
    """
    # Pre-condition the data
    df_top, df_summary = precondition_dataframe(df)

    # Serialise
    data_payload     = serialise_for_prompt(df_top, max_rows=max_data_rows, format=data_format)
    summary_payload  = serialise_for_prompt(df_summary, format=data_format)

    latest_week = df["week_start"].max()

    prompt = TREND_IDENTIFICATION_PROMPT.format(
        zone_a_system_frame   = ZONE_A_SYSTEM_FRAME,
        latest_week           = latest_week,
        zone_b_data_payload   = data_payload,
        zone_b_national_summary = summary_payload,
        zone_c_task_restatement = ZONE_C_TASK_RESTATEMENT,
    )

    return prompt


# ─────────────────────────────────────────────────────────────────────
# 2E.  STAND-ALONE EXECUTION
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import Step 1 to generate data
    from step1_sql_aggregation import generate_synthetic_data

    print("=" * 65)
    print("STEP 2: Building Context-Engineered Trend Identification Prompt")
    print("=" * 65)

    df = generate_synthetic_data(n_weeks=4)
    prompt = build_trend_identification_prompt(df)

    # Save prompt to file for inspection
    with open("outputs/trend_identification_prompt.txt", "w") as f:
        f.write(prompt)

    print(f"\nPrompt length: {len(prompt)} chars  (~{len(prompt)//4} tokens)")
    print(f"Saved to: outputs/trend_identification_prompt.txt")

    # Show structure
    print("\n── PROMPT STRUCTURE ──")
    lines = prompt.split("\n")
    for i, line in enumerate(lines[:5]):
        print(f"  [TOP]    {line[:80]}")
    print(f"  ... ({len(lines) - 10} lines in middle) ...")
    for line in lines[-5:]:
        print(f"  [BOTTOM] {line[:80]}")

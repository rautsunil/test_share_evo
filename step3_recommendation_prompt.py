"""
STEP 3 — Context Engineering: Campaign Recommendation Prompt Builder
=====================================================================
This is the SECOND LLM call in the chain:
    Step 2 (Trends)  →  Step 3 (Recommendations)

CONTEXT ENGINEERING STRATEGY:
    The trend output from Step 2 is now UPSTREAM CONTEXT for Step 3.
    We apply Lost-in-the-Middle again, but with a twist:

    ┌──────────────────────────────────────────────────────────────┐
    │  ZONE A — ROLE + CAMPAIGN CONSTRAINTS (beginning)            │
    │    • Budget tiers, channel options, compliance rules          │
    │    • Output schema for recommendations                       │
    │                                                              │
    │  ZONE B — TREND INSIGHTS (middle, but structured)            │
    │    • Injected as numbered list (not raw JSON)                 │
    │    • Each trend tagged with an ID for citation tracing        │
    │    • We COMPRESS the data here — only trend summaries,        │
    │      not the full DataFrame                                   │
    │                                                              │
    │  ZONE C — RECOMMENDATION TASK + EXAMPLES (end)               │
    │    • Concrete few-shot example of a good recommendation       │
    │    • Re-anchor: "cite TREND-ID for every recommendation"      │
    │    • Grounding: "do not recommend for cohorts not in trends"   │
    └──────────────────────────────────────────────────────────────┘

ANTI-HALLUCINATION DESIGN:
    Each recommendation MUST cite a TREND-ID from Zone B.
    The citation validator in Step 4 will cross-check these.
"""

import json
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────
# 3A.  ZONE A — SYSTEM FRAME
# ─────────────────────────────────────────────────────────────────────

ZONE_A_RECOMMENDATION_FRAME = """\
<ROLE>
You are a CRM Campaign Strategist at a consumer electronics company.
Your job is to design the TOP 5 targeted marketing campaigns based on
the trend insights provided below.
</ROLE>

<CAMPAIGN_CONSTRAINTS>
1. Each campaign MUST target a specific demographic segment:
   - At least one of: age_group, gender, city must be specified.
   - Targeting "everyone" is NOT allowed.

2. Available campaign channels:
   - PUSH_NOTIFICATION  (low cost, high reach, low conversion)
   - EMAIL              (low cost, medium reach, medium conversion)
   - SMS                (medium cost, medium reach, high conversion)
   - IN_APP_BANNER      (no cost, high reach, low conversion)
   - SOCIAL_ADS         (high cost, high reach, variable conversion)
   - WHATSAPP           (medium cost, medium reach, high conversion)

3. Budget tiers:
   - TIER_1: ₹0 – ₹50K     (push, in-app, email only)
   - TIER_2: ₹50K – ₹2L    (+ SMS, WhatsApp)
   - TIER_3: ₹2L+           (+ social ads, influencer)

4. Compliance rules:
   - No campaigns targeting users under 18.
   - SMS/WhatsApp requires prior opt-in.
   - FALLING trends may need retention (not acquisition) framing.
</CAMPAIGN_CONSTRAINTS>

<OUTPUT_SCHEMA>
Return EXACTLY this JSON structure:

```json
{{
  "recommendations": [
    {{
      "rank": 1,
      "campaign_name": "Catchy campaign name",
      "cited_trend_ids": ["TREND-1"],
      "target_segment": {{
        "age_group": "Gen-Z",
        "gender": "All | Male | Female",
        "city": "All | specific city"
      }},
      "product_focus": "Product-X",
      "campaign_type": "ACQUISITION | RETENTION | REACTIVATION | UPSELL",
      "recommended_channels": ["PUSH_NOTIFICATION", "EMAIL"],
      "budget_tier": "TIER_1 | TIER_2 | TIER_3",
      "estimated_reach": "approximate user count from data",
      "key_message": "The core message / value proposition",
      "timing": "Immediate | Next 7 days | Next 30 days",
      "success_metric": "What KPI to track (e.g., purchase_rate, cart_conversion_rate)",
      "expected_lift": "Estimated % improvement with reasoning",
      "rationale": "2-3 sentence explanation grounded in the cited trend"
    }}
  ],
  "campaigns_not_recommended": [
    {{
      "segment": "description",
      "reason": "Why this segment was excluded despite appearing in trends"
    }}
  ]
}}
```
</OUTPUT_SCHEMA>"""


# ─────────────────────────────────────────────────────────────────────
# 3B.  TREND → STRUCTURED CONTEXT CONVERTER
# ─────────────────────────────────────────────────────────────────────

def format_trends_for_context(trend_json: dict) -> str:
    """
    Convert the raw JSON output from Step 2 into a numbered,
    citation-ready format for Zone B.

    Each trend gets a TREND-ID that the recommendation must cite.
    This enables the citation validator in Step 4 to trace
    every recommendation back to its supporting evidence.
    """
    if isinstance(trend_json, str):
        trend_json = json.loads(trend_json)

    trends = trend_json.get("trends", [])
    lines = []

    for t in trends:
        trend_id = f"TREND-{t['rank']}"
        lines.append(f"""
[{trend_id}]
  Title:      {t['trend_title']}
  Segment:    {t['age_group']} / {t['gender']} / {t.get('city', 'All')}
  Product:    {t['product_category']}
  Signal:     {t['signal_type']} → {t['direction']}
  Value:      {t['metric_name']} = {t['metric_value']}
  Benchmark:  {t['benchmark_comparison']}
  Confidence: {t['confidence']}
  Evidence:   {t['evidence_summary']}
""")

    # Data quality notes at the end (bottom zone)
    notes = trend_json.get("data_quality_notes", "None")
    lines.append(f"\n[DATA-QUALITY-NOTES]\n  {notes}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 3C.  ZONE C — TASK RESTATEMENT + FEW-SHOT EXAMPLE
# ─────────────────────────────────────────────────────────────────────

ZONE_C_RECOMMENDATION_TASK = """\
<FEW_SHOT_EXAMPLE>
Here is an example of a GOOD recommendation:

{{
  "rank": 1,
  "campaign_name": "Gen-Z Product-X Flash Sale",
  "cited_trend_ids": ["TREND-1"],
  "target_segment": {{
    "age_group": "Gen-Z",
    "gender": "All",
    "city": "All"
  }},
  "product_focus": "Product-X",
  "campaign_type": "ACQUISITION",
  "recommended_channels": ["PUSH_NOTIFICATION", "IN_APP_BANNER"],
  "budget_tier": "TIER_1",
  "estimated_reach": "~12,000 active Gen-Z users",
  "key_message": "Product-X is trending with your peers — grab it before stock runs out",
  "timing": "Immediate",
  "success_metric": "purchase_rate for Gen-Z × Product-X cohort",
  "expected_lift": "8-12% based on current 18% WoW velocity",
  "rationale": "TREND-1 shows Gen-Z purchase velocity on Product-X is RISING at +18% WoW with HIGH confidence. Capitalising with a low-cost push + in-app campaign can amplify organic momentum."
}}

Notice how the example:
  ✓ Cites a specific TREND-ID
  ✓ Targets a specific segment (not "everyone")
  ✓ Recommends channels appropriate to the budget tier
  ✓ Grounds the expected lift in actual data from the trend
</FEW_SHOT_EXAMPLE>

<TASK_REMINDER>
You have been given {n_trends} trend insights above.

YOUR TASK: Design the TOP 5 targeted campaigns.

CRITICAL GROUNDING RULES:
1. Every recommendation MUST cite at least one TREND-ID from the list above.
2. Do NOT recommend campaigns for segments that have no supporting trend.
3. FALLING trends should get RETENTION campaigns, not ACQUISITION.
4. RISING trends should get ACQUISITION or UPSELL campaigns.
5. Match budget tier to the campaign channels you recommend.
6. estimated_reach should be grounded in the active_users number from trends.

Return ONLY the JSON object. No commentary outside the JSON.
</TASK_REMINDER>"""


# ─────────────────────────────────────────────────────────────────────
# 3D.  PROMPT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────

RECOMMENDATION_PROMPT_TEMPLATE = """
{zone_a}

---

## TREND INSIGHTS (from upstream analysis)

The following trends were identified from this week's micro-signal data.
Each trend has a TREND-ID that you MUST cite in your recommendations.

{zone_b_trends}

---

{zone_c}
"""


def build_recommendation_prompt(trend_json: dict) -> str:
    """
    Assemble the full recommendation prompt.

    Args:
        trend_json: The parsed JSON output from Step 2's trend identification.
    """
    zone_b = format_trends_for_context(trend_json)
    n_trends = len(trend_json.get("trends", []))

    prompt = RECOMMENDATION_PROMPT_TEMPLATE.format(
        zone_a      = ZONE_A_RECOMMENDATION_FRAME,
        zone_b_trends = zone_b,
        zone_c      = ZONE_C_RECOMMENDATION_TASK.format(n_trends=n_trends),
    )

    return prompt


# ─────────────────────────────────────────────────────────────────────
# 3E.  STAND-ALONE EXECUTION
# ─────────────────────────────────────────────────────────────────────

# Example trend output (simulating Step 2 result)
SAMPLE_TREND_OUTPUT = {
    "trends": [
        {
            "rank": 1,
            "trend_title": "Gen-Z Purchase Surge on Product-X",
            "age_group": "Gen-Z",
            "product_category": "Product-X",
            "gender": "All",
            "city": "All",
            "signal_type": "purchase_velocity",
            "direction": "RISING",
            "metric_value": 0.54,
            "metric_name": "wow_purchase_velocity_pct",
            "benchmark_comparison": "54% above previous week",
            "confidence": "HIGH",
            "evidence_summary": "Gen-Z cohort shows a +54% WoW increase in purchase count for Product-X across all cities, with 3,200+ active users."
        },
        {
            "rank": 2,
            "trend_title": "Millennial Product-Y Purchase Decline",
            "age_group": "Millennial",
            "product_category": "Product-Y",
            "gender": "All",
            "city": "All",
            "signal_type": "purchase_velocity",
            "direction": "FALLING",
            "metric_value": -0.45,
            "metric_name": "wow_purchase_velocity_pct",
            "benchmark_comparison": "45% below previous week",
            "confidence": "HIGH",
            "evidence_summary": "Millennial purchase velocity on Product-Y dropped 45% WoW, indicating potential churn or product fatigue."
        },
        {
            "rank": 3,
            "trend_title": "Male Bengaluru Web Engagement Spike",
            "age_group": "All",
            "product_category": "All",
            "gender": "Male",
            "city": "Bengaluru",
            "signal_type": "web_footprint",
            "direction": "RISING",
            "metric_value": 1.62,
            "metric_name": "web_footprint_vs_national",
            "benchmark_comparison": "62% above national average",
            "confidence": "HIGH",
            "evidence_summary": "Male users in Bengaluru show page views 62% above national average across all product categories."
        },
        {
            "rank": 4,
            "trend_title": "Gen-X Female Steady Demand for Product-Z",
            "age_group": "Gen-X",
            "product_category": "Product-Z",
            "gender": "Female",
            "city": "All",
            "signal_type": "purchase_velocity",
            "direction": "RISING",
            "metric_value": 0.22,
            "metric_name": "wow_purchase_velocity_pct",
            "benchmark_comparison": "22% above previous week, consistently rising",
            "confidence": "MEDIUM",
            "evidence_summary": "Gen-X females show a steady +22% WoW purchase increase for Product-Z with above-average purchase rates."
        },
        {
            "rank": 5,
            "trend_title": "Boomer Cart Abandonment on Product-W in Delhi",
            "age_group": "Boomer",
            "product_category": "Product-W",
            "gender": "All",
            "city": "Delhi",
            "signal_type": "cart_conversion",
            "direction": "FALLING",
            "metric_value": 0.16,
            "metric_name": "cart_conversion_rate",
            "benchmark_comparison": "Cart conversion at 16%, well below 40% category average",
            "confidence": "MEDIUM",
            "evidence_summary": "Boomers in Delhi show high add-to-cart activity for Product-W but very low conversion, suggesting UX friction or price sensitivity."
        }
    ],
    "data_quality_notes": "Boomer cohort in smaller cities has fewer than 100 active users — trends for those sub-segments should be treated with caution."
}


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 3: Building Context-Engineered Recommendation Prompt")
    print("=" * 65)

    prompt = build_recommendation_prompt(SAMPLE_TREND_OUTPUT)

    with open("outputs/recommendation_prompt.txt", "w") as f:
        f.write(prompt)

    print(f"\nPrompt length: {len(prompt)} chars  (~{len(prompt)//4} tokens)")
    print(f"Saved to: outputs/recommendation_prompt.txt")

    # Show Zone B (trend context)
    zone_b = format_trends_for_context(SAMPLE_TREND_OUTPUT)
    print("\n── ZONE B: Trend Context (what the LLM sees in the middle) ──")
    print(zone_b[:600])
    print("...")

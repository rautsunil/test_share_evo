"""
Context Engineering Workbench — Flask App
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, stream_with_context, Response
import json
import os
import urllib.request
import urllib.error

app = Flask(__name__)

LAYER_DEFAULTS = {
    "role": "You are a senior CRM data scientist at a B2C e-commerce company. You specialize in customer segmentation, behavioral analysis, and personalized campaign strategy.",
    "instructions": "Always respond in structured JSON with keys: segment_label, insight, recommended_action, channel, urgency, confidence_score.\nRules:\n- Be specific, never generic\n- Base urgency on recency of last purchase\n- Prefer push notifications for mobile-active users\n- confidence_score must be 0.0 to 1.0",
    "memory": "Past campaign performance:\n- Email campaign 2 weeks ago → 12% conversion, 34% open rate (above avg)\n- Push notification last month → 8% CTR\n- Segment historically responds well to urgency-based messaging",
    "tools": "Available tools:\n- get_segment_stats(segment_id): Returns live engagement metrics\n- fetch_product_catalog(): Returns current inventory and pricing\n- schedule_campaign(channel, message, segment_id): Queues a campaign",
    "state": "Current context:\n- Date: 2026-03-23 (Monday)\n- Active promotion: End of Quarter Sale (ends March 31)\n- Inventory alert: Electronics category 40% stock remaining\n- Platform: Mobile app (85% of user traffic)",
    "examples": 'Example 1:\nInput: Age 25-30, last purchase 10 days ago, high AOV\nOutput: {"segment_label":"High-Value Recent","insight":"Active buyer with premium spending","recommended_action":"Cross-sell premium accessories","channel":"push","urgency":"medium","confidence_score":0.87}\n\nExample 2:\nInput: Age 35-45, last purchase 90 days ago, low engagement\nOutput: {"segment_label":"At-Risk Dormant","insight":"Lapsing customer needs reactivation","recommended_action":"Win-back offer with 20% discount","channel":"email","urgency":"high","confidence_score":0.91}',
    "query": "Segment data: Age 28-35, City: Bengaluru, Last purchase: 45 days ago, Mobile-active: Yes, Avg order value: ₹2200, Category affinity: Electronics"
}


def build_context(layers: dict, enabled: dict) -> tuple[str, str]:
    system_parts = []
    if enabled.get("role"):
        system_parts.append(f"## Role\n{layers.get('role','')}")
    if enabled.get("instructions"):
        system_parts.append(f"## Instructions\n{layers.get('instructions','')}")
    if enabled.get("tools"):
        system_parts.append(f"## Available Tools\n{layers.get('tools','')}")
    if enabled.get("examples"):
        system_parts.append(f"## Examples\n{layers.get('examples','')}")

    user_parts = []
    if enabled.get("memory"):
        user_parts.append(f"### Memory & History\n{layers.get('memory','')}")
    if enabled.get("state"):
        user_parts.append(f"### Current State\n{layers.get('state','')}")
    if enabled.get("query"):
        user_parts.append(f"### Query\n{layers.get('query','')}")

    return "\n\n".join(system_parts), "\n\n".join(user_parts)


@app.route("/")
def index():
    # Send as static file — bypasses Jinja2 so JSX curly braces are never parsed
    return app.send_static_file("index.html")


@app.route("/api/defaults")
def get_defaults():
    return jsonify(LAYER_DEFAULTS)


@app.route("/api/run", methods=["POST"])
def run_llm():
    data = request.json
    api_key = data.get("api_key", "").strip()
    layers  = data.get("layers", {})
    enabled = data.get("enabled", {})

    if not api_key:
        return jsonify({"error": "API key is required."}), 400

    active = [k for k, v in enabled.items() if v]
    if not active:
        return jsonify({"error": "Enable at least one context layer."}), 400

    system_prompt, user_message = build_context(layers, enabled)

    payload = json.dumps({
        "model": "claude-opus-4-5",
        "max_tokens": 1024,
        "system": system_prompt or "You are a helpful assistant.",
        "messages": [{"role": "user", "content": user_message or "Hello"}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result.get("content", [{}])[0].get("text", "No response")
            usage = result.get("usage", {})
            return jsonify({
                "output": text,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            msg = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            msg = body
        return jsonify({"error": f"API error {e.code}: {msg}"}), e.code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

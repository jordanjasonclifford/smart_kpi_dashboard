from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv


# Load local API keys from .env for development.
load_dotenv()


# System prompt used for Claude commentary.
# The app computes the numbers, Claude explains what they mean.
# This keeps the model away from doing arithmetic and makes the output easier to audit.
SYSTEM_INSTRUCTIONS = """
You are an analytics partner writing for a mid-size operations leadership team.
Translate KPI movement into concise business commentary.
Use plain English, mention the direction and likely operational implication, and avoid claiming causes not present in the data.
Return JSON with keys: revenue, orders, profit, profit_margin, ai_summary.
The KPI values must each be 2 to 3 sentences.
ai_summary must be one paragraph, no more than 8 sentences, synthesizing the KPI movement and including practical next-step insight.
""".strip()


def generate_ai_analysis(
    context: dict[str, Any],
    model: str = "claude-sonnet-4-20250514",
    enabled: bool = True,
) -> dict[str, str]:
    # Returns Claude commentary when enabled and configured.
    # Falls back to local commentary so the demo still works without a key.
    # The app calls this only after the user clicks the generate button.
    if not enabled or not has_anthropic_api_key():
        return generate_fallback_analysis(context)

    try:
        return _generate_with_anthropic(context, model)
    except Exception as exc:
        # If the network, model, or key fails, the dashboard still returns a usable result.
        # The error is attached to revenue so it is visible during development.
        fallback = generate_fallback_analysis(context)
        fallback["revenue"] += f" Claude was unavailable, so this demo is showing local fallback commentary. Error: {exc}"
        return fallback

    return generate_fallback_analysis(context)


def generate_ai_summary(
    context: dict[str, Any],
    model: str = "claude-sonnet-4-20250514",
    enabled: bool = True,
) -> dict[str, str]:
    # Backwards-compatible alias in case older code imports this name.
    return generate_ai_analysis(context=context, model=model, enabled=enabled)


def _generate_with_anthropic(context: dict[str, Any], model: str) -> dict[str, str]:
    # Sends the KPI context to Claude and parses the JSON response.
    # This is a one-shot analysis call, not a chat history loop.
    api_key = _get_anthropic_api_key()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    from anthropic import Anthropic

    # Initialize the Anthropic client using the API key from .env or Streamlit secrets.
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=700,
        temperature=0.2,
        system=SYSTEM_INSTRUCTIONS,
        messages=[{"role": "user", "content": _build_prompt(context)}],
    )
    text = response.content[0].text
    return _parse_json_response(text)


def has_anthropic_api_key() -> bool:
    # Used by the UI to decide whether the button should say Claude or Demo.
    return bool(_get_anthropic_api_key())


def _get_anthropic_api_key() -> str | None:
    # Supports both names because some local projects use CLAUDE_API_KEY.
    # Streamlit secrets are supported for hosted deployments.
    env_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if env_key:
        return env_key

    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("CLAUDE_API_KEY")
    except Exception:
        return None


def _build_prompt(context: dict[str, Any]) -> str:
    # Builds the user message with the compact KPI history.
    # The context is already aggregated, so no raw customer or order-level data is sent.
    return f"""
Analyze this KPI history and write executive commentary.

Rules:
- Tie commentary to the latest month and recent trend.
- Flag risks or opportunities when the data supports them.
- Do not invent segment-level causes unless a segment is present.
- Include an ai_summary paragraph that synthesizes the metrics and recommends practical next steps.
- Return valid JSON only.

KPI context:
{json.dumps(context, indent=2, default=str)}
""".strip()


def _parse_json_response(text: str) -> dict[str, str]:
    # Claude should return JSON, but this also handles fenced code blocks.
    # It also filters out unexpected keys so the UI only sees fields it knows how to render.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    parsed = json.loads(cleaned)
    parsed_summary = {
        key: _stringify_summary_value(value)
        for key, value in parsed.items()
        if key in {"revenue", "orders", "profit", "profit_margin", "ai_summary", "total_summary"}
    }
    # Keeps older total_summary responses working after the UI rename.
    if "total_summary" in parsed_summary and "ai_summary" not in parsed_summary:
        parsed_summary["ai_summary"] = parsed_summary.pop("total_summary")
    return parsed_summary


def _stringify_summary_value(value: Any) -> str:
    # Keeps weird model output from rendering as one character per line.
    # If Claude returns a list by mistake, collapse it into one paragraph.
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return " ".join(str(item).strip() for item in value.values() if str(item).strip())
    return str(value).strip()


def generate_fallback_analysis(context: dict[str, Any]) -> dict[str, str]:
    # Local deterministic commentary used when Claude is off or unavailable.
    # The wording is fixed, but the values and trend directions still come from the active dataset.
    history = context["history"]
    latest = history[-1]
    previous = history[-2] if len(history) > 1 else latest

    revenue_change = _pct_change(latest["revenue"], previous["revenue"])
    orders_change = _pct_change(latest["orders"], previous["orders"])
    profit_change = _pct_change(latest["profit"], previous["profit"])
    margin_change = _point_change(latest["profit_margin"], previous["profit_margin"])
    revenue_direction = "grew" if revenue_change >= 0 else "declined"
    profit_direction = "improved" if profit_change >= 0 else "declined"
    margin_direction = "expanded" if margin_change >= 0 else "compressed"
    return {
        "revenue": (
            f"Revenue closed {latest['period']} at ${latest['revenue']:,.0f}, "
            f"{_direction_phrase(revenue_change)} versus the prior month. "
            "This gives leadership a quick read on whether recent demand is expanding or softening."
        ),
        "orders": (
            f"Order volume closed at {latest['orders']:,}, "
            f"{_direction_phrase(orders_change)} versus the prior month. "
            "This helps separate demand changes from pricing or mix changes in the revenue trend."
        ),
        "profit": (
            f"Profit closed {latest['period']} at ${latest['profit']:,.0f}, "
            f"{_direction_phrase(profit_change)} versus the prior month. "
            "Leadership should compare this movement with revenue to understand whether growth is translating into margin dollars."
        ),
        "profit_margin": (
            f"Profit margin finished at {latest['profit_margin']:.1%}, "
            f"{_point_phrase(margin_change)} from the prior month. "
            "A margin decline may point to heavier discounting or less favorable category mix, while improvement suggests healthier commercial efficiency."
        ),
        "ai_summary": (
            f"Overall, revenue {revenue_direction}, profit {profit_direction}, and margin {margin_direction} in {latest['period']}. "
            "The next step is to drill into region, category, and segment performance to identify which parts of the business are driving the change. "
            "Leadership should also compare discounting against profit margin to confirm whether growth is being achieved efficiently."
        ),
    }


def generate_fallback_summary(context: dict[str, Any]) -> dict[str, str]:
    # Backwards-compatible alias for the old fallback function name.
    return generate_fallback_analysis(context)


def _pct_change(current: float, previous: float) -> float:
    # Calculates percentage change while avoiding a zero division crash.
    # A zero previous value gets treated as no change for fallback wording.
    if previous == 0:
        return 0
    return (current - previous) / abs(previous)


def _point_change(current: float, previous: float) -> float:
    # Used for percentage-point style movement like profit margin.
    return current - previous


def _direction_phrase(change: float) -> str:
    # Converts a numeric change into plain wording for fallback commentary.
    # Small moves are called roughly flat so the app does not overstate noise.
    if abs(change) < 0.005:
        return "roughly flat"
    direction = "up" if change > 0 else "down"
    return f"{direction} {abs(change):.1%}"


def _point_phrase(change: float) -> str:
    # Formats percentage-point movement for fallback commentary.
    # Used for rates like profit margin where point movement is easier to read.
    if abs(change) < 0.0005:
        return "essentially unchanged"
    direction = "up" if change > 0 else "down"
    return f"{direction} {abs(change) * 100:.1f} percentage points"

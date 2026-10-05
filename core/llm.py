"""Optional Claude integration. Every caller has a rule-based fallback, so the
app works without an API key; these functions return None when Claude is
unavailable, errors, or declines.
"""

from __future__ import annotations

import json
import os

from core.config import LOCALITIES

DEFAULT_MODEL = "claude-opus-5-5"
_BETAS = ["server-side-fallback-2026-07-01"]


def _api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[no-any-return]
    except Exception:
        return None


_CLIENT = None


def _client():
    """Build the client once a key is available; a missing key is re-checked on every call."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    key = _api_key()
    if not key:
        return None
    try:
        import anthropic

        _CLIENT = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=1)
    except Exception:
        return None
    return _CLIENT


def is_enabled() -> bool:
    return _client() is not None


def model_name() -> str:
    return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


def _call(system: str, messages: list[dict], max_tokens: int, output_schema: dict | None = None) -> str | None:
    client = _client()
    if client is None:
        return None
    import anthropic

    output_config: dict = {"effort": "low"}
    if output_schema:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}
    try:
        response = client.beta.messages.create(
            model=model_name(),
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
            betas=_BETAS,
            # Server-side refusal fallback: if the model declines, the API reroutes in the same call
            fallbacks="default",
        )
    except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError):
        return None
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


_EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "locality": {"type": ["string", "null"], "enum": [*LOCALITIES, None]},
        "city": {"type": ["string", "null"], "enum": ["Bangalore", "Mumbai", "Pune", None]},
        "area_sqft": {"type": ["number", "null"]},
        "bhk": {"type": ["integer", "null"]},
        "bathrooms": {"type": ["integer", "null"]},
        "property_type": {"type": ["string", "null"], "enum": ["Apartment", "Villa", "Independent House", None]},
        "furnishing": {"type": ["string", "null"], "enum": ["Unfurnished", "Semi-Furnished", "Fully Furnished", None]},
        "parking": {"type": ["integer", "null"]},
        "property_age": {"type": ["integer", "null"]},
        "floor": {"type": ["integer", "null"]},
        "total_floors": {"type": ["integer", "null"]},
        "has_gym": {"type": ["boolean", "null"]},
        "has_pool": {"type": ["boolean", "null"]},
        "has_lift": {"type": ["boolean", "null"]},
        "metro_distance_km": {"type": ["number", "null"]},
    },
    "required": [
        "locality", "city", "area_sqft", "bhk", "bathrooms", "property_type", "furnishing", "parking",
        "property_age", "floor", "total_floors", "has_gym", "has_pool", "has_lift", "metro_distance_km",
    ],
}

_EXTRACT_SYSTEM = (
    "You extract residential property attributes from a buyer's message for a price estimator "
    "covering Bangalore, Mumbai and Pune. Map the place they mention to the closest supported locality "
    "if it is clearly the same area (for example 'Bandra' is 'Bandra West'); otherwise leave locality null. "
    "Use null for anything the message does not state. 'Brand new' or 'ready to move' means property_age 0. "
    "In Indian usage 'house' alone usually just means a home, so only set 'Independent House' when stated. "
    "Convert '1.2k sqft' to 1200."
)


def extract_fields(text: str) -> tuple[dict, str | None] | None:
    """Returns (fields without nulls, city) or None if Claude is unavailable."""
    raw = _call(_EXTRACT_SYSTEM, [{"role": "user", "content": text}], 1024, _EXTRACT_SCHEMA)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    city = data.pop("city", None)
    fields = {k: v for k, v in data.items() if v is not None}
    return fields, city


_SUPPORT_SYSTEM = (
    "You are the customer support assistant for PropertyLens, a property price estimation app for "
    "Bangalore, Mumbai and Pune. Answer using ONLY the knowledge base excerpts provided in the user turn. "
    "If the excerpts do not answer the question, reply exactly with NO_ANSWER and nothing else. "
    "Be concise (under 150 words), use plain text with short paragraphs or simple dashes for lists, "
    "and never use emojis. Do not invent figures, policies or contact details."
)


def answer_support(question: str, excerpts: list[tuple[str, str]], history: list[dict] | None = None) -> str | None:
    """Grounded support answer, or None when unavailable or the KB does not cover it."""
    kb = "\n\n".join(f"<excerpt title=\"{title}\">\n{body}\n</excerpt>" for title, body in excerpts)
    messages = list(history or [])[-6:]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({"role": "user", "content": f"<knowledge_base>\n{kb}\n</knowledge_base>\n\nQuestion: {question}"})
    text = _call(_SUPPORT_SYSTEM, messages, 1024)
    if text is None or "NO_ANSWER" in text:
        return None
    return text

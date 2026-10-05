"""Shared Streamlit helpers: styling, cached resources, formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from core.config import DATA_PATH, METRICS_PATH, POINT_MODEL_PATH, TEST_PREDICTIONS_PATH

ACCENT = "#2563EB"
TEAL = "#0D9488"
GREEN = "#059669"
RED = "#DC2626"
SLATE = "#64748B"
CITY_COLORS = {"Bangalore": "#2563EB", "Mumbai": "#0D9488", "Pune": "#D97706"}

_template = go.layout.Template(pio.templates["plotly_white"])
_template.layout.font = dict(family="Inter, system-ui, sans-serif", color="#334155", size=13)
_template.layout.colorway = [ACCENT, TEAL, "#D97706", "#7C3AED", "#DB2777", SLATE]
_template.layout.margin = dict(l=10, r=10, t=40, b=10)
_template.layout.title = dict(font=dict(size=15, color="#0F172A"))
pio.templates["lens"] = _template
pio.templates.default = "lens"

_CSS_PATH = Path(__file__).parent / "styles.css"
ASSETS = Path(__file__).parent / "assets"


def apply_style() -> None:
    st.html(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>")


def sidebar_brand(llm_on: bool) -> None:
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "mark.svg"))
    with st.sidebar:
        dot = "" if llm_on else " off"
        mode = "Claude-enhanced assistants" if llm_on else "Rule-based assistants"
        st.html(
            f"""<div class="lens-side-card"><b>3 cities, 30 localities</b><br>
Bangalore, Mumbai and Pune price estimates with calibrated ranges.
<div class="lens-status"><span class="lens-dot{dot}"></span>{mode}</div></div>"""
        )


def icon(name: str) -> str:
    return f'<span class="ms">{name}</span>'


def page_header(title: str, subtitle: str = "", eyebrow: str = "", icon_name: str = "") -> None:
    eb = f'<div class="lens-eyebrow">{icon(icon_name) if icon_name else ""}{eyebrow}</div>' if eyebrow else ""
    sub = f'<p class="lens-sub">{subtitle}</p>' if subtitle else ""
    st.html(f'<div class="lens-header">{eb}<div class="lens-title">{title}</div>{sub}</div>')


def section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="s">{subtitle}</div>' if subtitle else ""
    st.html(f'<div class="lens-section"><div class="t">{title}</div>{sub}</div>')


def footer() -> None:
    st.html(
        '<div class="lens-footer"><span>PropertyLens. Indicative estimates, not a certified valuation.</span>'
        "<span>Synthetic data calibrated to 2024-25 locality price levels</span></div>"
    )


def fmt_lakhs(v: float) -> str:
    return f"₹{v / 100:,.2f} Cr" if v >= 100 else f"₹{v:,.1f} L"


def fmt_rupees(v: float) -> str:
    """Indian-style short format for rupee amounts."""
    if v >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if v >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


def card(label: str, value: str, sub: str = "", icon_name: str = "", tone: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    ic = f'<span class="lens-icon">{icon(icon_name)}</span>' if icon_name else ""
    st.html(
        f'<div class="lens-card stat tone-{tone}"><div class="top"><div class="label">{label}</div>{ic}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def feature_card(icon_name: str, title: str, body: str, tone: str = "") -> None:
    st.html(
        f'<div class="lens-card lens-feature tone-{tone}"><span class="lens-icon">{icon(icon_name)}</span>'
        f"<h4>{title}</h4><p>{body}</p></div>"
    )


def result_card(price: float, low: float, high: float, detail: str) -> None:
    """Hero estimate with a range bar placing the point estimate between P10 and P90."""
    span = max(high - low, 1e-9)
    lo_edge, hi_edge = low - span * 0.25, high + span * 0.25
    width = hi_edge - lo_edge
    fill_l = (low - lo_edge) / width * 100
    fill_w = (high - low) / width * 100
    pin = (price - lo_edge) / width * 100
    st.html(
        f"""<div class="lens-result">
<div class="k">Most likely price</div>
<div class="v">{fmt_lakhs(price)}</div>
<div class="d">{detail}</div>
<div class="lens-range">
  <div class="track"><div class="fill" style="left:{fill_l:.1f}%;width:{fill_w:.1f}%"></div><div class="pin" style="left:{pin:.1f}%"></div></div>
  <div class="ends"><span>Low {fmt_lakhs(low)}</span><span>80% likely range</span><span>High {fmt_lakhs(high)}</span></div>
</div></div>"""
    )


def pills(items: list[str]) -> None:
    st.html("".join(f'<span class="lens-pill">{i}</span>' for i in items))


def artefacts_ready() -> bool:
    return DATA_PATH.exists() and POINT_MODEL_PATH.exists() and METRICS_PATH.exists()


def require_artefacts() -> None:
    if not artefacts_ready():
        st.error(
            "The dataset or trained model is missing. Run `python -m core.data_gen` and then "
            "`python -m core.train` from the project folder, then reload this page."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Price_per_sqft"] = (df["Price_Lakhs"] * 1e5 / df["Area_sqft"]).round(0)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PREDICTIONS_PATH)


@st.cache_resource(show_spinner="Loading model")
def get_predictor():
    from core.predict import Predictor

    return Predictor()


@st.cache_resource(show_spinner=False)
def get_kb():
    from core.support_kb import KnowledgeBase

    return KnowledgeBase()


def drivers_chart(drivers: dict[str, float], height: int = 320) -> go.Figure:
    items = sorted(drivers.items(), key=lambda kv: abs(kv[1]))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=[GREEN if v >= 0 else RED for v in vals],
            text=[f"{v:+.1f}%" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        title="What moved the price (vs average listing)",
        xaxis_title="Effect on price (%)",
        xaxis=dict(zeroline=True, zerolinecolor="#94A3B8"),
        showlegend=False,
    )
    return fig

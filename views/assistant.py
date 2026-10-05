import streamlit as st

from core import llm, tickets
from core.assistant import respond
from core.features import FEATURE_LABELS
from ui.components import fmt_lakhs, footer, get_predictor, page_header, pills, require_artefacts

require_artefacts()
predictor = get_predictor()

QUICK_PROMPTS = [
    "I want to buy a 2 BHK house with 900 sq.ft.",
    "3 BHK in Baner, around 1.4k sqft, 5 years old",
    "Brand new villa in Whitefield with a pool",
    "2 bed flat in Bandra, 12th floor, no parking",
    "How accurate are your estimates?",
]
SLOT_LABELS = {
    "locality": "Locality",
    "area_sqft": "Area",
    "bhk": "BHK",
    "bathrooms": "Bathrooms",
    "property_type": "Type",
    "furnishing": "Furnishing",
    "parking": "Parking",
    "property_age": "Age",
    "floor": "Floor",
    "total_floors": "Total floors",
    "has_gym": "Gym",
    "has_pool": "Pool",
    "has_lift": "Lift",
    "metro_distance_km": "Metro km",
}

st.session_state.setdefault("pa_messages", [])
st.session_state.setdefault("pa_slots", {})

page_header(
    "Price Assistant",
    "Describe a property in your own words. I remember details across messages, so you can refine as you go. "
    + ("Language understanding by Claude." if llm.is_enabled() else "Language understanding by built-in rules."),
    "Conversational estimate",
    "forum",
)
known_col, side = st.columns([4, 1], vertical_alignment="center")
if side.button("New conversation", icon=":material/restart_alt:", width="stretch"):
    st.session_state["pa_messages"] = []
    st.session_state["pa_slots"] = {}
    st.rerun()

slots = st.session_state["pa_slots"]
known = [f"{SLOT_LABELS[k]}: {v}" for k, v in slots.items() if k in SLOT_LABELS]
if slots.get("_city") and "locality" not in slots:
    known.insert(0, f"City: {slots['_city']}")
with known_col:
    if known:
        pills(known)
    else:
        st.caption("Details you mention will appear here.")


def _render_estimate(msg: dict, idx: int) -> None:
    est = msg["estimate"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Most likely", fmt_lakhs(est["price"]))
    c2.metric("Range", f"{fmt_lakhs(est['low'])} - {fmt_lakhs(est['high'])}")
    c3.metric("Per sq.ft", f"₹{est['ppsf']:,.0f}")
    if st.button("Open full analysis", key=f"open_{idx}", icon=":material/open_in_new:"):
        st.session_state["prefill"] = msg["property"]
        st.switch_page("views/predict.py")


def _handle(prompt: str) -> None:
    msgs = st.session_state["pa_messages"]
    msgs.append({"role": "user", "content": prompt})
    reply = respond(prompt, st.session_state["pa_slots"], predictor)
    st.session_state["pa_slots"] = reply.slots
    entry: dict = {"role": "assistant", "content": reply.text, "question": prompt}
    if reply.estimate and reply.property:
        entry["estimate"] = {
            "price": reply.estimate.price_lakhs,
            "low": reply.estimate.low_lakhs,
            "high": reply.estimate.high_lakhs,
            "ppsf": reply.estimate.price_per_sqft,
        }
        entry["property"] = reply.property.model_dump()
    msgs.append(entry)


if not st.session_state["pa_messages"]:
    with st.chat_message("assistant", avatar=":material/real_estate_agent:"):
        st.markdown(
            "Hello. Tell me about a property in Bangalore, Mumbai or Pune and I will estimate its price. "
            "Try one of these to start:"
        )
        with st.container(key="chips_pa", horizontal=True):
            for qp in QUICK_PROMPTS:
                if st.button(qp, key=f"qp_{qp}"):
                    _handle(qp)
                    st.rerun()

for i, msg in enumerate(st.session_state["pa_messages"]):
    avatar = ":material/real_estate_agent:" if msg["role"] == "assistant" else ":material/person:"
    box_key = f"{'umsg' if msg['role'] == 'user' else 'amsg'}_{i}"
    with st.container(key=box_key), st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("estimate"):
            _render_estimate(msg, i)
        if msg["role"] == "assistant":
            fb = st.feedback("thumbs", key=f"pa_fb_{i}")
            if fb is not None and not msg.get("fb_logged"):
                tickets.record_feedback("price", msg.get("question", ""), msg["content"], bool(fb))
                msg["fb_logged"] = True

if prompt := st.chat_input("e.g. 3 BHK flat in Powai, 1200 sq.ft, semi-furnished"):
    _handle(prompt)
    st.rerun()

with st.sidebar:
    st.caption("Supported details: " + ", ".join(FEATURE_LABELS[k] for k in ("Locality", "Area_sqft", "BHK", "Bathrooms", "Property_Type", "Property_Age", "Floor", "Parking", "Furnishing")) + ", gym, pool, lift, metro distance.")

footer()

import streamlit as st

from core.config import CITIES, LOCALITIES
from core.schema import PropertyInput
from ui.components import (
    CITY_COLORS,
    card,
    feature_card,
    fmt_lakhs,
    footer,
    get_predictor,
    icon,
    load_data,
    load_metrics,
    require_artefacts,
    section,
)

require_artefacts()
df = load_data()
metrics = load_metrics()

# The hero's sample card is a live estimate from the production model
sample = PropertyInput(locality="Baner", area_sqft=1450, bhk=3, bathrooms=3, property_age=4)
sample_est = get_predictor().estimate(sample)
top3 = sample_est.top_drivers(3)
scale = max(abs(v) for _, v in top3) or 1
bars = "".join(
    f'<div class="bar-row"><span class="lbl">{name}</span><div class="bar"><i style="width:{abs(v) / scale * 85:.0f}%;'
    f'background:{"#059669" if v >= 0 else "#DC2626"}"></i></div><span>{v:+.0f}%</span></div>'
    for name, v in top3
)

st.html(
    f"""<div class="lens-hero-wrap">
<div>
  <div class="lens-eyebrow">{icon("auto_awesome")}Bangalore, Mumbai and Pune</div>
  <h1>Know what a home is worth <span>before you negotiate</span></h1>
  <p>PropertyLens estimates a fair price for any apartment, villa or independent house, shows the range it is likely
  to fall in, and explains exactly which factors moved the number.</p>
  <div class="lens-checks">
    <span>{icon("check_circle")}Calibrated price range</span>
    <span>{icon("check_circle")}Explains every estimate</span>
    <span>{icon("check_circle")}Similar listings to compare</span>
    <span>{icon("check_circle")}EMI and stamp duty built in</span>
  </div>
</div>
<div class="lens-mock">
  <div class="row"><div><div class="addr">3 BHK Apartment, Baner</div><div class="meta">Pune &middot; 1,450 sq.ft &middot; 4 years old</div></div>
  <span class="tag">Sample estimate</span></div>
  <div class="big">{fmt_lakhs(sample_est.price_lakhs)}</div>
  <div class="meta">Likely range {fmt_lakhs(sample_est.low_lakhs)} to {fmt_lakhs(sample_est.high_lakhs)} &middot; ₹{sample_est.price_per_sqft:,.0f} per sq.ft</div>
  <div style="margin-top:0.9rem;font-size:0.75rem;font-weight:600;color:#64748B;letter-spacing:.06em;text-transform:uppercase">What moved the price</div>
  {bars}
</div>
</div>
</div>"""
)

b1, b2, b3, _ = st.columns([1.1, 1.2, 1.1, 2])
if b1.button("Get an estimate", type="primary", width="stretch", icon=":material/calculate:"):
    st.switch_page("views/predict.py")
if b2.button("Ask the Price Assistant", width="stretch", icon=":material/forum:"):
    st.switch_page("views/assistant.py")
if b3.button("Compare localities", width="stretch", icon=":material/map:"):
    st.switch_page("views/compare.py")

st.write("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Listings analysed", f"{len(df):,}", f"{len(CITIES)} cities, {len(LOCALITIES)} localities", "home_work")
with c2:
    card("Typical error", f"{metrics['production_test_mape']:.1f}%", "MAPE on held-out test data", "target", "teal")
with c3:
    card("Range reliability", f"{metrics['interval_coverage_pct']:.0f}%", "of test prices inside the range", "verified", "green")
with c4:
    card("Model fit", f"R² {metrics['production_test_r2']:.3f}", f"{metrics['production_model']}, 5 models benchmarked", "insights", "violet")

section("Everything you need to price a home", "From a quick question to a full purchase plan.")
f1, f2, f3 = st.columns(3)
with f1:
    feature_card("forum", "Ask in plain English", "Type something like \"3 BHK in Powai, 1200 sq.ft, semi-furnished\". "
                 "The Price Assistant remembers details as you refine them.")
with f2:
    feature_card("query_stats", "See why, not just what", "Every estimate shows how location, size, age, floor and "
                 "amenities pushed the price up or down versus an average listing.", "teal")
with f3:
    feature_card("account_balance", "Plan the purchase", "EMI, affordability, stamp duty and registration for your "
                 "city, plus the five most similar listings to negotiate with.", "amber")

section("Market snapshot", "Median values across all listings in each city.")
cols = st.columns(3)
for col, city in zip(cols, CITIES):
    sub = df[df["City"] == city]
    by_loc = sub.groupby("Locality")["Price_per_sqft"].median().sort_values()
    with col:
        st.html(
            f"""<div class="lens-card lens-city" style="border-top-color:{CITY_COLORS[city]}">
<div class="name">{city}</div>
<div class="kv"><span>Median price</span><b>{fmt_lakhs(sub["Price_Lakhs"].median())}</b></div>
<div class="kv"><span>Median per sq.ft</span><b>₹{sub["Price_per_sqft"].median():,.0f}</b></div>
<div class="kv"><span>Most expensive</span><b>{by_loc.index[-1]}</b></div>
<div class="kv"><span>Most affordable</span><b>{by_loc.index[0]}</b></div>
</div>"""
        )

section("How it works")
s1, s2, s3, s4 = st.columns(4)
steps = [
    ("edit_note", "Describe", "Locality, size, bedrooms, age, floor, furnishing, parking and amenities."),
    ("model_training", "Estimate", "A LightGBM model trained on 6,000 listings predicts the most likely price."),
    ("tune", "Calibrate", "A range that contains the real price about 80% of the time, not a single guess."),
    ("handshake", "Decide", "Compare with similar listings, check your EMI and budget, then negotiate."),
]
for col, (ic, title, body), tone in zip((s1, s2, s3, s4), steps, ("", "teal", "violet", "green")):
    with col:
        feature_card(ic, title, body, tone)

st.write("")
st.info(
    "About the data: this release uses a synthetic dataset calibrated to approximate 2024-25 locality price "
    "levels. Estimates are indicative and are not a certified valuation.",
    icon=":material/info:",
)
footer()

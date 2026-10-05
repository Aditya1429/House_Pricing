import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import finance
from core.config import CITIES, FURNISHING, LOCALITIES, PROPERTY_TYPES, localities_for
from core.schema import PropertyInput
from ui.components import (
    ACCENT,
    card,
    drivers_chart,
    fmt_lakhs,
    fmt_rupees,
    footer,
    get_predictor,
    page_header,
    require_artefacts,
    result_card,
    section,
)

require_artefacts()
predictor = get_predictor()

page_header(
    "Property price estimate",
    "Describe the property on the left. The estimate, range and explanation update instantly as you change details.",
    "Predict",
    "calculate",
)

DEFAULTS = {
    "city": "Bangalore", "locality": "Whitefield", "property_type": "Apartment", "area_sqft": 1200,
    "bhk": 2, "bathrooms": 2, "property_age": 5, "furnishing": "Semi-Furnished", "floor": 4,
    "total_floors": 12, "parking": 1, "has_gym": False, "has_pool": False, "has_lift": True,
}

# Values handed over from the Price Assistant
prefill: dict = st.session_state.pop("prefill", None) or {}
if prefill:
    for key, value in prefill.items():
        if key in DEFAULTS:
            st.session_state[f"f_{key}"] = value
    st.session_state["f_city"] = LOCALITIES[prefill["locality"]].city
    st.session_state["f_metro"] = float(prefill.get("metro_distance_km") or LOCALITIES[prefill["locality"]].metro_km)
    st.session_state["_metro_for"] = prefill["locality"]
    st.toast("Loaded the property from the Price Assistant")
for key, value in DEFAULTS.items():
    st.session_state.setdefault(f"f_{key}", value)
st.session_state["f_area_sqft"] = int(st.session_state["f_area_sqft"])

form, result = st.columns([2, 3], gap="large")

with form, st.container(key="card_form"):
    st.markdown("### Property details")
    city = st.selectbox("City", CITIES, key="f_city")
    locs = localities_for(city)
    if st.session_state.get("f_locality") not in locs:
        st.session_state["f_locality"] = locs[0]
    locality = st.selectbox("Locality", locs, key="f_locality")
    ptype = st.radio("Property type", PROPERTY_TYPES, horizontal=True, key="f_property_type")
    area = st.number_input("Area (sq.ft)", 250, 10000, step=50, key="f_area_sqft")
    c1, c2 = st.columns(2)
    bhk = c1.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5, 6], key="f_bhk")
    baths = c2.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 7], key="f_bathrooms")
    age = st.slider("Property age (years)", 0, 40, key="f_property_age")
    furnishing = st.select_slider("Furnishing", FURNISHING, key="f_furnishing")

    with st.expander("Building, parking and amenities"):
        is_apartment = ptype == "Apartment"
        c1, c2 = st.columns(2)
        floor = c1.number_input("Floor", 0, 80, key="f_floor", disabled=not is_apartment)
        total_floors = c2.number_input("Total floors", 1, 80, key="f_total_floors", disabled=not is_apartment)
        parking = st.radio("Covered parking", [0, 1, 2, 3], horizontal=True, key="f_parking")
        a1, a2, a3 = st.columns(3)
        gym = a1.checkbox("Gym", key="f_has_gym")
        pool = a2.checkbox("Pool", key="f_has_pool")
        lift = a3.checkbox("Lift", key="f_has_lift")
        # Reset the metro distance to the locality's typical value when the locality changes
        if st.session_state.get("_metro_for") != locality or "f_metro" not in st.session_state:
            st.session_state["f_metro"] = float(LOCALITIES[locality].metro_km)
            st.session_state["_metro_for"] = locality
        metro = st.slider(
            "Distance to metro / station (km)", 0.1, 25.0, step=0.1, key="f_metro",
            help="Defaults to the typical distance for the locality.",
        )

try:
    prop = PropertyInput(
        locality=locality,
        area_sqft=area,
        bhk=bhk,
        bathrooms=baths,
        property_type=ptype,
        floor=floor,
        total_floors=max(total_floors, floor),
        parking=parking,
        property_age=age,
        furnishing=furnishing,
        has_gym=gym,
        has_pool=pool,
        has_lift=lift,
        metro_distance_km=metro,
    )
except ValueError as exc:
    result.error(f"Please check the inputs: {exc}")
    st.stop()

est = predictor.estimate(prop)

with result:
    result_card(
        est.price_lakhs,
        est.low_lakhs,
        est.high_lakhs,
        f"{prop.bhk} BHK {prop.property_type.lower()} &middot; {prop.area_sqft:,.0f} sq.ft &middot; {prop.locality}, {prop.city}",
    )
    st.write("")
    k1, k2, k3 = st.columns(3)
    with k1:
        card("Per sq.ft", f"₹{est.price_per_sqft:,.0f}", "estimated rate", "straighten")
    loc_median = predictor.data.loc[predictor.data["Locality"] == locality]
    loc_ppsf = float((loc_median["Price_Lakhs"] * 1e5 / loc_median["Area_sqft"]).median())
    with k2:
        diff = (est.price_per_sqft / loc_ppsf - 1) * 100
        card("vs locality", f"{diff:+.0f}%", f"median ₹{loc_ppsf:,.0f}/sq.ft", "compare_arrows", "teal")
    costs = finance.purchase_costs(est.price_lakhs * 1e5, prop.city)
    with k3:
        card("Purchase costs", fmt_rupees(costs["total"]), "stamp duty + registration", "receipt_long", "amber")

    st.write("")
    top = est.top_drivers(3)
    st.html(
        '<div class="lens-note" style="margin-bottom:.4rem">Biggest price drivers</div>'
        + "".join(f'<span class="lens-pill">{n} {v:+.0f}%</span>' for n, v in top)
    )
    if st.button("Save to comparison list", icon=":material/bookmark_add:"):
        saved = st.session_state.setdefault("saved", [])
        saved.append(
            {
                "City": prop.city,
                "Locality": prop.locality,
                "Type": prop.property_type,
                "BHK": prop.bhk,
                "Area_sqft": prop.area_sqft,
                "Age": prop.property_age,
                "Furnishing": prop.furnishing,
                "Estimate_Lakhs": round(est.price_lakhs, 2),
                "Low_Lakhs": round(est.low_lakhs, 2),
                "High_Lakhs": round(est.high_lakhs, 2),
                "Per_sqft": round(est.price_per_sqft),
            }
        )
        st.toast(f"Saved. {len(saved)} propert{'y' if len(saved) == 1 else 'ies'} in your list.")

section("Detailed analysis", "Explanation, comparable listings, sensitivity and financing for this property.")
tab_why, tab_comps, tab_whatif, tab_money, tab_saved = st.tabs(
    ["Why this price", "Comparable listings", "What if", "EMI and affordability", "Saved comparisons"]
)

with tab_why:
    st.plotly_chart(drivers_chart(est.drivers), width="stretch")
    st.caption(
        f"Effects are relative to the average listing in the dataset ({fmt_lakhs(est.baseline_lakhs)}), computed "
        "from the model's SHAP contributions. They multiply together to give the estimate."
    )

with tab_comps:
    comps = predictor.comparables(prop)
    st.dataframe(
        comps,
        hide_index=True,
        width="stretch",
        column_config={
            "Price_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
            "Price_per_sqft": st.column_config.NumberColumn("₹ / sq.ft", format="%d"),
            "Area_sqft": st.column_config.NumberColumn("Area (sq.ft)", format="%d"),
            "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100, format="%d"),
        },
    )
    st.caption("The five listings most similar to this property in the same city.")

with tab_whatif:
    curve = predictor.area_curve(prop)
    fig = go.Figure(go.Scatter(x=curve["Area_sqft"], y=curve["Price_Lakhs"], mode="lines", line=dict(color=ACCENT, width=3), name="Estimate"))
    fig.add_trace(go.Scatter(x=[prop.area_sqft], y=[est.price_lakhs], mode="markers", marker=dict(size=12, color="#0F172A"), name="This property"))
    fig.update_layout(height=360, title="Estimated price as area changes", xaxis_title="Area (sq.ft)", yaxis_title="Price (₹ Lakhs)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Everything except area is held fixed.")

with tab_money:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**EMI calculator**")
        price_rs = est.price_lakhs * 1e5
        down_pct = st.slider("Down payment (%)", 10, 90, 20, 5)
        rate = st.number_input("Interest rate (% per year)", 6.0, 15.0, 8.75, 0.05)
        years = st.slider("Tenure (years)", 5, 30, 20)
        loan = price_rs * (1 - down_pct / 100)
        monthly = finance.emi(loan, rate, years)
        total_interest = monthly * years * 12 - loan
        m1, m2 = st.columns(2)
        with m1:
            card("Monthly EMI", fmt_rupees(monthly))
        with m2:
            card("Loan amount", fmt_rupees(loan))
        st.caption(
            f"Total interest over {years} years: {fmt_rupees(total_interest)}. Upfront cash needed: "
            f"{fmt_rupees(price_rs - loan + costs['total'])} (down payment plus stamp duty and registration). "
            f"{costs['note']}."
        )
    with c2:
        st.markdown("**What can I afford?**")
        income = st.number_input("Net monthly income (₹)", 10000, 10_000_000, 150000, 5000)
        existing = st.number_input("Existing EMIs (₹ per month)", 0, 5_000_000, 0, 1000)
        savings = st.number_input("Savings for down payment (₹)", 0, 500_000_000, 2_000_000, 100000)
        aff = finance.affordability(income, existing, savings, rate, years)
        card("Maximum property budget", fmt_rupees(aff["max_price"]), f"Max EMI {fmt_rupees(aff['max_emi'])} at 50% of income")
        if aff["max_price"] >= price_rs:
            st.success("This property is within your estimated budget.", icon=":material/check_circle:")
        else:
            st.warning(f"This property is {fmt_rupees(price_rs - aff['max_price'])} above your estimated budget.", icon=":material/warning:")

with tab_saved:
    saved = st.session_state.get("saved", [])
    if not saved:
        st.caption("Use 'Save to comparison list' to collect properties here and compare them side by side.")
    else:
        saved_df = pd.DataFrame(saved)
        st.dataframe(saved_df, hide_index=True, width="stretch")
        d1, d2 = st.columns([1, 5])
        d1.download_button("Download CSV", saved_df.to_csv(index=False), "propertylens_comparison.csv", "text/csv")
        if d2.button("Clear list"):
            st.session_state["saved"] = []
            st.rerun()

footer()

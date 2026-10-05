import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CITIES, LOCALITIES
from ui.components import CITY_COLORS, fmt_lakhs, footer, load_data, page_header, require_artefacts, section

require_artefacts()
df = load_data()

page_header(
    "Compare localities",
    "See how price per sq.ft varies across neighbourhoods, on a map and side by side across cities.",
    "Insights",
    "map",
)

stats = (
    df.groupby(["City", "Locality"])
    .agg(
        Median_ppsf=("Price_per_sqft", "median"),
        Median_price=("Price_Lakhs", "median"),
        Listings=("Listing_ID", "count"),
        Median_age=("Property_Age", "median"),
        Metro_km=("Metro_Distance_km", "median"),
    )
    .reset_index()
)
stats["lat"] = stats["Locality"].map(lambda l: LOCALITIES[l].lat)
stats["lon"] = stats["Locality"].map(lambda l: LOCALITIES[l].lon)

city = st.segmented_control("City", CITIES, default="Bangalore", key="cmp_city") or "Bangalore"
cs = stats[stats["City"] == city].sort_values("Median_ppsf", ascending=False)

left, right = st.columns([3, 2], gap="large")
with left:
    fig = px.bar(
        cs.sort_values("Median_ppsf"),
        x="Median_ppsf",
        y="Locality",
        orientation="h",
        color_discrete_sequence=[CITY_COLORS[city]],
        text=cs.sort_values("Median_ppsf")["Median_ppsf"].map(lambda v: f"₹{v:,.0f}"),
        labels={"Median_ppsf": "Median ₹ per sq.ft", "Locality": ""},
        title=f"Median price per sq.ft in {city}",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=430)
    st.plotly_chart(fig, width="stretch")
with right:
    fig = px.scatter_map(
        cs,
        lat="lat",
        lon="lon",
        size="Median_ppsf",
        color="Median_ppsf",
        color_continuous_scale="Blues",
        hover_name="Locality",
        hover_data={"Median_ppsf": ":,.0f", "lat": False, "lon": False, "Listings": True},
        size_max=28,
        zoom=10,
        map_style="carto-positron",
        title="Price map",
    )
    fig.update_layout(height=430, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

st.dataframe(
    cs[["Locality", "Median_ppsf", "Median_price", "Listings", "Median_age", "Metro_km"]],
    hide_index=True,
    width="stretch",
    column_config={
        "Median_ppsf": st.column_config.NumberColumn("Median ₹/sq.ft", format="%d"),
        "Median_price": st.column_config.NumberColumn("Median price (₹ L)", format="%.1f"),
        "Median_age": st.column_config.NumberColumn("Median age (yrs)", format="%d"),
        "Metro_km": st.column_config.NumberColumn("Median metro distance (km)", format="%.1f"),
    },
)

section("Side by side", "Pick up to four localities from any city to compare typical prices.")
default_pick = ["Whitefield", "Powai", "Baner"]
pick = st.multiselect("Localities", list(LOCALITIES), default=default_pick, max_selections=4)
if pick:
    rows = []
    for loc in pick:
        sub = df[df["Locality"] == loc]
        for bhk in (1, 2, 3):
            b = sub[sub["BHK"] == bhk]
            if len(b):
                rows.append({"Locality": loc, "City": LOCALITIES[loc].city, "BHK": f"{bhk} BHK", "Median price (₹ L)": b["Price_Lakhs"].median()})
    comp = pd.DataFrame(rows)
    fig = px.bar(
        comp,
        x="BHK",
        y="Median price (₹ L)",
        color="Locality",
        barmode="group",
        title="Median price by BHK",
    )
    fig.update_layout(height=380, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    cols = st.columns(len(pick))
    for col, loc in zip(cols, pick):
        s = stats[stats["Locality"] == loc].iloc[0]
        color = CITY_COLORS[LOCALITIES[loc].city]
        with col:
            st.html(
                f"""<div class="lens-card lens-city" style="border-top-color:{color}">
<div class="name">{loc}</div><div class="lens-note">{LOCALITIES[loc].city}</div>
<div class="kv"><span>Per sq.ft</span><b>₹{s.Median_ppsf:,.0f}</b></div>
<div class="kv"><span>Median price</span><b>{fmt_lakhs(s.Median_price)}</b></div>
<div class="kv"><span>Metro distance</span><b>{s.Metro_km:.1f} km</b></div>
</div>"""
            )

footer()

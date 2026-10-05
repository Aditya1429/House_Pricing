import numpy as np
import plotly.express as px
import streamlit as st

from core.config import CITIES, PROPERTY_TYPES, localities_for
from core.features import NUMERIC
from ui.components import CITY_COLORS, footer, load_data, page_header, require_artefacts

require_artefacts()
df = load_data()

page_header(
    "Explore the data",
    "Filter the listings behind the model. Filters apply to every chart and table on this page.",
    "Insights",
    "table_chart",
)

with st.container(key="card_filters"):
    f1, f2, f3, f4 = st.columns(4)
    cities = f1.multiselect("City", CITIES, default=CITIES)
    loc_options = [l for c in cities for l in localities_for(c)]
    locs = f2.multiselect("Locality", loc_options, placeholder="All localities")
    types = f3.multiselect("Property type", PROPERTY_TYPES, default=PROPERTY_TYPES)
    bhks = f4.multiselect("BHK", sorted(df["BHK"].unique()), default=sorted(df["BHK"].unique()))
    pmin, pmax = float(df["Price_Lakhs"].min()), float(df["Price_Lakhs"].max())
    price = st.slider("Price (₹ Lakhs)", pmin, pmax, (pmin, pmax))

mask = (
    df["City"].isin(cities)
    & df["Property_Type"].isin(types)
    & df["BHK"].isin(bhks)
    & df["Price_Lakhs"].between(*price)
)
if locs:
    mask &= df["Locality"].isin(locs)
view = df[mask]
st.markdown(f"**{len(view):,}** of {len(df):,} listings match.")
if view.empty:
    st.stop()

tab_charts, tab_table, tab_dict = st.tabs(["Charts", "Table", "Data dictionary"])

with tab_charts:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(view, x="Price_Lakhs", color="City", nbins=60, log_x=False, opacity=0.75,
                           color_discrete_map=CITY_COLORS, barmode="overlay", title="Price distribution",
                           labels={"Price_Lakhs": "Price (₹ Lakhs)"})
        fig.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        sample = view.sample(min(len(view), 2500), random_state=1)
        fig = px.scatter(sample, x="Area_sqft", y="Price_Lakhs", color="City", color_discrete_map=CITY_COLORS,
                         opacity=0.55, log_y=True,
                         hover_data=["Locality", "BHK", "Property_Age"], title="Price vs area (log scale)",
                         labels={"Area_sqft": "Area (sq.ft)", "Price_Lakhs": "Price (₹ Lakhs)"})
        fig.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        bins = [-1, 2, 5, 10, 15, 20, 60]
        labels = ["0-2", "3-5", "6-10", "11-15", "16-20", "20+"]
        v = view.assign(Age_band=np.asarray(np.digitize(view["Property_Age"], bins[1:-1], right=True)))
        v["Age_band"] = v["Age_band"].map(dict(enumerate(labels)))
        fig = px.box(v, x="Age_band", y="Price_per_sqft", color="City", color_discrete_map=CITY_COLORS,
                     category_orders={"Age_band": labels}, points=False, title="Price per sq.ft by property age",
                     labels={"Age_band": "Age (years)", "Price_per_sqft": "₹ per sq.ft"})
        fig.update_layout(height=380, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        num = view[NUMERIC + ["Price_Lakhs", "Price_per_sqft"]].astype(float)
        corr = num.corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        title="Correlation between numeric features", aspect="auto")
        fig.update_layout(height=380)
        fig.update_traces(textfont_size=9)
        st.plotly_chart(fig, width="stretch")

    by_loc = view.groupby(["City", "Locality"])["Price_per_sqft"].median().reset_index().sort_values("Price_per_sqft")
    fig = px.bar(by_loc, x="Locality", y="Price_per_sqft", color="City", color_discrete_map=CITY_COLORS,
                 title="Median price per sq.ft by locality", labels={"Price_per_sqft": "₹ per sq.ft", "Locality": ""})
    fig.update_layout(height=380, legend_title_text="")
    st.plotly_chart(fig, width="stretch")

with tab_table:
    st.dataframe(view, hide_index=True, width="stretch", height=480)
    st.download_button("Download filtered CSV", view.to_csv(index=False), "propertylens_listings.csv", "text/csv")
    st.markdown("**Summary statistics**")
    st.dataframe(view.describe().T.round(2), width="stretch")

with tab_dict:
    st.table(
        {
            "Column": ["Listing_ID", "City", "Locality", "Property_Type", "Area_sqft", "BHK", "Bathrooms", "Floor",
                       "Total_Floors", "Parking", "Property_Age", "Furnishing", "Has_Gym / Has_Pool / Has_Lift",
                       "Metro_Distance_km", "Price_Lakhs", "Price_per_sqft"],
            "Description": [
                "Unique listing identifier",
                "Bangalore, Mumbai or Pune",
                "One of 10 localities per city",
                "Apartment, Villa or Independent House",
                "Saleable (built-up) area in square feet",
                "Number of bedrooms",
                "Number of bathrooms",
                "Floor of the unit (0 = ground; houses and villas are 0)",
                "Floors in the building",
                "Covered parking spaces",
                "Years since construction",
                "Unfurnished, Semi-Furnished or Fully Furnished",
                "Amenity availability (true / false)",
                "Distance to nearest metro or suburban station",
                "Target: price in ₹ Lakhs (1 Lakh = ₹100,000)",
                "Derived: price per square foot in ₹",
            ],
        }
    )

footer()

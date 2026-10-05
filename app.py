"""PropertyLens entry point. Run: streamlit run app.py"""

import streamlit as st

from core.llm import is_enabled as llm_enabled
from ui.components import ASSETS, apply_style, sidebar_brand

st.set_page_config(
    page_title="PropertyLens",
    page_icon=str(ASSETS / "mark.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_style()

pages = {
    "Estimate": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/predict.py", title="Predict", icon=":material/calculate:", url_path="predict"),
        st.Page("views/assistant.py", title="Price Assistant", icon=":material/forum:", url_path="assistant"),
    ],
    "Insights": [
        st.Page("views/compare.py", title="Compare Localities", icon=":material/map:", url_path="compare"),
        st.Page("views/explore.py", title="Explore Data", icon=":material/table_chart:", url_path="explore"),
        st.Page("views/evaluation.py", title="Model Evaluation", icon=":material/monitoring:", url_path="evaluation"),
    ],
    "Help": [
        st.Page("views/support.py", title="Help & Support", icon=":material/support_agent:", url_path="support"),
    ],
}

nav = st.navigation(pages)
sidebar_brand(llm_enabled())
nav.run()

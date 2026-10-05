import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    ACCENT,
    CITY_COLORS,
    card,
    footer,
    load_metrics,
    load_test_predictions,
    page_header,
    require_artefacts,
    section,
)

require_artefacts()
metrics = load_metrics()
preds = load_test_predictions()

page_header(
    "Model evaluation",
    f"Trained {metrics['trained_at'][:10]} on {metrics['train_rows']:,} listings and tested on {metrics['test_rows']:,} "
    "held-out listings. All models learn log(price).",
    "Insights",
    "monitoring",
)

bench = pd.DataFrame(metrics["benchmarks"]).set_index("model")
prod = bench.loc[metrics["production_model"]]

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Production model", metrics["production_model"], f"Best on CV: {metrics['best_cv_model']}", "workspace_premium")
with c2:
    card("Test MAPE", f"{prod.Test_MAPE:.2f}%", f"CV {prod.CV_MAPE:.2f}% ± {prod.CV_MAPE_std:.2f}", "target", "teal")
with c3:
    card("Test R²", f"{prod.Test_R2:.4f}", f"MAE ₹{prod.Test_MAE:.1f} L", "insights", "violet")
with c4:
    card("Range coverage", f"{metrics['interval_coverage_pct']:.1f}%", "target 80%, calibrated", "verified", "green")

section("Model comparison", "5-fold cross-validation on the training split, then the held-out test split.")
show = bench[["CV_MAPE", "CV_R2", "Test_MAE", "Test_RMSE", "Test_R2", "Test_MAPE"]].sort_values("CV_MAPE")
st.dataframe(
    show,
    width="stretch",
    column_config={
        "CV_MAPE": st.column_config.NumberColumn("CV MAPE (%)", format="%.2f"),
        "CV_R2": st.column_config.NumberColumn("CV R²", format="%.4f"),
        "Test_MAE": st.column_config.NumberColumn("Test MAE (₹ L)", format="%.2f"),
        "Test_RMSE": st.column_config.NumberColumn("Test RMSE (₹ L)", format="%.2f"),
        "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
        "Test_MAPE": st.column_config.NumberColumn("Test MAPE (%)", format="%.2f"),
    },
)
fig = px.bar(show.reset_index(), x="model", y="Test_MAPE", text=show["Test_MAPE"].map(lambda v: f"{v:.1f}%").values,
             color_discrete_sequence=[ACCENT], labels={"model": "", "Test_MAPE": "Test MAPE (%)"},
             title="Typical percentage error by model (lower is better)")
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(height=340)
st.plotly_chart(fig, width="stretch")

section("Diagnostics", "How the selected model's errors are distributed.")
model = st.selectbox("Model", list(bench.index), index=list(bench.index).index(metrics["production_model"]))
p = preds.assign(Predicted=preds[model], Error_pct=(preds[model] / preds["Actual"] - 1) * 100)

c1, c2 = st.columns(2)
with c1:
    lim = [p[["Actual", "Predicted"]].min().min() * 0.9, p[["Actual", "Predicted"]].max().max() * 1.1]
    fig = px.scatter(p, x="Actual", y="Predicted", color="City", color_discrete_map=CITY_COLORS, opacity=0.6,
                     log_x=True, log_y=True, hover_data=["Locality", "BHK", "Area_sqft"],
                     title="Actual vs predicted (log scale)", labels={"Actual": "Actual (₹ L)", "Predicted": "Predicted (₹ L)"})
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#94A3B8", dash="dash"), name="Perfect"))
    fig.update_layout(height=400, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.histogram(p, x="Error_pct", nbins=60, color_discrete_sequence=[ACCENT],
                       title="Residuals: percentage error", labels={"Error_pct": "Prediction error (%)"})
    fig.add_vline(x=0, line_dash="dash", line_color="#0F172A")
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    by = p.groupby(["City", "Property_Type"]).apply(
        lambda g: pd.Series({"MAPE": np.mean(np.abs(g["Error_pct"])), "n": len(g)}), include_groups=False
    ).reset_index()
    fig = px.bar(by, x="City", y="MAPE", color="Property_Type", barmode="group", hover_data=["n"],
                 title="Error by city and property type", labels={"MAPE": "MAPE (%)", "Property_Type": ""})
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Villas and independent houses have fewer examples, so their error is higher.")
with c2:
    imp = pd.Series(metrics["feature_importance"]).sort_values()
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[ACCENT],
                 title="Feature importance (mean |SHAP|, LightGBM)", labels={"x": "Mean absolute effect on log(price)", "y": ""})
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

footer()

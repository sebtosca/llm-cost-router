import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from llm_cost_router.storage.db import get_connection
from llm_cost_router.storage.stats import compute_daily_series, compute_stats

st.set_page_config(page_title="LLM Cost Router Dashboard", layout="wide")
st.title("LLM Cost Router — Cost & Quality Dashboard")

stats = compute_stats()

col1, col2, col3 = st.columns(3)
col1.metric("Total cost", f"${stats.total_cost_usd:.4f}")
col2.metric("Cost if everything used the priciest model", f"${stats.baseline_cost_usd:.4f}")
col3.metric("You saved", f"{stats.savings_pct:.1f}%", f"${stats.savings_usd:.4f}")

st.divider()

daily = compute_daily_series()
if daily:
    df_daily = pd.DataFrame([d.model_dump() for d in daily]).set_index("date")

    st.subheader("Cost per day: actual vs. all-priciest-model baseline")
    st.line_chart(df_daily[["cost_usd", "baseline_usd"]])

    st.subheader("Escalation rate over time")
    st.line_chart(df_daily[["escalation_rate"]])
else:
    st.info("No logged requests yet. Hit the API or run scripts/baseline_test.py first.")

st.divider()

col4, col5 = st.columns(2)

with col4:
    st.subheader("Routing distribution (cost by model)")
    if stats.cost_by_model:
        fig = px.pie(
            names=list(stats.cost_by_model.keys()),
            values=list(stats.cost_by_model.values()),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No cost data yet.")

with col5:
    st.subheader("Quality score distribution")
    with get_connection() as conn:
        scores = [
            row[0]
            for row in conn.execute(
                "SELECT quality_score FROM request_log WHERE quality_score IS NOT NULL"
            ).fetchall()
        ]
    if scores:
        fig = px.histogram(x=scores, nbins=5, labels={"x": "quality_score"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No quality scores yet (verification runs asynchronously after requests).")

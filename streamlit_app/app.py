"""Streamlit entry point."""

import streamlit as st
from views import campaigns, churn, crm, forecasting, panel, recommendations

st.set_page_config(page_title="Retail Intelligence", layout="wide")
st.markdown(
    """
<style>
h1, h2 { color: #365f73; }
h1 { border-bottom: 4px solid #689EB8; padding-bottom: .45rem; }
[data-testid="stMetric"] { background: white; border-top: 3px solid #689EB8; border-radius: .6rem; }
.st-key-customer_risk [data-testid="stMetric"], .st-key-manual_risk [data-testid="stMetric"] { border-top-color: #FF5A60; }
</style>
""",
    unsafe_allow_html=True,
)
st.title("Retail Intelligence")
st.caption("Dunnhumby · Historical shopping data")
tabs = st.tabs(
    [
        "CRM & Customer Segmentation",
        "Customer Churn",
        "Statistical Campaign Analysis",
        "Recommendation Systems",
        "Product Demand Forecasting",
    ]
)
for tab, view in zip(tabs, (crm, churn, campaigns, recommendations, forecasting)):
    with tab:
        panel(view)

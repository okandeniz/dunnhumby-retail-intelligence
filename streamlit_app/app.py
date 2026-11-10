"""Streamlit entry point."""

import streamlit as st
from views import campaigns, churn, crm, forecasting, panel, recommendations

st.set_page_config(page_title="Retail Intelligence", layout="wide")
st.markdown(
    """
<style>
h1, h2, h3 { color: #0D6628; }
h1 { border-bottom: 4px solid #6EAD50; padding-bottom: .45rem; }
[data-testid="stMetric"] { background: white; border-top: 3px solid #6EAD50; border-radius: .6rem; }
.st-key-churn_id_cards [data-testid="stMetric"], .st-key-churn_manual_cards [data-testid="stMetric"] {
    border-top-color: #FE0000;
}
:is(.st-key-churn_id_cards, .st-key-churn_manual_cards) [data-testid="stMetric"] * {
    color: #FE0000;
}
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

"""Dashboard views. All numbers and model results come from the API."""

import pandas as pd
import plotly.express as px
import streamlit as st
from charts import AVERAGE, BLUE, LAST, future_chart, history_chart, style
from client import ApiError, read, request


def metrics(entries: dict) -> None:
    for column, (label, value) in zip(st.columns(len(entries)), entries.items()):
        column.metric(label, value, border=True)


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", theme=None)


def panel(view) -> None:
    try:
        view()
    except ApiError as exc:
        st.warning(str(exc))


SEGMENTS = {
    "Champions": "Shop often, bought recently and spend a lot.",
    "Loyal Customers": "Shop often and bought recently.",
    "Potential Loyalists": "Bought recently and shop at a medium rate.",
    "Recent Low Frequency": "Bought recently but have made few purchases.",
    "Recently Active": "Fall in the middle group for time since the last purchase.",
    "Less Recent - Frequent": "Used to shop regularly. Their last visit was some time ago.",
    "Less Recent - Infrequent": "Shop less often. Their last visit was some time ago.",
    "Dormant (365d)": "Had no positive-value basket in the last year.",
}


def crm() -> None:
    st.header("Customer value and shopping habits")
    data = read("/crm/summary")
    metrics(
        {
            "Customers": f"{data['customers']:,}",
            "Spending in 365 days": f"{data['spending']:,.0f}",
            "Expected revenue in 90 days": f"{data['expected_revenue']:,.0f}",
        }
    )
    left, right = st.columns([3, 2])
    segments = pd.DataFrame(data["segments"])
    with left:
        measure = st.selectbox(
            "Compare RFM segments by", ["Customer count", "Average expected revenue"]
        )
        key = (
            "customer_count" if measure == "Customer count" else "avg_expected_revenue"
        )
        fig = px.bar(
            segments.sort_values(key),
            x=key,
            y="segment",
            orientation="h",
            color_discrete_sequence=[BLUE],
            labels={key: measure, "segment": "RFM segment"},
            title=measure + " by segment",
        )
        chart(style(fig))
        st.caption("Revenue estimates cover customers with enough purchase history.")
    with right:
        st.subheader("Read a segment")
        label = st.selectbox("RFM segment", segments.segment.tolist())
        st.write(SEGMENTS.get(label, label))
        st.caption(
            "RFM groups customers by days since the last purchase, number of baskets and total spending."
        )
    st.subheader("Shopping groups")
    st.caption(
        f"K-Means grouped {data['clustered']:,} customers by purchase count, average basket value and days since the last purchase."
    )
    clusters = pd.DataFrame(data["clusters"]).rename(
        columns={
            "cluster_name": "Group",
            "customer_count": "Customers",
            "median_frequency": "Median baskets",
            "median_recency": "Median days since purchase",
            "median_basket_value": "Median basket value",
            "spending_share_pct": "Share of spending (%)",
        }
    )
    st.dataframe(clusters, hide_index=True)
    st.subheader("Customer details")
    with st.form("find_crm"):
        cid = st.number_input("Customer ID", min_value=1, value=1, key="crm_id")
        submitted = st.form_submit_button("Find customer")
    if submitted:
        st.session_state.crm_selection = cid
    if st.session_state.get("crm_selection") == cid:
        person = read(f"/crm/customers/{cid}")
        st.write(f"**{person['segment']}** · {person['cluster_name']}")
        metrics(
            {
                "Days since last purchase": person["recency"],
                "Baskets in 365 days": person["frequency"],
                "Spending in 365 days": f"{person['monetary']:,.2f}",
            }
        )
        st.caption(
            f"RFM scores (1–5) • Recency {person['R_score']} • Frequency {person['F_score']} • Spending {person['M_score']}"
        )
        revenue = person["expected_revenue_90d"]
        st.write(
            f"Expected revenue in 90 days: {revenue:,.2f}"
            if revenue is not None
            else person["forecast_status"]
        )


def prediction(result: dict) -> None:
    st.subheader("Model prediction")
    if result["customer_id"] is not None:
        st.write(f"Customer {result['customer_id']}")
    metrics(
        {
            "Predicted status": result["status"],
            "Churn probability": f"{result['probability']:.1%}",
            "Decision threshold": f"{result['threshold']:.0%}",
        }
    )
    st.caption(
        "The model predicts churn when the chance of no purchase in the next 90 days reaches the threshold."
    )
    if result["observed"] is not None:
        st.write(f"Observed outcome in days 622–711: **{result['observed']}**")


def churn_id() -> None:
    with st.form("churn_id"):
        cid = st.number_input("Customer ID", min_value=1, key="churn_customer")
        submitted = st.form_submit_button("Predict by customer ID", type="primary")
    if submitted:
        st.session_state.churn_selection = cid
    if st.session_state.get("churn_selection") == cid:
        prediction(read(f"/churn/customers/{cid}"))


def manual() -> None:
    st.write(
        "Enter shopping activity before the forecast day, or start with a real test customer."
    )
    if st.button("Random Fill"):
        row = request("/churn/sample")
        st.session_state.pop("manual_result", None)
        for key, value in row["features"].items():
            st.session_state["manual_" + key] = value
        st.session_state["sample_id"] = row["customer_id"]
    if "sample_id" in st.session_state:
        st.caption(
            f"Starting values from test customer {st.session_state.sample_id}. You can edit every field."
        )
    with st.form("manual_prediction"):
        values = {}
        left, right = st.columns(2)
        fields = [
            ("purchase_frequency", "Baskets in the last 365 days", left),
            ("avg_basket_value", "Average value per basket in those 365 days", right),
            ("recency", "Days since the last purchase", left),
            ("baskets_last_30d", "Baskets in the last 30 days", right),
            ("baskets_prev_30d", "Baskets 31 to 60 days ago", left),
        ]
        for key, label, column in fields:
            money = key == "avg_basket_value"
            values[key] = column.number_input(
                label,
                min_value=0.0 if money else int(key == "recency"),
                value=None,
                key="manual_" + key,
                format="%.6f" if money else "%d",
            )
        values["has_purchase_365d"] = right.selectbox(
            "Any purchase in the last 365 days?",
            [0, 1],
            index=None,
            format_func=lambda v: "Yes" if v else "No",
            key="manual_has_purchase_365d",
        )
        submitted = st.form_submit_button("Predict Churn", type="primary")
    if submitted:
        st.session_state.pop("manual_result", None)
        if None in values.values():
            st.warning("Complete all six fields before making a prediction.")
        else:
            st.session_state.manual_result = request("/churn/predict", body=values)
    if "manual_result" in st.session_state:
        prediction(st.session_state.manual_result)


def churn() -> None:
    st.header("Customer churn")
    st.caption(
        "This experimental model uses a year of shopping history to estimate the chance of no purchase for 90 days."
    )
    by_id, by_hand = st.tabs(["Customer ID Prediction", "Manual Prediction"])
    with by_id, st.container(key="customer_risk"):
        panel(churn_id)
    with by_hand, st.container(key="manual_risk"):
        panel(manual)


def campaigns() -> None:
    st.header("Campaign participation and spending")
    cid = st.selectbox("Campaign", read("/campaigns"))
    selected = read(f"/campaigns/{cid}")
    st.write(f"Campaign {cid} · {selected['campaign_type']}")
    metrics(
        {
            "Assignments": selected["assignments"],
            "Redeemed": selected["redeemed"],
            "Average spending": f"{selected['avg_spending']:,.2f}",
            "Average baskets": f"{selected['avg_baskets']:.1f}",
        }
    )
    st.subheader("Across all campaigns")
    study = read("/campaigns/study")
    fig = px.bar(
        study["types"],
        x="DESCRIPTION",
        y="redemption_rate_pct",
        text_auto=".1f",
        color_discrete_sequence=[BLUE],
        title="Coupon use by campaign type",
        labels={
            "DESCRIPTION": "Campaign type",
            "redemption_rate_pct": "Assignments with a redemption (%)",
        },
    )
    chart(style(fig))
    st.write(
        f"The chi-square test compares coupon use across types (χ² = {study['chi2']:.2f}, "
        f"df = {study['degrees_of_freedom']}, p = {study['p_value']:.2g}). "
        f"The link is statistically significant at 5%. Its size is small (Cramér’s V = {study['cramers_v']:.2f})."
    )
    tests = pd.DataFrame(study["spending_tests"]).rename(
        columns={
            "campaign_type": "Type",
            "median_redeemed": "Used a coupon",
            "median_not_redeemed": "Did not use a coupon",
            "p_adjusted": "Adjusted p-value",
            "cliffs_delta": "Cliff’s delta",
        }
    )
    st.subheader("Spending during the campaign")
    st.write(
        "In all three types, customers who used a coupon had higher median spending. The table shows two-sided Mann–Whitney tests with a Holm adjustment."
    )
    tests["Adjusted p-value"] = tests["Adjusted p-value"].map("{:.2g}".format)
    st.dataframe(tests, hide_index=True)
    with st.expander("How do repeated visits affect the comparison?"):
        st.write(
            "A household can receive more than one campaign. The GEE model groups records by household. It compares the odds of using a coupon with TypeB as the reference."
        )
        estimates = pd.DataFrame(study["redemption_estimates"])
        estimates["Campaign type"] = estimates.term.str.extract(r"\[T\.(.*?)\]")[0]
        estimates = estimates.dropna(subset=["Campaign type"]).rename(
            columns={
                "odds_ratio": "Odds ratio",
                "ci_lower": "95% lower limit",
                "ci_upper": "95% upper limit",
                "p_value": "p-value",
            }
        )
        estimates["p-value"] = estimates["p-value"].map("{:.2g}".format)
        st.dataframe(
            estimates[
                [
                    "Campaign type",
                    "Odds ratio",
                    "95% lower limit",
                    "95% upper limit",
                    "p-value",
                ]
            ],
            hide_index=True,
        )
    st.caption(
        "These groups come from observed shopping records. Customer habits may explain some differences. The study cannot measure a causal effect."
    )


def recommendations() -> None:
    st.header("Product recommendations")
    modes = read("/recommendations/models")
    left, middle, right = st.columns([1, 2, 1])
    mode = left.selectbox("Recommend for a", list(modes))
    model = middle.selectbox("Recommendation model", modes[mode])
    count = right.number_input("Number of results", min_value=1, max_value=50, value=10)
    if mode == "Customer":
        entity = st.number_input("Customer ID", min_value=1, key="rec_customer")
    else:
        query = st.text_input(
            "Search by product name or ID",
            help="Enter more letters to narrow the first 50 matches.",
        )
        choices = {
            r["PRODUCT_ID"]: r["product_label"]
            for r in read("/recommendations/products", model=model, q=query)
        }
        entity = st.selectbox(
            "Product", list(choices), format_func=choices.get, key="rec_product"
        )
        if not choices:
            st.info("No products match this search.")
    if st.button("Show recommendations", type="primary", disabled=entity is None):
        st.session_state.rec_selection = (mode, model, entity, count)
    if st.session_state.get("rec_selection") == (mode, model, entity, count):
        path = "customers" if mode == "Customer" else "products"
        ranking = read(f"/recommendations/{path}/{entity}", model=model, top_n=count)
        if ranking["items"]:
            frame = pd.DataFrame(ranking["items"]).rename(
                columns={
                    "PRODUCT_ID": "Product ID",
                    "product_label": "Product",
                    "score": ranking["score_label"],
                }
            )
            st.dataframe(frame, hide_index=True)
        else:
            st.info("This model found no matching products.")
    st.subheader("Compare all seven methods")
    st.caption(
        "Each method suggested up to 10 new products per customer. The test covers days 622–711."
    )
    frame = pd.DataFrame(read("/recommendations/comparison")).rename(
        columns={
            "model": "Model",
            "precision_at_k": "Precision@10",
            "recall_at_k": "Recall@10",
            "hit_rate_at_k": "Hit Rate@10",
            "coverage_at_k": "Coverage@10",
        }
    )
    metric = st.selectbox("Comparison metric", list(frame.columns[1:]))
    fig = px.bar(
        frame.sort_values(metric),
        x=metric,
        y="Model",
        orientation="h",
        color_discrete_sequence=[BLUE],
        title=f"{metric} by method",
    )
    fig.update_xaxes(tickformat=".0%")
    chart(style(fig))
    st.dataframe(frame, hide_index=True)
    st.caption(
        "Precision counts hits in the top 10. Recall measures how many new purchases appeared in the list. Hit rate counts customers with a hit. Coverage measures how much of the catalog appeared. NMF training did not confirm convergence."
    )


def forecasting() -> None:
    st.header("Product demand forecasting")
    st.write("Explore total units across stores for the 30 days after dataset day 711.")
    categories = read("/forecasting/categories")
    left, right = st.columns([1, 2])
    category = left.selectbox(
        "Category",
        categories,
        index=categories.index("PRODUCE / TROPICAL FRUIT")
        if "PRODUCE / TROPICAL FRUIT" in categories
        else 0,
    )
    options = {
        p["PRODUCT_ID"]: p["product_label"]
        for p in read("/forecasting/products", category=category)
    }
    ids = list(options)
    prod = right.selectbox(
        "Product",
        ids,
        index=ids.index(1082185) if 1082185 in ids else 0,
        format_func=options.get,
        key="forecast_product",
    )
    row = read(f"/forecasting/products/{prod}")
    metrics(
        {
            "XGBoost + Log Target": f"{row['xgb_prediction']:,.2f}",
            LAST: f"{row['baseline_last_30d']:,.1f}",
            AVERAGE: f"{row['baseline_avg_60d']:,.1f}",
        }
    )
    chart(future_chart(row))
    st.caption(
        "Each bar shows one total for days 712–741. Actual sales for this period are unavailable."
    )
    chart(history_chart(read(f"/forecasting/products/{prod}/history")))
    st.subheader("Test results across all eligible products")
    results = pd.DataFrame(read("/forecasting/evaluation")).rename(
        columns={"model": "Method", "wape_pct": "WAPE (%)"}
    )
    results["Method"] = results["Method"].replace(
        {"Last 30 Days": LAST, "Last 60-Day Average": AVERAGE}
    )
    st.dataframe(results.round(3), hide_index=True)
    st.caption(
        "MAE is the average error in units. RMSE gives more weight to large errors. WAPE divides total absolute error by total actual sales. Lower values are better."
    )

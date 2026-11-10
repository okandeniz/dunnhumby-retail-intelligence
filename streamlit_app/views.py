"""Dashboard views. All numbers and model results come from the API."""

import pandas as pd
import plotly.express as px
import streamlit as st
from charts import AVERAGE, GREEN, LAST, history_chart, sales_forecast_chart, style
from client import ApiError, read, request
from streamlit_searchbox import st_searchbox


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
            color_discrete_sequence=[GREEN],
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
        segment, group = st.columns(2)
        with segment.container(border=True):
            st.caption("RFM segment")
            st.write(f"**{person['segment']}**")
        with group.container(border=True):
            st.caption("Shopping group")
            st.write(f"**{person['cluster_name'] or 'No group assigned'}**")
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
    is_churn = result["status"] == "Churn"
    source = "id" if result["customer_id"] is not None else "manual"
    with st.container(key=f"{'churn' if is_churn else 'active'}_{source}_cards"):
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
    with st.container(border=True):
        st.markdown("**Suggested action**")
        st.write(
            "Consider a reminder or an offer on a product this customer bought before. Check whether they return in the next 90 days."
            if is_churn
            else "Keep the usual contact schedule. Suggest products that fit this customer's shopping habits and watch for longer gaps between visits."
        )


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
    with by_id:
        panel(churn_id)
    with by_hand:
        panel(manual)


def display_p_value(value: float) -> str:
    """Keep small probabilities readable without rounding them to zero."""
    return "< 0.000001" if value < 0.000001 else f"{value:.6f}".rstrip("0").rstrip(".")


def analysis_note(title: str, text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)


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
    analysis_note(
        "How to read this campaign",
        f"Of the {selected['assignments']:,} households assigned to this campaign, "
        f"{selected['redeemed']:,} used at least one coupon. The averages cover all assigned households during the campaign.",
    )
    st.subheader("Coupon use across all campaigns")
    study = read("/campaigns/study")
    rates = pd.DataFrame(study["types"])
    rates["campaign_type"] = rates.DESCRIPTION.str.replace("Type", "Type ", regex=False)
    fig = px.bar(
        rates,
        x="campaign_type",
        y="redemption_rate_pct",
        color_discrete_sequence=[GREEN],
        title="How often did households use a campaign coupon?",
        custom_data=["redeemed_assignments", "targeted_assignments"],
        labels={
            "campaign_type": "Type of campaign received",
            "redemption_rate_pct": "Share of assignments with coupon use",
        },
    )
    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Coupon used in %{customdata[0]:,} of %{customdata[1]:,} assignments (%{y:.1f}%)<extra></extra>",
    )
    fig.update_yaxes(ticksuffix="%", range=[0, rates.redemption_rate_pct.max() * 1.18])
    chart(style(fig))
    analysis_note(
        "What do the bars tell us?",
        "Each household counts once for each campaign it received. A taller bar means a larger share used a coupon. "
        f"The chi-square test finds a difference across types (p-value {display_p_value(study['p_value'])}). "
        f"The link is small (Cramér’s V = {study['cramers_v']:.2f}).",
    )
    tests = pd.DataFrame(study["spending_tests"]).rename(
        columns={
            "campaign_type": "Type",
            "median_redeemed": "Median spending with coupon use",
            "median_not_redeemed": "Median spending without coupon use",
            "p_adjusted": "Adjusted p-value",
            "cliffs_delta": "Cliff’s delta",
        }
    )
    st.subheader("Spending during the campaign")
    analysis_note(
        "How did spending differ?",
        "In each campaign type, households that used a coupon had higher median spending. The median is the middle spending value. "
        "The Mann–Whitney test compares spending in the two groups. We adjusted the p-values because we made three comparisons (Holm method). "
        "An adjusted p-value below 0.05 gives evidence of a difference. "
        "A positive Cliff’s delta means the household that used a coupon usually spent more when we compare pairs from the two groups.",
    )
    tests["Adjusted p-value"] = tests["Adjusted p-value"].map(display_p_value)
    st.dataframe(tests, hide_index=True)
    with st.expander("What if a household received several campaigns?"):
        analysis_note(
            "Compare each type with Type B",
            "The GEE model keeps records from the same household together. An odds ratio above 1 means the odds of using a coupon were higher than in Type B. "
            "If the 95% interval includes 1, the data are also consistent with equal odds.",
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
        estimates["p-value"] = estimates["p-value"].map(display_p_value)
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
    analysis_note(
        "What can we conclude?",
        "The records show a link between coupon use and spending. Customers who already shop more may also use more coupons. "
        "These comparisons cannot tell us how much extra spending a campaign caused.",
    )


METRIC_GUIDE = {
    "Precision@10": "For each customer, count how many of the 10 suggestions they later bought for the first time. Divide by 10, then average across customers. Two matches out of 10 give 20%.",
    "Recall@10": "Of the new products a customer later bought, how many appeared in their list? Two matches out of five new purchases give 40%. The chart shows the average across customers.",
    "Hit Rate@10": "What share of customers bought at least one new product from their list? If 30 of 100 customers bought a suggested product, the hit rate is 30%.",
    "Coverage@10": "What share of the product catalog appeared in at least one customer's list? A product counts once, even if many customers saw it. Higher coverage means a wider range of suggestions.",
}


def product_options(query: str, model: str) -> list[tuple[str, int]]:
    rows = read("/recommendations/products", model=model, q=query[:100])
    return [(row["product_label"], row["PRODUCT_ID"]) for row in rows]


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
        entity = st_searchbox(
            product_options,
            model=model,
            label="Product",
            placeholder="Choose a product or type its name or ID",
            default_options=product_options("", model),
            key=f"rec_product_{model}",
        )
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
    analysis_note(metric, METRIC_GUIDE[metric])
    fig = px.bar(
        frame.sort_values(metric),
        x=metric,
        y="Model",
        orientation="h",
        color_discrete_sequence=[GREEN],
        title=f"{metric} by method",
    )
    fig.update_xaxes(tickformat=".0%")
    chart(style(fig))
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="percent", help=guide)
            for name, guide in METRIC_GUIDE.items()
        },
    )
    st.caption("NMF training did not confirm convergence.")


def store_forecasts(product_id: int) -> None:
    try:
        estimates = read(f"/forecasting/products/{product_id}/stores")
    except ApiError as exc:
        st.warning(str(exc))
        return
    if not estimates:
        st.info("This product has no store sales history for the last 180 days.")
        return
    st.write(
        "Each store gets a share of the product’s 30-day forecast based on how many units "
        "it sold in the last 180 days. Only stores that sold this product in that period are shown."
    )
    frame = pd.DataFrame(estimates).rename(
        columns={
            "STORE_ID": "Store ID",
            "units_last_180d": "Units sold in 180 days",
            "sales_share": "Share of sales",
            "estimated_units": "Estimated units",
        }
    )
    top = frame.head(10).copy()
    top["Store"] = "Store " + top["Store ID"].astype(str)
    fig = px.bar(
        top,
        x="Estimated units",
        y="Store",
        orientation="h",
        color_discrete_sequence=[GREEN],
        title=f"30-day estimates for the top {len(top)} stores",
        text_auto=",.1f",
    )
    fig.update_yaxes(title=None, autorange="reversed")
    chart(style(fig))
    st.caption(
        "Store estimates are approximate. In three past 30-day periods, store-level "
        "errors were large, especially compared with the forecast for all stores together."
    )
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "Share of sales": st.column_config.NumberColumn(format="percent"),
            "Estimated units": st.column_config.NumberColumn(format="%.2f"),
        },
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
    st.subheader(
        f"Forecast totals for days {row['forecast_start_day']}–{row['forecast_end_day']}"
    )
    metrics(
        {
            "XGBoost + Log Target": f"{row['xgb_prediction']:,.2f}",
            LAST: f"{row['baseline_last_30d']:,.1f}",
            AVERAGE: f"{row['baseline_avg_60d']:,.1f}",
        }
    )
    st.caption("Actual sales for this period are unavailable.")
    history = read(f"/forecasting/products/{prod}/history")
    future_tab, history_tab, store_tab = st.tabs(
        ["Next 30-day forecast", "Actual and predicted demand", "Forecast by store"]
    )
    with future_tab:
        chart(sales_forecast_chart(history, row))
        st.caption(
            "Each point covers 30 days. The dashed line leads to the model forecast. "
            "The diamond shows average sales over the three previous periods."
        )
    with history_tab:
        chart(history_chart(history))
    with store_tab:
        store_forecasts(prod)
    st.subheader("Test results for product totals across all stores")
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

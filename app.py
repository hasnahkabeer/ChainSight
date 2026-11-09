import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ChainSight | Supply Chain Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #101d32 100%);
        color: #f1f5f9;
    }
    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3, h4 { color: #f8fafc !important; }
    p, label, .stCaption { color: #cbd5e1; }
    [data-testid="stSidebar"] {
        background: #0b1628;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #14243b, #102039);
        border: 1px solid #244365;
        padding: 17px;
        border-radius: 14px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #0284c7, #2563eb);
        color: white !important;
        border: 0;
        border-radius: 10px;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] {
        background: #13243a;
        border: 1px dashed #38bdf8;
        border-radius: 14px;
        padding: 12px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #244365;
        border-radius: 10px;
    }
    hr { border-color: #263b55; }

    /* Premium animated dashboard elements */
    @keyframes cardEnter {
        from { opacity: 0; transform: translateY(18px) scale(.985); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes softGlow {
        0%, 100% { box-shadow: 0 8px 24px rgba(2,132,199,.10); }
        50%      { box-shadow: 0 10px 30px rgba(56,189,248,.23); }
    }
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-banner {
        position: relative;
        overflow: hidden;
        padding: 25px 28px;
        border-radius: 18px;
        border: 1px solid rgba(56,189,248,.35);
        background: linear-gradient(115deg, #102c4c, #075985, #123e65, #102c4c);
        background-size: 250% 250%;
        animation: gradientFlow 12s ease infinite, cardEnter .7s ease both;
        margin: 8px 0 22px 0;
    }
    .hero-banner h2 { margin: 0 0 6px 0; color: #fff !important; }
    .hero-banner p { margin: 0; color: #dbeafe; }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 15px;
        margin: 12px 0 20px 0;
    }
    .kpi-card {
        position: relative;
        overflow: hidden;
        min-height: 126px;
        padding: 18px 19px;
        border-radius: 16px;
        border: 1px solid rgba(56,189,248,.24);
        background: linear-gradient(145deg, rgba(20,36,59,.98), rgba(13,31,53,.98));
        animation: cardEnter .65s ease both, softGlow 5s ease-in-out infinite;
        transition: transform .25s ease, border-color .25s ease;
    }
    .kpi-card:hover { transform: translateY(-5px); border-color: rgba(56,189,248,.75); }
    .kpi-card::after {
        content: "";
        position: absolute;
        width: 110px;
        height: 110px;
        right: -38px;
        top: -42px;
        border-radius: 50%;
        background: rgba(56,189,248,.10);
        filter: blur(2px);
    }
    .kpi-top { display: flex; align-items: center; gap: 9px; color: #a9bdd2; font-size: 13px; font-weight: 650; letter-spacing: .3px; }
    .kpi-icon { font-size: 20px; }
    .kpi-value { color: #f8fafc; font-size: clamp(23px, 2.3vw, 31px); font-weight: 800; margin-top: 13px; line-height: 1.15; animation: cardEnter .9s ease both; }
    .kpi-note { color: #7dd3fc; font-size: 11px; margin-top: 7px; }
    .section-heading { padding: 10px 0 4px 0; border-bottom: 1px solid rgba(148,163,184,.18); margin-bottom: 12px; }
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] { border-radius: 14px; }
    @media (max-width: 850px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 520px) { .kpi-grid { grid-template-columns: 1fr; } }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. HEADER
# =========================================================
st.title("🔗 ChainSight")
st.subheader("AI-Powered Supply Chain Intelligence Platform")
st.caption(
    "Predict disruptions. Evaluate supplier risks. "
    "Make more informed supply chain decisions."
)
st.markdown(
    """
    <div class="hero-banner">
        <h2>See risk earlier. Make decisions smarter.</h2>
        <p>Your supply chain overview, supplier insights and disruption scenarios — in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# =========================================================
# 4. UPLOAD CSV
# =========================================================
st.header("📂 Upload Supply Chain Dataset")
uploaded_file = st.file_uploader(
    "Choose your supplier CSV file",
    type=["csv"],
    help="Upload a CSV containing the required supply-chain columns.",
)

if uploaded_file is None:
    st.info("Welcome to ChainSight! Upload your CSV dataset to begin.")
    feature_cols = st.columns(3)
    feature_cols[0].markdown(
        "### 📊 Supplier Analysis\nCompare supplier performance and inventory exposure."
    )
    feature_cols[1].markdown(
        "### ⚠️ Disruption Simulator\nExplore hypothetical delivery and inventory disruptions."
    )
    feature_cols[2].markdown(
        "### 🤖 AI Risk Prediction\nEstimate inventory risk for a new order using the trained model."
    )
    st.stop()

# =========================================================
# 5. LOAD AND VALIDATE DATA
# IMPORTANT: The dataframe is named `data` throughout this app.
# The AI page must use `data`, NOT an undefined variable `df`.
# =========================================================
try:
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file, encoding="utf-8-sig")
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

data.columns = data.columns.astype(str).str.strip()

required_columns = [
    "order_id",
    "supplier",
    "product",
    "promised_delivery_days",
    "quantity",
    "unit_cost",
    "inventory_days",
    "demand_units",
    "disruption_type",
    "actual_delivery_days",
    "delivery_delay_days",
    "order_value",
]

missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error(
        "Your CSV is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.write("Columns found in your file:", list(data.columns))
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

categorical_columns = ["supplier", "product", "disruption_type"]
numeric_columns = [
    "promised_delivery_days",
    "quantity",
    "unit_cost",
    "inventory_days",
    "demand_units",
    "actual_delivery_days",
    "delivery_delay_days",
    "order_value",
]

original_rows = len(data)

for column in categorical_columns:
    data[column] = (
        data[column].astype("string").str.strip().replace("", pd.NA)
    )

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

invalid_rows = data[
    data[categorical_columns + numeric_columns].isna().any(axis=1)
].copy()
data = data.dropna(subset=categorical_columns + numeric_columns).copy()
data = data.reset_index(drop=True)

if data.empty:
    st.error(
        "No valid rows remain after checking the required columns. "
        "Please correct missing or non-numeric values in your CSV."
    )
    st.stop()

if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
    st.error("Your CSV contains infinite numerical values. Please correct them.")
    st.stop()

st.success(f"Dataset loaded successfully — {len(data)} valid records.")
with st.expander("🔍 View data quality report"):
    q1, q2, q3 = st.columns(3)
    q1.metric("Original rows", original_rows)
    q2.metric("Valid rows", len(data))
    q3.metric("Rows removed", original_rows - len(data))
    st.write("Missing values in the cleaned dataset:")
    st.dataframe(data[required_columns].isna().sum(), use_container_width=True)

# =========================================================
# 6. SIDEBAR AND NAVIGATION
# =========================================================
st.sidebar.title("🔗 ChainSight")
st.sidebar.caption("SUPPLY CHAIN INTELLIGENCE")
st.sidebar.success(f"📦 {len(data)} records loaded")
page = st.sidebar.radio(
    "Explore ChainSight",
    [
        "Overview",
        "Supplier Analysis",
        "Disruption Simulator",
        "AI Risk Prediction",
    ],
)
st.sidebar.divider()
st.sidebar.caption("ChainSight | AI Supply Chain Intelligence")

# =========================================================
# 7. COMMON METRICS FUNCTION
# =========================================================
def calculate_metrics(frame):
    delayed = frame["delivery_delay_days"] > 0
    inventory_risk = (
        frame["actual_delivery_days"] > frame["inventory_days"]
    )
    positive_delay = frame["delivery_delay_days"].clip(lower=0)
    estimated_delay_cost = (
        frame["order_value"] * positive_delay * 0.02
    ).clip(upper=frame["order_value"]).sum()

    return {
        "Total Orders": int(len(frame)),
        "Delayed Orders": int(delayed.sum()),
        "Delay Percentage": round(float(delayed.mean() * 100), 2),
        "Average Delivery Days": round(
            float(frame["actual_delivery_days"].mean()), 2
        ),
        "Inventory Risk Orders": int(inventory_risk.sum()),
        "Estimated Delay Cost": round(float(estimated_delay_cost), 2),
    }


def render_kpi(label, value, icon="📊", note="Live dataset metric", index=0):
    """Render an animated KPI card using the current dataset values."""
    st.markdown(
        f"""
        <div class="kpi-card" style="animation-delay:{index * 0.10}s">
            <div class="kpi-top"><span class="kpi-icon">{icon}</span><span>{label}</span></div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 8. OVERVIEW
# =========================================================
if page == "Overview":
    st.header("📊 Supply Chain Command Center")
    st.caption("Monitor operational performance and identify potential exposure.")

    metrics = calculate_metrics(data)
    kpi_items = [
        ("Total Orders", f'{metrics["Total Orders"]:,}', "📦", "Orders in uploaded dataset"),
        ("Delayed Orders", f'{metrics["Delayed Orders"]:,}', "⚠️", "Orders with a positive delivery delay"),
        ("Delay Percentage", f'{metrics["Delay Percentage"]}% ', "📉", "Share of orders marked delayed"),
        ("Average Delivery", f'{metrics["Average Delivery Days"]} days', "🚚", "Mean actual delivery time"),
        ("Inventory Risk Orders", f'{metrics["Inventory Risk Orders"]:,}', "🏭", "Delivery time exceeds inventory days"),
        ("Estimated Delay Cost", f'₹{metrics["Estimated Delay Cost"]:,.2f}', "💰", "Illustrative cost estimate"),
    ]
    for row_start in (0, 3):
        kpi_columns = st.columns(3)
        for column, item, index in zip(kpi_columns, kpi_items[row_start:row_start + 3], range(row_start, row_start + 3)):
            with column:
                render_kpi(item[0], item[1], item[2], item[3], index)

    st.divider()
    st.subheader("📈 Supply Chain Performance")
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("#### Orders by Supplier")
        st.bar_chart(data["supplier"].value_counts(), color="#38bdf8")
    with chart2:
        st.markdown("#### Disruption Distribution")
        st.bar_chart(data["disruption_type"].value_counts(), color="#2dd4bf")

    st.subheader("🚚 Delivery Status")
    delayed_count = int((data["delivery_delay_days"] > 0).sum())
    on_time_count = len(data) - delayed_count
    delivery_chart = pd.DataFrame(
        {"Orders": [delayed_count, on_time_count]},
        index=["Delayed", "On Time"],
    )
    st.bar_chart(delivery_chart, color="#38bdf8")

    with st.expander("📋 View uploaded dataset"):
        st.dataframe(data, use_container_width=True, hide_index=True)

# =========================================================
# 9. SUPPLIER ANALYSIS
# =========================================================
elif page == "Supplier Analysis":
    st.header("🏭 Supplier Risk Analysis")
    st.caption("Compare supplier delivery performance and inventory exposure.")

    analysis_data = data.copy()
    analysis_data["inventory_risk"] = (
        analysis_data["actual_delivery_days"]
        > analysis_data["inventory_days"]
    )

    supplier_risk = (
        analysis_data.groupby("supplier")
        .agg(
            Total_Orders=("order_id", "count"),
            Average_Delivery=("actual_delivery_days", "mean"),
            Average_Delay=("delivery_delay_days", "mean"),
            Risk_Orders=("inventory_risk", "sum"),
            Total_Order_Value=("order_value", "sum"),
        )
        .reset_index()
    )
    supplier_risk["Risk_Percentage"] = (
        supplier_risk["Risk_Orders"] / supplier_risk["Total_Orders"] * 100
    ).round(2)
    supplier_risk["Average_Delivery"] = supplier_risk["Average_Delivery"].round(2)
    supplier_risk["Average_Delay"] = supplier_risk["Average_Delay"].round(2)
    supplier_risk = supplier_risk.sort_values(
        "Risk_Percentage", ascending=False
    )

    a, b, c = st.columns(3)
    with a:
        render_kpi("Total Suppliers", supplier_risk["supplier"].nunique(), "🏭", "Suppliers in uploaded data", 0)
    with b:
        render_kpi("Highest Supplier Risk", f'{supplier_risk["Risk_Percentage"].max():.2f}%', "🚨", "Highest calculated risk share", 1)
    with c:
        render_kpi("Average Supplier Delay", f'{supplier_risk["Average_Delay"].mean():.2f} days', "🚚", "Mean delay across suppliers", 2)

    def classify_risk(value):
        if value >= 50:
            return "High"
        if value >= 25:
            return "Medium"
        return "Low"

    supplier_risk["Risk Level"] = supplier_risk["Risk_Percentage"].apply(
        classify_risk
    )
    st.subheader("Supplier Performance Table")
    st.dataframe(supplier_risk, use_container_width=True, hide_index=True)

    st.subheader("📊 Supplier Inventory Risk")
    st.bar_chart(
        supplier_risk.set_index("supplier")["Risk_Percentage"],
        color="#38bdf8",
    )

    st.subheader("🚦 Supplier Risk Classification")
    for _, row in supplier_risk.iterrows():
        message = (
            f'{row["supplier"]} — {row["Risk Level"].upper()} RISK '
            f'({row["Risk_Percentage"]:.2f}%)'
        )
        if row["Risk Level"] == "High":
            st.error(message)
        elif row["Risk Level"] == "Medium":
            st.warning(message)
        else:
            st.success(message)

# =========================================================
# 10. DISRUPTION SIMULATOR
# =========================================================
elif page == "Disruption Simulator":
    st.header("⚠️ What-If Disruption Simulator")
    st.caption(
        "Explore a hypothetical disruption for one supplier. "
        "The scenario changes are assumptions, not a forecast."
    )

    left, right = st.columns(2)
    with left:
        selected_supplier = st.selectbox(
            "Select Supplier", sorted(data["supplier"].unique().tolist())
        )
        added_delay = st.slider("Additional Delivery Delay (days)", 0, 20, 7)
    with right:
        inventory_reduction = st.slider("Inventory Reduction (%)", 0, 50, 20)
        shipping_increase = st.slider("Additional Shipping Cost (%)", 0, 50, 15)

    if st.button("🚀 Run Disruption Simulation", use_container_width=True):
        scenario = data.copy()
        affected = scenario["supplier"] == selected_supplier

        scenario.loc[affected, "actual_delivery_days"] += added_delay
        scenario.loc[affected, "delivery_delay_days"] += added_delay

        # Convert integer inventory values to float before applying a
        # percentage reduction (for example, 4 * 0.8 = 3.2).
        scenario["inventory_days"] = pd.to_numeric(
            scenario["inventory_days"], errors="coerce"
        ).astype(float)
        scenario.loc[affected, "inventory_days"] = (
            scenario.loc[affected, "inventory_days"]
            * (1 - inventory_reduction / 100)
        )
        scenario["inventory_days"] = scenario["inventory_days"].clip(lower=0)

        baseline = calculate_metrics(data)
        disrupted = calculate_metrics(scenario)

        extra_shipping_cost = (
            scenario.loc[affected, "order_value"] * shipping_increase / 100
        ).sum()
        disrupted["Estimated Delay Cost"] += float(extra_shipping_cost)

        comparison = pd.DataFrame(
            {
                "Metric": list(baseline.keys()),
                "Baseline": list(baseline.values()),
                "Disrupted": list(disrupted.values()),
            }
        )

        st.divider()
        st.subheader("📊 Baseline vs Disrupted Scenario")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        added_delayed_orders = (
            disrupted["Delayed Orders"] - baseline["Delayed Orders"]
        )
        added_inventory_risks = (
            disrupted["Inventory Risk Orders"]
            - baseline["Inventory Risk Orders"]
        )
        added_cost = (
            disrupted["Estimated Delay Cost"]
            - baseline["Estimated Delay Cost"]
        )

        st.subheader("🚨 Disruption Impact Summary")
        x, y, z = st.columns(3)
        with x:
            render_kpi("Additional Delayed Orders", added_delayed_orders, "⏱️", "Change vs baseline", 0)
        with y:
            render_kpi("Additional Inventory Risk", added_inventory_risks, "⚠️", "Change vs baseline", 1)
        with z:
            render_kpi("Additional Estimated Cost", f"₹{added_cost:,.2f}", "💸", "Scenario estimate", 2)

        scenario_chart = pd.DataFrame(
            {
                "Delayed Orders": [
                    baseline["Delayed Orders"],
                    disrupted["Delayed Orders"],
                ],
                "Inventory Risk Orders": [
                    baseline["Inventory Risk Orders"],
                    disrupted["Inventory Risk Orders"],
                ],
            },
            index=["Baseline", "Disrupted"],
        )
        st.subheader("📈 Scenario Comparison")
        st.bar_chart(scenario_chart)

        if added_inventory_risks > 0 or added_delayed_orders > 0:
            st.warning(
                "This scenario increases supply-chain exposure. "
                "Review safety stock, delivery status and alternative suppliers."
            )
        else:
            st.success(
                "This scenario did not add delayed orders or inventory-risk orders "
                "in the uploaded dataset."
            )

# =========================================================
# 11. AI INVENTORY RISK PREDICTION
# =========================================================
elif page == "AI Risk Prediction":
    st.header("🤖 AI Inventory Risk Prediction")
    st.write(
        "Enter the details of a new order. The trained machine-learning "
        "pipeline will estimate its inventory-risk class."
    )

    # Use the same dataframe variable created after CSV upload: `data`.
    # This fixes the original NameError caused by referring to `df`.
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "chainsight_risk_model.pkl",
    )

    if not os.path.isfile(model_path):
        st.error(
            "AI model file not found. Place "
            "`chainsight_risk_model.pkl` in the same folder as `app.py`, "
            "then restart the app."
        )
        st.info(
            "If you have not trained the model yet, run "
            "`python train_model.py` from your ChainSight project folder."
        )
    else:
        try:
            model = joblib.load(model_path)
            st.success("AI model loaded successfully.")

            # These options now correctly come from `data`, not undefined `df`.
            supplier_options = sorted(data["supplier"].dropna().unique().tolist())
            product_options = sorted(data["product"].dropna().unique().tolist())
            disruption_options = sorted(
                data["disruption_type"].dropna().unique().tolist()
            )

            with st.form("inventory_prediction_form"):
                st.subheader("Enter New Order Details")
                col1, col2 = st.columns(2)

                with col1:
                    selected_supplier = st.selectbox(
                        "Supplier", supplier_options
                    )
                    selected_product = st.selectbox(
                        "Product", product_options
                    )
                    promised_delivery_days = st.number_input(
                        "Promised Delivery Days",
                        min_value=1,
                        value=7,
                        step=1,
                    )
                    inventory_days = st.number_input(
                        "Available Inventory (Days)",
                        min_value=0,
                        value=10,
                        step=1,
                    )

                with col2:
                    selected_disruption = st.selectbox(
                        "Disruption Type", disruption_options
                    )
                    demand_units = st.number_input(
                        "Demand Units", min_value=1, value=100, step=1
                    )
                    quantity = st.number_input(
                        "Order Quantity", min_value=1, value=150, step=1
                    )
                    unit_cost = st.number_input(
                        "Unit Cost",
                        min_value=0.01,
                        value=100.0,
                        step=1.0,
                    )

                submitted = st.form_submit_button(
                    "🔍 Predict Inventory Risk",
                    use_container_width=True,
                )

            if submitted:
                input_data = pd.DataFrame(
                    [
                        {
                            "supplier": selected_supplier,
                            "product": selected_product,
                            "promised_delivery_days": promised_delivery_days,
                            "inventory_days": inventory_days,
                            "demand_units": demand_units,
                            "quantity": quantity,
                            "unit_cost": unit_cost,
                            "disruption_type": selected_disruption,
                        }
                    ]
                )

                prediction = model.predict(input_data)[0]
                predicted_probability = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    model_classes = list(model.classes_)
                    if 1 in model_classes:
                        predicted_probability = (
                            float(probabilities[model_classes.index(1)]) * 100
                        )

                st.divider()
                st.subheader("AI Prediction Result")
                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    if int(prediction) == 1:
                        st.error("⚠️ HIGH INVENTORY RISK")
                    else:
                        st.success("✅ LOW INVENTORY RISK")

                with result_col2:
                    if predicted_probability is not None:
                        st.metric(
                            "Estimated Risk Probability",
                            f"{predicted_probability:.2f}%",
                        )
                    else:
                        st.info("This model does not provide probability estimates.")

                if predicted_probability is not None:
                    st.progress(
                        max(0, min(100, int(round(predicted_probability))))
                    )

                if int(prediction) == 1:
                    st.warning(
                        "The model predicts that delivery time may exceed "
                        "available inventory days."
                    )
                    st.subheader("Recommended Mitigation Actions")
                    st.markdown(
                        "- Review available safety stock.\n"
                        "- Consider an alternative supplier.\n"
                        "- Monitor supplier delivery status.\n"
                        "- Evaluate expedited shipping where practical."
                    )
                else:
                    st.success(
                        "The model predicts a lower inventory-risk class "
                        "for the supplied order details."
                    )
                    st.subheader("Recommended Actions")
                    st.markdown(
                        "- Continue monitoring inventory levels.\n"
                        "- Maintain regular supplier communication.\n"
                        "- Review demand changes periodically."
                    )

                st.caption(
                    "This is a model estimate based on its training data, "
                    "not a guarantee of an actual stockout. If the model was "
                    "trained on synthetic data, results are demonstration estimates."
                )

        except Exception as exc:
            st.error(f"AI Prediction Error: {exc}")
            st.warning(
                "If the error mentions scikit-learn version compatibility, "
                "retrain the model in the same Python environment used to run "
                "this Streamlit app. Do not reuse a pickle created with a "
                "different scikit-learn version."
            )

# =========================================================
# 12. FOOTER
# Use native Streamlit elements so HTML is not shown as page text.
# =========================================================
st.divider()
st.markdown("### 🔗 ChainSight")
st.caption("AI-Powered Supply Chain Intelligence Platform")
st.caption(
    "Predictions are estimates from the uploaded data and trained model. "
    "They should be reviewed before making operational decisions."
)

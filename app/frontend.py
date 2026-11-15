
import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from app.analytics import get_sensor_averages
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Machine Intelligence",
    page_icon="⚙️",
    layout="wide"
)

API_URL = os.getenv("API_URL") or st.secrets.get(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #344438 !important;
    }

    .subtitle {
        color: #718171;
        font-size: 15px;
    }

    .kpi-card {
        padding: 22px;
        border-radius: 16px;
        min-height: 125px;
    }

    .green-card {
        background-color: #DDEEDC;
    }

    .yellow-card {
        background-color: #FFF1B8;
    }

    .white-card {
    background-color: #F8F5E9;
    border: 1px solid #E9E3D2;
    }

    .kpi-label {
        color: #52765A;
        font-size: 13px;
        font-weight: 600;
    }

    .kpi-value {
        color: #344438;
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }

    div.stButton > button {
        background-color: #52765A;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #344438;
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #DDEEDC;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# SESSION STATE
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# -----------------------------------
# HEADER
# -----------------------------------

st.title("Predictive Maintenance Dashboard")
st.markdown(
    '<p class="subtitle">AI-powered predictive maintenance and machine health monitoring</p>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------------
# KPI CARDS
# -----------------------------------

st.subheader("Machine Overview")

result = st.session_state.latest_result

if result is None:
    status = "Awaiting prediction"
    probability = "—"
    status_class = "white-card"
else:
    status = result["machine_status"]
    probability = f'{result["failure_probability"]}%'
    status_class = (
        "green-card" if result["prediction"] == 0
        else "yellow-card"
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card {status_class}">
        <div class="kpi-label">MACHINE STATUS</div>
        <div class="kpi-value">{status}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card yellow-card">
        <div class="kpi-label">FAILURE PROBABILITY</div>
        <div class="kpi-value">{probability}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card white-card">
        <div class="kpi-label">TOTAL PREDICTIONS</div>
        <div class="kpi-value">{len(st.session_state.history)}</div>
    </div>
    """, unsafe_allow_html=True)


st.divider()


# -----------------------------------
# SENSOR INPUTS
# -----------------------------------

st.subheader("Machine Sensor Readings")

left, right = st.columns(2)

with left:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=310.0,
        value=300.5
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=320.0,
        value=310.2
    )

with right:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=1000,
        max_value=3000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=3.0,
        max_value=80.0,
        value=45.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=120
    )


st.divider()


# -----------------------------------
# SENSOR ANALYTICS
# -----------------------------------

st.divider()
st.subheader("Sensor Analytics")
st.caption(
    "Compare your current sensor readings against "
    "historical healthy and failed machine averages."
)

# Retrieve reference averages from original dataset
averages = get_sensor_averages()

# Map dashboard labels to dataset feature names
sensor_options = {
    "Torque (Nm)": ("Torque [Nm]", torque),
    "Tool Wear (min)": ("Tool wear [min]", tool_wear),
    "Rotational Speed (rpm)": ("Rotational speed [rpm]", rotational_speed),
    "Air Temperature (K)": ("Air temperature [K]", air_temperature),
    "Process Temperature (K)": ("Process temperature [K]", process_temperature)
}

selected_sensor = st.selectbox(
    "Select a sensor to compare",
    list(sensor_options.keys())
)

feature_name, current_value = sensor_options[selected_sensor]

healthy_avg = averages.loc[feature_name, "Healthy Average"]
failed_avg = averages.loc[feature_name, "Failed Average"]

# Display comparison metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Healthy Average",
        f"{healthy_avg:.2f}"
    )

with col2:
    st.metric(
        "Failed Average",
        f"{failed_avg:.2f}"
    )

with col3:
    st.metric(
        "Current Input",
        f"{current_value:.2f}",
        delta=f"{current_value - healthy_avg:+.2f} vs healthy average"
    )

# Create comparison chart

chart = go.Figure()

chart.add_trace(go.Bar(
    y=["Healthy Average"],
    x=[healthy_avg],
    name="Healthy Average",
    orientation="h",
    marker_color="#DDEEDC",
    text=[round(healthy_avg, 2)],
    textposition="outside"
))

chart.add_trace(go.Bar(
    y=["Failed Average"],
    x=[failed_avg],
    name="Failed Average",
    orientation="h",
    marker_color="#F4D66D",
    text=[round(failed_avg, 2)],
    textposition="outside"
))

chart.add_trace(go.Bar(
    y=["Current Input"],
    x=[current_value],
    name="Current Input",
    orientation="h",
    marker_color="#52765A",
    text=[round(current_value, 2)],
    textposition="outside"
))

chart.update_layout(
    title=f"{selected_sensor} Comparison",
    xaxis_title=selected_sensor,
    yaxis_title="",
    barmode="group",
    plot_bgcolor="#F8F5E9",
    paper_bgcolor="#F8F5E9",
    font=dict(color="#344438"),
    height=300,
    showlegend=False
)

st.plotly_chart(chart, use_container_width=True)




# -----------------------------------
# PREDICTION
# -----------------------------------

if st.button("Predict Machine Condition"):

    input_data = {
        "Type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=10
        )

        response.raise_for_status()

        prediction_result = response.json()

        st.session_state.latest_result = prediction_result

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Type": machine_type,
            "Air Temp": air_temperature,
            "Process Temp": process_temperature,
            "RPM": rotational_speed,
            "Torque": torque,
            "Tool Wear": tool_wear,
            "Status": prediction_result["machine_status"],
            "Failure Probability": prediction_result["failure_probability"]
        })

        
        st.subheader("Why did the model make this prediction?")

        shap_data = pd.DataFrame(prediction_result["shap_explanations"])

        shap_data = shap_data.sort_values(
        by="shap_value",
        key=abs,
        ascending=True
        )

        shap_data["color"] = shap_data["direction"].apply(
        lambda x: "Increases failure risk"
        if x == "Increases failure prediction"
        else "Decreases failure risk"
        )

        fig = px.bar(
        shap_data,
        x="shap_value",
        y="feature",
        color="color",
        orientation="h",
        color_discrete_map={
            "Increases failure risk": "#D96C5F",
            "Decreases failure risk": "#52765A"
        },
        labels={
            "shap_value": "SHAP contribution",
            "feature": "Sensor feature",
            "color": "Effect"
        },
        title="Factors influencing this prediction"
        )

        fig.update_layout(
        plot_bgcolor="#F8F5E9",
        paper_bgcolor="#F8F5E9",
        font_color="#344438"
        )

        st.plotly_chart(fig, use_container_width=True)

        # AUTOMATIC SHAP INTERPRETATION

        # AUTOMATIC SHAP INTERPRETATION

        st.subheader("AI-Generated Prediction Insights")

        # Ignore negligible SHAP contributions
        SIGNIFICANCE_THRESHOLD = 0.01

        significant_features = shap_data[
            shap_data["shap_value"].abs() >= SIGNIFICANCE_THRESHOLD
        ].copy()

        top_positive = significant_features[
            significant_features["shap_value"] > 0
        ].sort_values(
            by="shap_value",
            ascending=False
        )

        top_negative = significant_features[
            significant_features["shap_value"] < 0
        ].sort_values(
            by="shap_value"
        )

        if not top_positive.empty:
            strongest_risk = top_positive.iloc[0]

            st.warning(
                f"**{strongest_risk['feature']}** has the strongest "
                f"meaningful contribution towards failure prediction "
                f"({strongest_risk['shap_value']:+.4f})."
            )

        if not top_negative.empty:
            strongest_safe = top_negative.iloc[0]

            st.success(
                f"**{strongest_safe['feature']}** has the strongest "
                f"meaningful contribution away from failure prediction "
                f"({strongest_safe['shap_value']:+.4f})."
            )

        if significant_features.empty:
            st.info(
                "No individual feature has a SHAP contribution "
                "above the selected display threshold."
            )

        

    except requests.exceptions.RequestException as error:
        st.error(f"API connection failed: {error}")

      


# -----------------------------------
# DISPLAY LATEST RESULT
# -----------------------------------

if st.session_state.latest_result is not None:

    result = st.session_state.latest_result

    st.subheader("Latest Prediction")

    col1, col2 = st.columns(2)

    with col1:
        if result["prediction"] == 1:
            st.error("⚠️ Machine Failure Predicted")
        else:
            st.success("✅ Machine Predicted Healthy")

    with col2:
        st.metric(
            "Failure Probability",
            f'{result["failure_probability"]}%'
        )

    failure_probability = float(
    result["failure_probability"]
    )
        # Risk Level Indicator
    if failure_probability < 20:
        risk_level = "Low"
        risk_color = "success"
    elif failure_probability < 40:
        risk_level = "Moderate"
        risk_color = "warning"
    elif failure_probability < 60:
        risk_level = "High"
        risk_color = "error"
    else:
        risk_level = "Critical"
        risk_color = "error"

    st.markdown(f"### Risk Level: {risk_level}")

    st.progress(
        failure_probability / 100,
        text=f"Estimated failure probability: {failure_probability:.1f}%"
    )


# -----------------------------------
# PREDICTION HISTORY
# -----------------------------------

st.divider()

st.subheader("Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

        # =================================
    # PREDICTION ANALYTICS
    # =================================

    st.divider()

    st.header("Prediction Analytics")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        total_predictions = len(history_df)

        failure_count = (
            history_df["Status"] == "Failure"
        ).sum()

        healthy_count = (
            history_df["Status"] == "Healthy"
        ).sum()

        average_probability = (
            history_df["Failure Probability"].mean()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", total_predictions)

        with col2:
            st.metric("Healthy Predictions", healthy_count)

        with col3:
            st.metric("Failure Predictions", failure_count)

        st.metric(
            "Average Failure Probability",
            f"{average_probability:.2f}%"
        )

        # Prediction Distribution Chart

        st.subheader("Prediction Distribution")

        distribution_df = pd.DataFrame({
            "Machine Status": ["Healthy", "Failure"],
            "Count": [healthy_count, failure_count]
        })

        fig_distribution = px.pie(
            distribution_df,
            names="Machine Status",
            values="Count",
            hole=0.55,
            color="Machine Status",
            color_discrete_map={
                "Healthy": "#52765A",
                "Failure": "#D96C5F"
            },
            title="Healthy vs Failure Predictions"
        )

        fig_distribution.update_traces(
            textinfo="label+percent",
            textposition="inside"
        )

        fig_distribution.update_layout(
            paper_bgcolor="#F8F5E9",
            plot_bgcolor="#F8F5E9",
            font_color="#344438"
        )

        st.plotly_chart(fig_distribution, use_container_width=True)

    else:
        st.info("Make some predictions to see analytics.")


else:
    st.info("No predictions recorded yet. Enter sensor readings and run your first prediction.")

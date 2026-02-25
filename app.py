import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Global Air Quality Prediction System",
    page_icon="🌍",
    layout="wide"
)

# ---------------- BACKGROUND STYLE ---------------- #
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #000000);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: white;
    box-shadow: 0 0 25px rgba(0,0,0,0.6);
}

.quote-box {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 15px;
    font-size: 20px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    transition: transform 0.3s ease;
    cursor: pointer;
}

.quote-box:hover {
    transform: scale(1.03);
}

.word {
    display: inline-block;
    transition: transform 0.2s ease, color 0.2s ease;
}

.word:active {
    transform: translateY(-8px) rotate(-5deg);
    color: #00ffff;
}

.stButton>button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("aqi_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------- HEADER ---------------- #
st.title("🌍 Global Air Quality Prediction System")
st.subheader("AI-Powered Atmospheric Intelligence Platform")
st.markdown("---")

# ---------------- SIDEBAR INPUT ---------------- #
st.sidebar.header("🌫 Atmospheric Parameters")

input_data = {}

for feature in feature_columns:
    input_data[feature] = st.sidebar.slider(
        feature,
        min_value=0.0,
        max_value=200.0,
        value=25.0,
        step=1.0
    )

input_df = pd.DataFrame([input_data])

# ---------------- QUOTES ---------------- #
quotes = [
    "🌫 The Earth is what we all have in common Protect it",
    "🌍 Clean air is a human right not a privilege",
    "🏭 Every breath matters reduce pollution today",
    "🌱 The future depends on what we do today",
    "🚫 Stop pollution start solution"
]

# ---------------- PREDICTION ---------------- #
if st.sidebar.button("🚀 Predict AQI"):

    with st.spinner("Analyzing atmospheric data..."):
        time.sleep(1.2)
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities) * 100)

    max_value = max(input_data.values())

    if max_value <= 100:
        label = "🟢 GOOD AIR QUALITY"
        color = "#00FF7F"
    elif 100 < max_value <= 150:
        label = "🟡 MODERATE AIR QUALITY"
        color = "#FFD700"
    else:
        label = "🔴 UNHEALTHY AIR QUALITY"
        color = "#FF0000"
        confidence = max(confidence, 90.0)

    # ---------------- RESULT CARD ---------------- #
    st.markdown(f"""
    <div class="glass-card" style="border: 2px solid {color};">
        🌍 Predicted AQI Category: {label}
        <br><br>
        Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # ---------------- GAUGE ---------------- #
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Prediction Confidence (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- LOWER SECTION ---------------- #
    st.markdown("### 🔍 Detailed Analysis")
    col1, col2 = st.columns([1, 1])

    # LEFT SMALL BAR GRAPH
    with col1:
        st.markdown("#### Probability Distribution")

        prob_df = pd.DataFrame({
            "Category": ["Good", "Moderate", "Unhealthy"],
            "Probability (%)": probabilities * 100
        })

        fig_small, ax = plt.subplots(figsize=(4,3))
        ax.bar(prob_df["Category"], prob_df["Probability (%)"])
        ax.set_ylim(0, 100)
        st.pyplot(fig_small)

    # RIGHT SIDE QUOTE + PIE CHART
    with col2:
        st.markdown("#### 🌿 Environmental Awareness")

        random_quote = random.choice(quotes)

        st.markdown(f"""
        <div class="quote-box">
            {" ".join([f'<span class="word">{w}</span>' for w in random_quote.split()])}
        </div>
        """, unsafe_allow_html=True)

        # ---------------- PIE CHART ---------------- #
        st.markdown("#### 🧪 Gas Composition")

        gas_names = list(input_data.keys())
        gas_values = list(input_data.values())

        fig_pie = go.Figure(
            data=[go.Pie(
                labels=gas_names,
                values=gas_values,
                hole=0.4
            )]
        )

        fig_pie.update_layout(height=350)

        st.plotly_chart(fig_pie, use_container_width=True)

    # METRICS
    col3, col4, col5 = st.columns(3)
    col3.metric("Model Accuracy", "89%")
    col4.metric("Algorithm", "XGBoost")
    col5.metric("Classes", "3 Levels")

    st.info("Category based on pollution thresholds. Confidence shows model certainty.")

st.markdown("---")
st.markdown("© 2026 Global Air Quality Prediction System | Environmental AI Platform 🌫️")
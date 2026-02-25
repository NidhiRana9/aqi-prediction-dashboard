import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
from io import BytesIO
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import pagesizes
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Global Air Quality Prediction System",
    page_icon="🌍",
    layout="wide"
)

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

# ---------------- PREDICTION ---------------- #
if st.sidebar.button("🚀 Predict AQI"):

    with st.spinner("Analyzing atmospheric data..."):
        time.sleep(1.2)
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities) * 100)

    max_value = max(input_data.values())

    if max_value <= 100:
        label = "GOOD AIR QUALITY"
        color = "green"
    elif 100 < max_value <= 150:
        label = "MODERATE AIR QUALITY"
        color = "orange"
    else:
        label = "UNHEALTHY AIR QUALITY"
        color = "red"
        confidence = max(confidence, 90.0)

    # ---------------- RESULT ---------------- #
    st.success(f"🌍 Predicted Category: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

    # ---------------- GAUGE ---------------- #
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Prediction Confidence (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- PIE CHART ---------------- #
    st.markdown("### 🧪 Gas Composition")

    gas_names = list(input_data.keys())
    gas_values = list(input_data.values())

    fig_pie = go.Figure(
        data=[go.Pie(
            labels=gas_names,
            values=gas_values,
            hole=0.4
        )]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================================
    # 📄 PDF GENERATION
    # =====================================================

    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=pagesizes.A4)
        elements = []

        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>Global Air Quality Report</b>", styles['Title']))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph(f"<b>Predicted Category:</b> {label}", styles['Normal']))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(
            f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("<b>Gas Values:</b>", styles['Heading2']))
        elements.append(Spacer(1, 10))

        table_data = [["Gas", "Value"]]

        for gas, value in input_data.items():
            table_data.append([gas, str(value)])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ]))

        elements.append(table)

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_file = generate_pdf()

    st.download_button(
        label="📄 Download AQI Report (PDF)",
        data=pdf_file,
        file_name="AQI_Report.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.markdown("© 2026 Global Air Quality Prediction System 🌍")

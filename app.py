import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Health Dashboard", layout="wide")

# Dummy patient data
patient_data = {
    "Patient ID": "123456",
    "Visit Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Height": 178,
    "Weight": 80,
    "BMI": 26.1,
    "Systolic_BP": 130,
    "Diastolic_BP": 85,
    "Heart_Rate": 72,
    "Smoking_Status": "Non-smoker",
    "Health_Score": 75,
    "Heart_Disease_Risk_Score": 20,
    "Risk_Level": "Low",
    "Preventive_Measures": [
        "Maintain a healthy diet",
        "Exercise regularly",
        "Monitor blood pressure"
    ]
}

def donut_chart(score, label, color):
    fig = go.Figure(go.Pie(
        values=[score, 100 - score],
        hole=0.75,
        marker_colors=[color, "#f2f2f2"],
        textinfo='none'
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(text=f"<b>{score}</b><br>{label}", showarrow=False, font=dict(size=16))]
    )
    return fig

# CSS for styling
st.markdown("""
    <style>
    .card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
        font-size: 14px;
    }
    .section-title {
        font-weight: bold;
        font-size: 18px;
        margin-top: 1rem;
    }
    .metric-title {
        font-weight: 500;
        color: #333;
        margin-bottom: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Patient Health Dashboard")

# ───────────── Patient Info & Health Metrics ─────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Personal Info")
    st.write(f"**Patient ID:** {patient_data['Patient ID']}")
    st.write(f"**Visit Date:** {patient_data['Visit Date']}")
    st.write(f"**Height:** {patient_data['Height']} cm")
    st.write(f"**Weight:** {patient_data['Weight']} kg")
    st.write(f"**Smoking Status:** {patient_data['Smoking_Status']}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Health Metrics")
    st.write(f"**BMI:** {patient_data['BMI']}")
    st.write(f"**Blood Pressure:** {patient_data['Systolic_BP']}/{patient_data['Diastolic_BP']} mmHg")
    st.write(f"**Heart Rate:** {patient_data['Heart_Rate']} BPM")
    st.write(f"**Risk Level:** {patient_data['Risk_Level']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ───────────── Risk Score and Health Score ─────────────
st.markdown("#### 🛡️ Risk Assessment")

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 💓 Heart Disease Risk")
    st.plotly_chart(donut_chart(patient_data["Heart_Disease_Risk_Score"], "Risk", "#1f77b4"), use_container_width=True)
    st.markdown("**Preventive Measures:**")
    for tip in patient_data["Preventive_Measures"]:
        st.markdown(f"✔️ {tip}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 💯 Health Score")
    st.plotly_chart(donut_chart(patient_data["Health_Score"], "Score", "#ff7f0e"), use_container_width=True)
    st.markdown("Good health with some areas for improvement.")
    st.markdown('</div>', unsafe_allow_html=True)

# ───────────── Calendar Booking ─────────────
st.markdown("#### 📅 Book Appointment")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    date = st.date_input("Select Appointment Date")
    dept = st.selectbox("Select Department", ["Cardiology", "General Medicine", "Orthopedics", "Nutrition"])
    time = st.selectbox("Preferred Time", ["9:00 AM", "11:00 AM", "1:00 PM", "3:00 PM", "5:00 PM"])
    if st.button("Book Now"):
        st.success(f"Appointment booked on {date} with {dept} at {time}")
    st.markdown('</div>', unsafe_allow_html=True)

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import joblib

# Page config
st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("selected_20_final_patients.csv")

df = load_data()

# Donut chart generator
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

# UI Styling
st.markdown("""
    <style>
    .card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.04);
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.patient_id = ""

def show_login():
    st.title("Welcome to HealthPredict 🩺")
    st.subheader("Login with your Patient ID")
    patient_id = st.text_input("Enter Patient ID")
    if st.button("Login"):
        if patient_id in df["patient"].astype(str).values:
            st.session_state.logged_in = True
            st.session_state.patient_id = patient_id
            st.rerun()
        else:
            st.error("Invalid Patient ID. Please try again.")

def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")
    if patient_df.empty:
        st.error("No data available for this patient.")
        return
    latest = patient_df.iloc[-1]
    tab1, tab2 = st.tabs(["📊 Patient Overview", "📅 Visit History"])

    with tab1:
        st.title("🏥 Patient Health Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 👤 Personal Info")
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Visit Date:** {latest['date']}")
            st.write(f"**Height:** {latest['Height_cm']} cm")
            st.write(f"**Weight:** {latest['Weight_kg']} kg")
            st.write(f"**Smoking Status:** {latest['Smoking_Status']}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Health Metrics")
            st.write(f"**BMI:** {latest['BMI']}")
            st.write(f"**Blood Pressure:** {latest['Systolic_BP']}/{latest['Diastolic_BP']} mmHg")
            st.write(f"**Heart Rate:** {latest['Heart_Rate']} BPM")
            st.write(f"**Risk Level:** {latest['Risk_Level']}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🛡️ Risk Assessment")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 💓 Heart Disease Risk")
            try:
                if not os.path.exists("heart_disease_model.pkl"):
                    st.error("❌ Model file not found.")
                else:
                    model = joblib.load("heart_disease_model.pkl")
                    input_df = pd.DataFrame([{
                        "Height_cm": latest["Height_cm"],
                        "BMI": latest["BMI"],
                        "Weight_kg": latest["Weight_kg"],
                        "Diastolic_BP": latest["Diastolic_BP"],
                        "Heart_Rate": latest["Heart_Rate"],
                        "Systolic_BP": latest["Systolic_BP"],
                        "Diabetes": latest.get("Diabetes", 0),
                        "Hyperlipidemia": latest.get("Hyperlipidemia", 0),
                        "Smoking_Status": str(latest.get("Smoking_Status", ""))
                    }])
                    prediction = model.predict(input_df)[0]
                    confidence = model.predict_proba(input_df)[0][prediction] * 100
                    risk_score = int(confidence)
                    color = "#e74c3c" if prediction else "#2ecc71"
                    st.plotly_chart(donut_chart(risk_score, "Risk", color), use_container_width=True)
                    st.markdown("**Preventive Measures:**")
                    if prediction:
                        st.write("- Schedule cardiac checkup")
                        st.write("- Monitor BP & cholesterol")
                        st.write("- Adopt heart-healthy habits")
                    else:
                        st.write("- Maintain current lifestyle")
                        st.write("- Stay active")
                        st.write("- Regular checkups")
            except Exception as e:
                st.error(f"Model error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 💯 Health Score")
            st.plotly_chart(donut_chart(latest["Health_Score"], "Score", "#ff7f0e"), use_container_width=True)
            st.markdown("Good health with some areas for improvement.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🧠 Preventive Measures")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        bmi = latest["BMI"]
        hr = latest["Heart_Rate"]
        sys = latest["Systolic_BP"]
        if bmi < 18.5 or bmi > 25:
            st.write(f"• Adjust BMI (Current: {bmi}) – Balanced diet & exercise.")
        if hr > 90:
            st.write(f"• High Heart Rate ({hr} bpm) – Stress reduction & exercise.")
        if sys > 130:
            st.write(f"• Blood Pressure ({sys} mmHg) – Reduce salt & stay active.")
        if latest["Smoking_Status"].lower().startswith("current"):
            st.write("• Smoking Cessation – Enroll in quit programs.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("📅 Visit History")
        st.line_chart(patient_df.set_index(pd.to_datetime(patient_df["date"]))["Health_Score"])
        for _, row in patient_df.iterrows():
            with st.expander(f"Visit on {row['date']}"):
                st.write(f"**Height:** {row['Height_cm']} cm")
                st.write(f"**Weight:** {row['Weight_kg']} kg")
                st.write(f"**BMI:** {row['BMI']}")
                st.write(f"**Blood Pressure:** {row['Systolic_BP']}/{row['Diastolic_BP']}")
                st.write(f"**Heart Rate:** {row['Heart_Rate']}")
                st.write(f"**Smoking Status:** {row['Smoking_Status']}")
                st.write(f"**Health Score:** {row['Health_Score']}")
                st.write(f"**Risk Level:** {row['Risk_Level']}")

# Main app logic
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

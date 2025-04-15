# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("selected_20_final_patients.csv")

df = load_data()

# Appointment Calendar – shown only after login
def sidebar_appointment():
    st.sidebar.markdown("## 📅 Book an Appointment")
    doctor = st.sidebar.selectbox("Select Doctor Type", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
    appt_date = st.sidebar.date_input("Choose Date", min_value=date.today())
    notes = st.sidebar.text_input("Any Notes?", key="notes")
    if st.sidebar.button("Book Appointment"):
        st.sidebar.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

# Donut chart generator
def donut_chart(label, color):
    fig = go.Figure(data=[go.Pie(
        values=[70, 30],  # dummy values just for visual
        hole=0.75,
        marker_colors=[color, "#f0f2f6"],
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=5, b=5, l=5, r=5),
        height=160,
        width=160,
        annotations=[dict(
            text=f"<b>{label}</b>",
            font_size=14,
            showarrow=False
        )]
    )
    return fig

# Login session
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
    sidebar_appointment()

    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")
    latest = patient_df.iloc[-1]
    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    with tab1:
        st.markdown("## 👤 Patient Overview")
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.markdown("#### Personal Info")
                st.write(f"- **Patient ID:** {patient_id}")
                st.write(f"- **Date:** {latest['date']}")
                st.write(f"- **Height:** {latest['Height_cm']} cm")
                st.write(f"- **Weight:** {latest['Weight_kg']} kg")
                st.write(f"- **Smoking:** {latest['Smoking_Status']}")

        with col2:
            with st.container():
                st.markdown("#### Health Metrics")
                st.metric("BMI", latest["BMI"])
                st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
                st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Health Score")
            risk_level = latest["Risk_Level"].lower()
            score_color = "#4caf50" if "low" in risk_level else "#ffa94d" if "medium" in risk_level else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score_color), use_container_width=True)

        with col4:
            st.markdown("#### Heart Disease Risk")
            try:
                model = joblib.load("heart_disease_model.pkl")
                input_df = pd.DataFrame([{
                    "Height_cm": latest.get("Height_cm"),
                    "BMI": latest.get("BMI"),
                    "Weight_kg": latest.get("Weight_kg"),
                    "Diastolic_BP": latest.get("Diastolic_BP"),
                    "Heart_Rate": latest.get("Heart_Rate"),
                    "Systolic_BP": latest.get("Systolic_BP"),
                    "Diabetes": latest.get("Diabetes"),
                    "Hyperlipidemia": latest.get("Hyperlipidemia"),
                    "Smoking_Status": str(latest.get("Smoking_Status"))
                }])
                prediction = model.predict(input_df)[0]
                risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
                risk_label = "High Risk" if prediction == 1 else "Low Risk"
                st.plotly_chart(donut_chart(risk_label, risk_color), use_container_width=True)
            except Exception as e:
                st.error(f"Model error: {e}")

        st.markdown("#### 🛡️ Preventive Measures")
        with st.container():
            bmi = latest["BMI"]
            hr = latest["Heart_Rate"]
            sys = latest["Systolic_BP"]
            if bmi < 18.5 or bmi > 25:
                st.write(f"• Adjust BMI – Current: {bmi}")
            if hr > 90:
                st.write(f"• High Heart Rate ({hr} bpm) – Exercise & stress control.")
            if sys > 130:
                st.write(f"• High Blood Pressure ({sys} mmHg) – Diet & monitoring.")
            if str(latest["Smoking_Status"]).lower().startswith("current"):
                st.write("• Quit Smoking – For long-term heart and lung health.")
            if (
                18.5 <= bmi <= 25 and hr <= 90 and sys <= 130 and
                not str(latest["Smoking_Status"]).lower().startswith("current")
            ):
                st.write("✅ All health parameters in optimal range.")

    with tab2:
        st.markdown("## 📅 Visit History")
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

                st.markdown("**Preventive Measures:**")
                if row["BMI"] < 18.5 or row["BMI"] > 25:
                    st.write(f"• BMI: {row['BMI']} – Balanced diet/exercise.")
                if row["Heart_Rate"] > 90:
                    st.write(f"• Heart Rate: {row['Heart_Rate']} – Reduce stress.")
                if row["Systolic_BP"] > 130:
                    st.write(f"• BP: {row['Systolic_BP']} – Monitor regularly.")
                if str(row["Smoking_Status"]).lower().startswith("current"):
                    st.write("• Smoking – Consider quitting.")
                if (
                    18.5 <= row["BMI"] <= 25 and row["Heart_Rate"] <= 90 and row["Systolic_BP"] <= 130 and
                    not str(row["Smoking_Status"]).lower().startswith("current")
                ):
                    st.write("✅ All vitals within recommended limits.")

    st.markdown("---")
    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

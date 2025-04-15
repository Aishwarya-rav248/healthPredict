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

# Sidebar calendar and appointment form
with st.sidebar:
    st.markdown("## 📅 Book an Appointment")
    doctor = st.selectbox("Select Doctor Type", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
    appt_date = st.date_input("Choose Date", min_value=date.today())
    st.text_input("Any Notes?", key="notes")
    if st.button("Book Appointment"):
        st.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

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

def donut_chart(label, value, color):
    fig = go.Figure(data=[go.Pie(
        values=[value, 100 - value],
        hole=0.75,
        marker_colors=[color, "#f0f2f6"],
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=180,
        width=180,
        annotations=[dict(
            text=f"<b>{value:.0f}</b><br>{label}",
            font_size=14,
            showarrow=False
        )]
    )
    return fig

def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")
    latest = patient_df.iloc[-1]

    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    with tab1:
        st.markdown("## 🧑‍⚕️ Patient Overview")

        # Top row: Patient Info + Metrics
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 👤 Personal Info")
            st.markdown(f"- **Patient ID**: {patient_id}")
            st.markdown(f"- **Date**: {latest['date']}")
            st.markdown(f"- **Height**: {latest['Height_cm']} cm")
            st.markdown(f"- **Weight**: {latest['Weight_kg']} kg")
            st.markdown(f"- **Smoking**: {latest['Smoking_Status']}")
        with c2:
            st.markdown("### 📈 Key Metrics")
            st.metric("BMI", latest["BMI"])
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")

        # Middle row: Donut Charts
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 🧬 Health Score")
            score = latest["Health_Score"]
            risk_level = latest["Risk_Level"].lower()
            score_color = "#4caf50" if "low" in risk_level else "#ffa94d" if "medium" in risk_level else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, score_color), use_container_width=True)

        with c4:
            st.markdown("### 🧠 Heart Risk")
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
                confidence = model.predict_proba(input_df)[0][prediction] * 100
                risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart("Risk", confidence, risk_color), use_container_width=True)
            except Exception as e:
                st.error(f"Risk model error: {e}")

        # Bottom row: Preventive Measures
        st.markdown("### 🛡️ Preventive Measures")
        with st.container():
            bmi = latest["BMI"]
            hr = latest["Heart_Rate"]
            sys = latest["Systolic_BP"]
            if bmi < 18.5 or bmi > 25:
                st.write(f"• Adjust BMI (Current: {bmi}) – Balanced diet & exercise recommended.")
            if hr > 90:
                st.write(f"• High Heart Rate ({hr} bpm) – Try stress management & physical activity.")
            if sys > 130:
                st.write(f"• Blood Pressure ({sys} mmHg) – Limit salt, regular checkups needed.")
            if str(latest["Smoking_Status"]).lower().startswith("current"):
                st.write("• Smoking – Enroll in cessation programs for heart health.")
    
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

                st.markdown("**Visit Recommendations:**")
                if row["BMI"] < 18.5 or row["BMI"] > 25:
                    st.write(f"• BMI: {row['BMI']} – Consider dietary adjustment or activity.")
                if row["Heart_Rate"] > 90:
                    st.write(f"• Heart Rate: {row['Heart_Rate']} bpm – Reduce stress and exercise.")
                if row["Systolic_BP"] > 130:
                    st.write(f"• Systolic BP: {row['Systolic_BP']} mmHg – Lower salt & monitor BP.")
                if str(row["Smoking_Status"]).lower().startswith("current"):
                    st.write("• Smoking – Quit for better heart & lung function.")

    st.markdown("---")
    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

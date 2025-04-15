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

def donut_chart(label, value, color, show_value=False):
    fig = go.Figure(data=[go.Pie(
        values=[value, 100 - value],
        hole=0.7,
        marker_colors=[color, "#f0f2f6"],
        textinfo='none'
    )])
    fig.update_layout(
        height=170,
        width=170,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(
            text=f"<b>{label}</b>" if not show_value else f"<b>{value}%</b><br>{label}",
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

        # 🗓 Appointment calendar — inside tab1 only
        with st.sidebar:
            st.markdown("## 📅 Book Appointment")
            doctor = st.selectbox("Doctor Type", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
            appt_date = st.date_input("Choose Date", min_value=date.today())
            st.text_input("Any Notes?", key="notes")
            if st.button("Book Appointment"):
                st.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

        # 👤 Personal Info + Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👤 Patient Info")
            st.markdown(f"- **ID**: {patient_id}")
            st.markdown(f"- **Date**: {latest['date']}")
            st.markdown(f"- **Height**: {latest['Height_cm']} cm")
            st.markdown(f"- **Weight**: {latest['Weight_kg']} kg")
            st.markdown(f"- **Smoking**: {latest['Smoking_Status']}")
        with col2:
            st.markdown("### 📈 Vitals")
            st.metric("BMI", latest["BMI"])
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")

        # 🧬 Health Score + 🧠 Heart Risk
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### 🧬 Health Score")
            score = latest["Health_Score"]
            score_color = "#4caf50" if "low" in latest["Risk_Level"].lower() else "#ffa94d" if "medium" in latest["Risk_Level"].lower() else "#ff4d4d"
            st.plotly_chart(donut_chart("Health Score", score, score_color, show_value=True), use_container_width=True)

        with col4:
            st.markdown("### ❤️ Heart Disease Risk")
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
                risk_label = "High Risk" if prediction == 1 else "Low Risk"
                risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart(risk_label, 100, risk_color), use_container_width=True)
            except Exception as e:
                st.error(f"Model error: {str(e)}")

        # 🛡️ Preventive Measures
        st.markdown("### 🛡️ Preventive Measures")
        bmi = latest["BMI"]
        hr = latest["Heart_Rate"]
        sys = latest["Systolic_BP"]
        if bmi < 18.5 or bmi > 25:
            st.write(f"• Adjust BMI (Current: {bmi}) – Balanced diet & exercise recommended.")
        if hr > 90:
            st.write(f"• High Heart Rate ({hr} bpm) – Stress management advised.")
        if sys > 130:
            st.write(f"• Blood Pressure ({sys} mmHg) – Reduce salt & monitor BP.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            st.write("• Smoking – Consider cessation programs for better heart health.")

    # 📅 Visit History
    with tab2:
         st.markdown("### 📅 Visit History")
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

                st.markdown("**Preventive Tips:**")
                if row["BMI"] < 18.5 or row["BMI"] > 25:
                    st.write(f"• BMI: {row['BMI']} – Consider diet or physical activity changes.")
                if row["Heart_Rate"] > 90:
                    st.write(f"• Heart Rate: {row['Heart_Rate']} bpm – Try meditation, exercise.")
                if row["Systolic_BP"] > 130:
                    st.write(f"• Blood Pressure: {row['Systolic_BP']} mmHg – Reduce sodium intake.")
                if str(row["Smoking_Status"]).lower().startswith("current"):
                    st.write("• Smoking – Join cessation programs for long-term benefits.")

    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# 🔑 Entry
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

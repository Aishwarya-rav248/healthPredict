
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Cleaned Dataset.csv")

df = load_data()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.patient_id = ""

def donut_chart(label, value, color, show_score=True):
    text = f"<b>{value:.0f}</b><br>{label}" if show_score else f"<b>{label}</b>"
    fig = go.Figure(data=[go.Pie(
        values=[value if show_score else 50, 100 - value if show_score else 50],
        hole=0.75,
        marker_colors=[color, "#f0f2f6"],
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=180,
        width=180,
        annotations=[dict(text=text, font_size=14, showarrow=False)]
    )
    return fig

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
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("Date")
    latest = patient_df.iloc[-1]

    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    with tab1:
        with st.sidebar:
            st.markdown("## 📅 Book an Appointment")
            doctor = st.selectbox("Select Doctor Type", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
            appt_date = st.date_input("Choose Date", min_value=date.today())
            st.text_input("Any Notes?", key="notes")
            if st.button("Book Appointment"):
                st.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

        st.markdown("## 🧑‍⚕️ Patient Overview")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 👤 Personal Info")
            st.markdown(f"- **Patient ID**: {patient_id}")
            st.markdown(f"- **Date**: {latest['Date']}")
            st.markdown(f"- **Height**: {latest['Height_cm']} cm")
            st.markdown(f"- **Weight**: {latest['Weight_kg']} kg")
            st.markdown(f"- **Smoking**: {latest['Smoking_Status']}")
        with c2:
            st.markdown("### 📈 Key Metrics")
            st.metric("BMI", latest["BMI"])
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 🧬 Health Score")
            score = latest["Health_Score"]
            color = "#4caf50" if score >= 80 else "#ffa94d" if score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, color), use_container_width=True)

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
                risk_label = "High Risk" if prediction == 1 else "Low Risk"
                risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart(risk_label, 50, risk_color, show_score=False), use_container_width=True)
            except Exception as e:
                st.error(f"Risk model error: {e}")

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
        total_visits = len(patient_df)
        avg_score = round(patient_df["Health_Score"].mean(), 2)
        st.info(f"**Total Visits:** {total_visits} | **Average Score:** {avg_score}")

        metric_options = ["BMI", "Health_Score", "Systolic_BP", "Diastolic_BP", "Heart_Rate"]
        metric_choice = st.selectbox("Select Metric to View Trend", metric_options)
        st.line_chart(patient_df.set_index(pd.to_datetime(patient_df["Date"]))[metric_choice])

        for _, row in patient_df.iterrows():
            risk_color = "#ff4d4d" if row["Heart_Disease"] == 1 else "#4caf50"
            with st.container():
                st.markdown(
                    f"<div style='border:1px solid #ccc; border-radius:10px; padding:15px; margin-bottom:20px; background:#f9f9f9;'>"
                    f"<b>🗓️ Visit on {row['Date']}</b><br>"
                    f"<b>Height:</b> {row['Height_cm']} cm | <b>Weight:</b> {row['Weight_kg']} kg | <b>BMI:</b> {row['BMI']}<br>"
                    f"<b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']} | <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>"
                    f"<b>Health Score:</b> {row['Health_Score']} | "
                    f"<b>Heart Risk:</b> <span style='color:white;background:{risk_color};padding:2px 6px;border-radius:5px;'>"
                    f"{'High' if row['Heart_Disease']==1 else 'Low'}</span><br><br>"
                    f"<b>🛡️ Tips:</b><br>"
                    + ("• BMI outside healthy range.<br>" if row['BMI'] < 18.5 or row['BMI'] > 25 else "") +
                      ("• High Heart Rate – reduce stress.<br>" if row['Heart_Rate'] > 90 else "") +
                      ("• High BP – monitor regularly.<br>" if row['Systolic_BP'] > 130 else "") +
                      ("• Smoking – consider quitting.<br>" if str(row['Smoking_Status']).lower().startswith("current") else "") +
                    "</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

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
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")
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
            st.markdown(f"- **Date**: {latest['date']}")
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
            st.markdown("###  Health Score")
            score = latest["Health_Score"]
            level = latest["Risk_Level"].lower()
            color = "#4caf50" if "low" in level else "#ffa94d" if "medium" in level else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, color), use_container_width=True)

        with c4:
            st.markdown("###  Heart Risk")
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
        high_risk_pct = (patient_df["Risk_Level"].str.lower() == "high").mean() * 100
        st.info(f"**Total Visits:** {total_visits} | **Avg Score:** {avg_score} | **High Risk Visits:** {high_risk_pct:.0f}%")

        # Trend toggle
        st.markdown("### 📈 Health Metric Trends")
        chart_options = {
            "Health Score": "Health_Score",
            "BMI": "BMI",
            "Heart Rate": "Heart_Rate",
            "Systolic BP": "Systolic_BP",
            "Diastolic BP": "Diastolic_BP"
        }
        selected_label = st.selectbox("Choose Metric", list(chart_options.keys()))
        selected_metric = chart_options[selected_label]
        chart_data = patient_df.set_index(pd.to_datetime(patient_df["date"]))[selected_metric]
        st.line_chart(chart_data)

        # Visit Cards
        for _, row in patient_df.iterrows():
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"🗓️ **Visit on {pd.to_datetime(row['date']).strftime('%Y-%m-%d')}**")
                st.write(f"• Height: {row['Height_cm']} cm")
                st.write(f"• Weight: {row['Weight_kg']} kg")
                st.write(f"• BMI: {row['BMI']}")
                st.write(f"• Smoking Status: {row['Smoking_Status']}")
            with col2:
                st.write(f"• Blood Pressure: {row['Systolic_BP']}/{row['Diastolic_BP']}")
                st.write(f"• Heart Rate: {row['Heart_Rate']} bpm")
                st.write(f"• Health Score: {row['Health_Score']}")
                st.write(f"• Risk Level: {row['Risk_Level']}")

            st.write("**🛡️ Preventive Tips:**")
            if row["BMI"] < 18.5 or row["BMI"] > 25:
                st.write("• BMI outside healthy range – adjust diet & activity.")
            if row["Heart_Rate"] > 90:
                st.write("• High Heart Rate – reduce stress, exercise more.")
            if row["Systolic_BP"] > 130:
                st.write("• High BP – limit sodium, regular monitoring.")
            if str(row["Smoking_Status"]).lower().startswith("current"):
                st.write("• Smoking – consider cessation support.")

    st.markdown("---")
    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# MAIN
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

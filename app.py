import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import date
import os

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("selected_20_final_patients.csv")

df = load_data()

def donut_chart(score, label, color):
    fig = go.Figure(go.Pie(
        values=[score, 100 - score],
        hole=0.75,
        marker_colors=[color, "#f0f2f6"],
        textinfo='none'
    ))
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=160,
        width=160,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{score}</b><br>{label}",
            font_size=14,
            showarrow=False
        )]
    )
    return fig

def appointment_calendar():
    st.date_input("📅 Book Appointment", min_value=date.today(), label_visibility="collapsed")

def show_login():
    st.title("Welcome to HealthPredict 🩺")
    patient_id = st.text_input("Enter Patient ID")
    if st.button("Login"):
        if patient_id in df["patient"].astype(str).values:
            st.session_state.logged_in = True
            st.session_state.patient_id = patient_id
            st.rerun()
        else:
            st.error("Invalid Patient ID")

def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")
    if patient_df.empty:
        st.error("No data for this patient.")
        return

    latest = patient_df.iloc[-1]
    tab1, tab2 = st.tabs(["🏥 Overview", "📅 Visit History"])

    with tab1:
        st.markdown("## 👤 Patient Overview")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div style="background:#fff;padding:15px;border-radius:10px;">', unsafe_allow_html=True)
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Visit Date:** {latest['date']}")
            st.write(f"**Height:** {latest['Height_cm']} cm")
            st.write(f"**Weight:** {latest['Weight_kg']} kg")
            st.write(f"**Smoking Status:** {latest['Smoking_Status']}")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div style="background:#fff;padding:15px;border-radius:10px;">', unsafe_allow_html=True)
            st.write(f"**BMI:** {latest['BMI']}")
            st.write(f"**Blood Pressure:** {latest['Systolic_BP']}/{latest['Diastolic_BP']} mmHg")
            st.write(f"**Heart Rate:** {latest['Heart_Rate']} BPM")
            st.write(f"**Risk Level:** {latest['Risk_Level']}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("## 🧠 Risk Assessment")
        col3, col4 = st.columns([1, 1])
        with col3:
            st.markdown("#### ❤️ Heart Disease Risk")
            try:
                if os.path.exists("heart_disease_model.pkl"):
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
                    color = "#ff4d4d" if prediction == 1 else "#2ecc71"
                    label = "High" if prediction == 1 else "Low"
                    st.plotly_chart(donut_chart(int(confidence), label, color), use_container_width=True)
                    st.markdown("**Preventive Measures:**")
                    if prediction == 1:
                        st.write("• Schedule cardiac checkup")
                        st.write("• Monitor BP & cholesterol")
                        st.write("• Adopt heart-healthy habits")
                    else:
                        st.write("• Maintain current lifestyle")
                        st.write("• Routine checkups")
                else:
                    st.warning("Model not found.")
            except Exception as e:
                st.error(f"Model error: {str(e)}")

        with col4:
            st.markdown("#### 💯 Health Score")
            score = int(latest["Health_Score"])
            risk = latest["Risk_Level"].lower()
            color = "#4caf50" if "low" in risk else "#ffa94d" if "medium" in risk else "#ff4d4d"
            st.plotly_chart(donut_chart(score, "Score", color), use_container_width=True)

        st.markdown("## ✅ Preventive Measures")
        st.markdown('<div style="background:#fff;padding:15px;border-radius:10px;">', unsafe_allow_html=True)
        bmi, hr, sys = latest["BMI"], latest["Heart_Rate"], latest["Systolic_BP"]
        if bmi < 18.5 or bmi > 25:
            st.write(f"• Adjust BMI (Current: {bmi}) – Balanced diet & exercise.")
        if hr > 90:
            st.write(f"• Reduce Heart Rate ({hr} bpm) – Stress management & exercise.")
        if sys > 130:
            st.write(f"• Control Blood Pressure ({sys} mmHg) – Cut sodium, stay active.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            st.write("• Quit smoking – Enroll in cessation programs.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("## 📅 Book Appointment")
        appointment_calendar()

    with tab2:
        st.subheader("📊 Health Score Over Time")
        st.line_chart(patient_df.set_index(pd.to_datetime(patient_df["date"]))["Health_Score"])
        for _, row in patient_df.iterrows():
            with st.expander(f"Visit on {row['date']}"):
                st.write(f"**BMI:** {row['BMI']}")
                st.write(f"**BP:** {row['Systolic_BP']}/{row['Diastolic_BP']}")
                st.write(f"**Heart Rate:** {row['Heart_Rate']}")
                st.write(f"**Score:** {row['Health_Score']}")
                st.write(f"**Risk:** {row['Risk_Level']}")

    if st.button("🔙 Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.patient_id = ""

if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

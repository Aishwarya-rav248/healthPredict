# REFERENCE: https://chat.openai.com/share/ae4ff8d8-f810-4a43-b700-6a725f99f9eb
# Streamlit Healthcare Dashboard App (Modern UI + All cards + Risk + Score)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

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

def donut_chart(value, label, color):
    fig = go.Figure(data=[go.Pie(
        values=[value, 100 - value],
        hole=0.65,
        marker_colors=[color, "#f0f2f6"],
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(
            text=f"<b>{value}</b><br>{label}",
            font_size=18,
            showarrow=False
        )]
    )
    return fig

def show_dashboard(patient_id):
    st.markdown("""
    <style>
    .card {
        background-color: #f9fafa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #053B50;
    }
    </style>
    """, unsafe_allow_html=True)

    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")
    if patient_df.empty:
        st.error("No data available for this patient.")
        return

    latest = patient_df.iloc[-1]
    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

   with tab1:
    st.markdown("<h3 style='margin-bottom: 10px;'>👤 Patient Overview</h3>", unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                <div style='background-color: #f7f9fc; padding: 20px; border-radius: 10px; font-size: 14px;'>
                    <b>Patient ID:</b> {}<br>
                    <b>Visit Date:</b> {}<br>
                    <b>Height:</b> {} cm<br>
                    <b>Weight:</b> {} kg<br>
                    <b>Smoking Status:</b> {}
                </div>
            """.format(patient_id, latest["date"], latest["Height_cm"], latest["Weight_kg"], latest["Smoking_Status"]), unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div style='background-color: #f7f9fc; padding: 20px; border-radius: 10px; font-size: 14px;'>
                    <b>BMI:</b> {}<br>
                    <b>Blood Pressure:</b> {}/{}<br>
                    <b>Heart Rate:</b> {} BPM<br>
                    <b>Risk Level:</b> {}
                </div>
            """.format(latest["BMI"], latest["Systolic_BP"], latest["Diastolic_BP"], latest["Heart_Rate"], latest["Risk_Level"]), unsafe_allow_html=True)

    st.markdown("### 🛡️ Risk Assessment")
    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown("##### 🧠 Heart Disease Risk Predictor")

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

                risk_color = "#4caf50" if prediction == 0 else "#ff4d4d"
                risk_label = "Low Risk" if prediction == 0 else "High Risk"
                risk_advice = [
                    "Maintain a healthy diet",
                    "Exercise regularly",
                    "Monitor blood pressure"
                ] if prediction == 0 else [
                    "Schedule cardiac checkup",
                    "Monitor BP & cholesterol",
                    "Adopt heart-healthy habits"
                ]

                st.markdown(f"<div style='background-color: {risk_color}20; padding: 10px; border-radius: 8px;'>✅ <b>{risk_label}</b> ({confidence:.1f}%)</div>", unsafe_allow_html=True)

                st.markdown("**Preventive Measures:**")
                for tip in risk_advice:
                    st.markdown(f"- {tip}")

        except Exception as e:
            st.error(f"Model error: {str(e)}")

    with col4:
        st.markdown("##### 💯 Health Score")
        score = latest["Health_Score"]
        risk_level = latest["Risk_Level"].lower()
        color = "#4caf50" if "low" in risk_level else "#ffa94d" if "medium" in risk_level else "#ff4d4d"

        fig = go.Figure(data=[go.Pie(
            values=[score, 100 - score],
            hole=0.7,
            marker_colors=[color, "#f0f2f6"],
            textinfo='none'
        )])
        fig.update_layout(
            showlegend=False,
            height=250,
            annotations=[dict(
                text=f"<b>{score}</b><br>Score",
                font_size=16,
                showarrow=False
            )],
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Preventive Measures")
    with st.container():
        recs = []
        if latest["BMI"] < 18.5 or latest["BMI"] > 25:
            recs.append(f"• Adjust BMI (Current: {latest['BMI']}) – Balanced diet & exercise.")
        if latest["Heart_Rate"] > 90:
            recs.append(f"• High Heart Rate ({latest['Heart_Rate']} bpm) – Stress reduction & exercise.")
        if latest["Systolic_BP"] > 130:
            recs.append(f"• Blood Pressure ({latest['Systolic_BP']} mmHg) – Reduce salt & stay active.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            recs.append("• Smoking Cessation – Enroll in quit programs and seek support.")

        if recs:
            for r in recs:
                st.markdown(f"<p style='font-size: 13px;'>{r}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 13px;'>No immediate preventive actions needed.</p>", unsafe_allow_html=True)

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

    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# MAIN
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

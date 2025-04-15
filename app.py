
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

def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("date")

    if patient_df.empty:
        st.error("No data available for this patient.")
        return

    latest = patient_df.iloc[-1]
    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    with tab1:
        st.markdown("<h2 style='margin-bottom:0;'>📘 Patient Overview</h2>", unsafe_allow_html=True)
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Personal Info")
            st.markdown(f"<div style='padding:10px;background-color:#f9f9f9;border-radius:10px;'>"
                        f"<b>Patient ID:</b> {patient_id}<br>"
                        f"<b>Visit Date:</b> {latest['date']}<br>"
                        f"<b>Height:</b> {latest['Height_cm']} cm<br>"
                        f"<b>Weight:</b> {latest['Weight_kg']} kg<br>"
                        f"<b>Smoking Status:</b> {latest['Smoking_Status']}"
                        f"</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Health Metrics")
            st.markdown(f"<div style='padding:10px;background-color:#f9f9f9;border-radius:10px;'>"
                        f"<b>BMI:</b> {latest['BMI']}<br>"
                        f"<b>Blood Pressure:</b> {latest['Systolic_BP']}/{latest['Diastolic_BP']}<br>"
                        f"<b>Heart Rate:</b> {latest['Heart_Rate']}<br>"
                        f"<b>Risk Level:</b> {latest['Risk_Level']}"
                        f"</div>", unsafe_allow_html=True)

        st.markdown("---")
        col3, col4 = st.columns([2, 1])

        with col3:
            st.markdown("#### 🧠 Heart Disease Risk Predictor")
            try:
                if not os.path.exists("heart_disease_model.pkl"):
                    st.error("Model file not found. Please upload 'heart_disease_model.pkl'.")
                else:
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

                    if prediction == 1:
                        st.error(f"⚠️ High Risk ({confidence:.1f}%)")
                    else:
                        st.success(f"✅ Low Risk ({confidence:.1f}%)")
            except Exception as e:
                st.error(f"Model error: {str(e)}")

        with col4:
            st.markdown("#### 💯 Health Score")
            health_score = latest["Health_Score"]
            risk_level = latest["Risk_Level"].lower()
            color = "#4caf50" if "low" in risk_level else "#ffa94d" if "medium" in risk_level else "#ff4d4d"
            fig = go.Figure(data=[go.Pie(
                values=[health_score, 100 - health_score],
                hole=0.65,
                marker_colors=[color, "#f0f2f6"],
                textinfo='none'
            )])
            fig.update_layout(
                showlegend=False,
                height=260,
                annotations=[dict(
                    text=f"<b>{health_score}</b><br>Score",
                    font_size=18,
                    showarrow=False
                )],
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🛡️ Preventive Measures")
        bmi = latest["BMI"]
        hr = latest["Heart_Rate"]
        sys = latest["Systolic_BP"]

        if bmi < 18.5 or bmi > 25:
            st.write(f"• Adjust BMI (Current: {bmi}) – Balanced diet & exercise recommended.")
        if hr > 90:
            st.write(f"• Reduce Heart Rate ({hr} bpm) – Consider stress management & exercise.")
        if sys > 130:
            st.write(f"• Manage Blood Pressure ({sys} mmHg) – Reduce salt, exercise regularly.")
        if latest["Smoking_Status"].lower().startswith("current"):
            st.write("• Smoking Cessation – Enroll in quit smoking programs.")

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

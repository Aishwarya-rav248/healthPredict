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
    st.markdown("## 💙 Patient Health Dashboard")
    st.markdown(f"**Patient ID:** {patient_id} | **Date:** {latest['date']}")

    st.markdown("### 🔍 Health Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("BMI", f"{latest['BMI']} kg/m²")
        st.markdown(f"**Height:** {latest['Height_cm']} cm  \n**Weight:** {latest['Weight_kg']} kg")

    with col2:
        st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']} mmHg", help="Systolic/Diastolic")

    with col3:
        st.metric("Heart Rate", f"{latest['Heart_Rate']} BPM")
        st.markdown(f"**Smoking Status:** {latest['Smoking_Status']}")

    st.markdown("### 📊 Risk Assessment")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Health Score**")
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
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"<b>{health_score}</b><br>Score",
                font_size=18,
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.markdown("**Heart Disease Risk**")
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
                    st.markdown("- Schedule cardiac checkup\n- Monitor blood pressure & cholesterol\n- Improve lifestyle")
                else:
                    st.success(f"✅ Low Risk ({confidence:.1f}%)")
                    st.markdown("- Maintain current lifestyle\n- Regular health checkups")
        except Exception as e:
            st.error(f"Model error: {str(e)}")

    st.markdown("### ✅ Preventive Measures")
    if latest["BMI"] < 18.5 or latest["BMI"] > 25:
        st.write(f"• Adjust BMI ({latest['BMI']}) – Balanced diet & exercise.")
    if latest["Heart_Rate"] > 90:
        st.write(f"• High Heart Rate ({latest['Heart_Rate']}) – Try stress management & cardio.")
    if latest["Systolic_BP"] > 130:
        st.write(f"• High BP ({latest['Systolic_BP']}) – Limit salt & exercise more.")
    if str(latest["Smoking_Status"]).lower().startswith("current"):
        st.write("• Quit Smoking – Seek support programs.")

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

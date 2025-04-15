import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="HealthPredict Dashboard", layout="wide")

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

    # ---------------- OVERVIEW PAGE ----------------
    with tab1:
        st.markdown("## 👤 Patient Overview")

        top1, top2 = st.columns([2, 1])

        with top1:
            st.markdown("### Personal Info")
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Visit Date:** {latest['date']}")
            st.write(f"**Height:** {latest['Height_cm']} cm")
            st.write(f"**Weight:** {latest['Weight_kg']} kg")
            st.write(f"**Smoking Status:** {latest['Smoking_Status']}")

        with top2:
            st.markdown("### Health Metrics")
            st.metric("BMI", latest["BMI"])
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
            st.metric("Heart Rate", latest["Heart_Rate"])
            st.metric("Risk Level", latest["Risk_Level"])

        st.divider()

        mid1, mid2 = st.columns([1, 1])

        with mid1:
            st.markdown("### 🧠 Heart Disease Risk Predictor")
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

        with mid2:
            st.markdown("### 💯 Health Score")
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
                annotations=[dict(
                    text=f"<b>{health_score}</b><br>Score",
                    font_size=18,
                    showarrow=False
                )],
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### 🛡️ Preventive Measures")
        if latest["BMI"] < 18.5 or latest["BMI"] > 25:
            st.write(f"• Adjust BMI (Current: {latest['BMI']}) – Balanced diet & exercise recommended.")
        if latest["Heart_Rate"] > 90:
            st.write(f"• High Heart Rate ({latest['Heart_Rate']} bpm) – Practice stress relief and exercise.")
        if latest["Systolic_BP"] > 130:
            st.write(f"• Elevated Blood Pressure ({latest['Systolic_BP']} mmHg) – Reduce salt, exercise regularly.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            st.write("• Smoking Cessation – Enroll in quit programs and seek support.")

    # ---------------- VISIT HISTORY TAB ----------------
    with tab2:
        st.markdown("## 🕒 Patient Visit History")
        trend_df = patient_df[["date", "Health_Score"]].dropna()
        trend_df["date"] = pd.to_datetime(trend_df["date"])

        if not trend_df.empty:
            st.line_chart(trend_df.set_index("date"))
        else:
            st.warning("No Health Score data available over time.")

        for _, row in patient_df.iterrows():
            with st.expander(f"📅 Visit on {row['date']}"):
                st.write(f"**Height:** {row['Height_cm']} cm")
                st.write(f"**Weight:** {row['Weight_kg']} kg")
                st.write(f"**BMI:** {row['BMI']}")
                st.write(f"**BP:** {row['Systolic_BP']}/{row['Diastolic_BP']}")
                st.write(f"**Heart Rate:** {row['Heart_Rate']}")
                st.write(f"**Smoking:** {row['Smoking_Status']}")
                st.write(f"**Health Score:** {row['Health_Score']}")
                st.write(f"**Risk Level:** {row['Risk_Level']}")

                st.markdown("**Preventive Tips:**")
                if row["BMI"] < 18.5 or row["BMI"] > 25:
                    st.write(f"• BMI {row['BMI']} – Optimize weight via lifestyle changes.")
                if row["Heart_Rate"] > 90:
                    st.write(f"• Heart Rate {row['Heart_Rate']} – Encourage regular monitoring.")
                if row["Systolic_BP"] > 130:
                    st.write(f"• Blood Pressure {row['Systolic_BP']} mmHg – Reduce sodium intake.")
                if str(row["Smoking_Status"]).lower().startswith("current"):
                    st.write("• Smoking – Strongly recommend quitting support.")

    st.markdown("---")
    if st.button("🔙 Back to Login"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# --------------- MAIN ------------------
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

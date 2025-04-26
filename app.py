import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date
import streamlit.components.v1 as components

# ------------------- Setup -------------------
st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

def donut_chart(label, value, color, show_score=True):
    text = f"<b>{value:.0f}</b><br>{label}" if show_score else f"<b>{label}</b>"
    fig = go.Figure(data=[go.Pie(values=[value if show_score else 50, 100 - value if show_score else 50],
                                 hole=0.75, marker_colors=[color, "#f0f2f6"], textinfo='none')])
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=170, width=170,
                      annotations=[dict(text=text, font_size=14, showarrow=False)])
    return fig

def save_appointment(patient_id, doctor, appt_date, notes):
    record = pd.DataFrame([{
        "Patient_ID": patient_id,
        "Doctor": doctor,
        "Date": appt_date,
        "Notes": notes
    }])
    if os.path.exists("appointments.csv"):
        record.to_csv("appointments.csv", mode='a', header=False, index=False)
    else:
        record.to_csv("appointments.csv", index=False)

# ------------------- Login -------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.patient_id = ""

def show_login():
    st.title("Welcome to HealthPredict")
    patient_id = st.text_input("Enter Patient ID")
    if st.button("Login"):
        if patient_id in df["patient"].astype(str).values:
            st.session_state.logged_in = True
            st.session_state.patient_id = patient_id
            st.rerun()
        else:
            st.error("Invalid Patient ID. Please try again.")

# ------------------- Dashboard -------------------
def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("Date")
    latest = patient_df.iloc[-1]

    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    # ------------------- OVERVIEW -------------------
    with tab1:
        with st.sidebar:
            st.markdown("## 📅 Book Appointment")
            doctor = st.selectbox("Choose Doctor", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
            appt_date = st.date_input("Select Date", min_value=date.today())
            notes = st.text_input("Notes (optional)")
            if st.button("Book Appointment"):
                save_appointment(patient_id, doctor, appt_date, notes)
                st.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

        st.markdown("## 👤 Patient Overview")
        top1, top2, top3 = st.columns([1.2, 1.2, 1.2])

        with top1:
            st.markdown(f"**Patient ID:** {patient_id}")
            st.markdown(f"**Visit Date:** {latest['Date'].date()}")
            st.markdown(f"**Height:** {latest['Height_cm']} cm")
            st.markdown(f"**Weight:** {latest['Weight_kg']} kg")
        
        with top2:
            st.metric("BMI", latest["BMI"])
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")

        with top3:
            st.markdown(f"**Age:** {latest['AGE']}")
            st.markdown(f"**Gender:** {latest['GENDER']}")
            st.markdown(f"**Smoking:** {latest['Smoking_Status']}")
            st.markdown(f"**Diabetes:** {'Yes' if latest['Diabetes'] else 'No'}")
            st.markdown(f"**Hyperlipidemia:** {'Yes' if latest['Hyperlipidemia'] else 'No'}")
            st.markdown(f"**Heart Disease:** {'Yes' if latest['Heart_Disease'] else 'No'}")

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("### Health Score")
            health_score = latest["Health Score"]
            color = "#4caf50" if health_score >= 80 else "#ffa94d" if health_score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", health_score, color), use_container_width=True)

        with c4:
            st.markdown("### Heart Risk Prediction")
            try:
                model = joblib.load("Heart_Disease_Risk_Model_XGBoost.pkl")

                input_df = pd.DataFrame([{
                    "Height_cm": latest["Height_cm"],
                    "Weight_kg": latest["Weight_kg"],
                    "BMI": latest["BMI"],
                    "Systolic_BP": latest["Systolic_BP"],
                    "Diastolic_BP": latest["Diastolic_BP"],
                    "Heart_Rate": latest["Heart_Rate"],
                    "Smoking_Status": latest["Smoking_Status"],
                    "Diabetes": latest["Diabetes"],
                    "Hyperlipidemia": latest["Hyperlipidemia"],
                    "AGE": latest["AGE"],
                    "GENDER": latest["GENDER"]
                }])

                prediction_proba = model.predict_proba(input_df)[0][1] * 100
                prediction = model.predict(input_df)[0]

                label = "High Risk" if prediction == 1 else "Low Risk"
                risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart(label, prediction_proba, risk_color), use_container_width=True)

                # SHAP Visual
                st.markdown("### 🔎 Factors Influencing Risk Prediction")
                try:
                    with open("SHAP.html", "r", encoding="utf-8") as f:
                        shap_html = f.read()
                    components.html(shap_html, height=600, scrolling=True)
                except Exception:
                    st.warning("⚠️ SHAP visualization could not be loaded.")

                # Insight & Recommendation
                st.markdown("### Insight & Recommendation")
                if health_score >= 80 and prediction == 0:
                    st.success("✅ Health score and risk are aligned. Keep maintaining your good health!")
                elif health_score < 60 and prediction == 1:
                    st.error("🚨 Low health score and high risk detected. Immediate consultation recommended.")
                elif health_score >= 80 and prediction == 1:
                    st.warning("⚠️ Good health score but elevated risk detected. Recommend full checkup.")
                elif health_score < 60 and prediction == 0:
                    st.info("🟡 Low health score but low risk detected. Focus on healthy lifestyle improvements.")
            except Exception as e:
                st.error(f"Model Error: {e}")

        st.markdown("### 🛡️ Preventive Measures")
        if latest["BMI"] < 18.5 or latest["BMI"] > 25:
            st.write(f"• Your BMI is {latest['BMI']} – Adopt a balanced diet and regular exercise.")
        if latest["Heart_Rate"] > 90:
            st.write("• High heart rate – Manage stress, improve cardio fitness.")
        if latest["Systolic_BP"] > 130 or latest["Diastolic_BP"] > 85:
            st.write("• Elevated blood pressure – Reduce salt, regular checkups needed.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            st.write("• Smoking – Join cessation programs.")
        if latest["Hyperlipidemia"]:
            st.write("• Hyperlipidemia – Adopt heart-healthy diet and consider medications.")
        if latest["Diabetes"]:
            st.write("• Diabetes – Follow doctor's advice and monitor glucose regularly.")

    # ------------------- VISIT HISTORY -------------------
    with tab2:
        st.markdown("## 📅 Visit History")
        st.info(f"Total Visits: {len(patient_df)} | Avg. Health Score: {round(patient_df['Health Score'].mean(), 1)}")
        selected_metric = st.selectbox("Metric to View Trend", ["Health Score", "BMI", "Systolic_BP", "Heart_Rate"])
        st.line_chart(patient_df.set_index("Date")[selected_metric])

        for _, row in patient_df.iterrows():
            risk = "High" if row["Heart_Disease"] == 1 else "Low"
            color = "#ff4d4d" if row["Heart_Disease"] == 1 else "#4caf50"
            tips = []
            if row["BMI"] < 18.5 or row["BMI"] > 25:
                tips.append("• Maintain healthy BMI through diet and activity.")
            if row["Heart_Rate"] > 90:
                tips.append("• High heart rate – Work on aerobic fitness.")
            if row["Systolic_BP"] > 130 or row["Diastolic_BP"] > 85:
                tips.append("• Manage blood pressure through lifestyle changes.")
            if str(row["Smoking_Status"]).lower().startswith("current"):
                tips.append("• Quit smoking for heart health.")
            if row["Hyperlipidemia"]:
                tips.append("• Control cholesterol with diet and exercise.")
            if row["Diabetes"]:
                tips.append("• Manage diabetes with regular medical supervision.")

            tip_text = "<br>".join(tips)
            st.markdown(
                f"""<div style='border:1px solid #ccc;border-radius:10px;padding:10px;margin:10px 0;background:#f9f9f9;'>
                <b>🗓 Visit Date:</b> {row['Date'].date()}<br>
                <b>Height:</b> {row['Height_cm']} cm | <b>Weight:</b> {row['Weight_kg']} kg | <b>BMI:</b> {row['BMI']}<br>
                <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']} | <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>
                <b>Health Score:</b> {row['Health Score']} | <b>Heart Risk:</b> <span style='background:{color};color:white;padding:2px 5px;border-radius:4px;'>{risk}</span><br>
                <b>🛡️ Tips:</b><br>{tip_text}
                </div>
                """, unsafe_allow_html=True
            )

    if st.button("🔙 Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# ------------------- Run -------------------
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

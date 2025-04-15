import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Risk_Level"] = df["Health_Score"].apply(lambda score: "Low" if score >= 85 else "Medium" if score >= 70 else "High")
    return df

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
        if patient_id in df["Patient_ID"].astype(str).values:
            st.session_state.logged_in = True
            st.session_state.patient_id = patient_id
            st.rerun()
        else:
            st.error("Invalid Patient ID. Please try again.")

def show_dashboard(patient_id):
    patient_df = df[df["Patient_ID"].astype(str) == patient_id].sort_values("Date")
    latest = patient_df.iloc[-1]
    level = latest["Risk_Level"]

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
            st.markdown(f"- **Date**: {latest['Date'].date()}")
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
            color = "#4caf50" if level == "Low" else "#ffa94d" if level == "Medium" else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, color), use_container_width=True)

        with c4:
            st.markdown("### 🧠 Heart Risk")
            try:
                model = joblib.load("heart_disease_model.pkl")
                input_df = pd.DataFrame([{
                    "Height_cm": latest["Height_cm"],
                    "BMI": latest["BMI"],
                    "Weight_kg": latest["Weight_kg"],
                    "Diastolic_BP": latest["Diastolic_BP"],
                    "Heart_Rate": latest["Heart_Rate"],
                    "Systolic_BP": latest["Systolic_BP"],
                    "Diabetes": latest["Diabetes"],
                    "Hyperlipidemia": latest["Hyperlipidemia"],
                    "Smoking_Status": str(latest["Smoking_Status"])
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

        st.info(f"**Total Visits:** {len(patient_df)} | **Avg Score:** {patient_df['Health_Score'].mean():.1f} | **High Risk Visits:** {(patient_df['Risk_Level'] == 'High').mean()*100:.0f}%")

        chart_option = st.selectbox("Choose Metric to Visualize", ["Health_Score", "BMI", "Systolic_BP", "Heart_Rate"])
        st.line_chart(patient_df.set_index("Date")[chart_option])

        for _, row in patient_df.iterrows():
            risk_color = "#ff4d4d" if row["Risk_Level"] == "High" else "#ffa94d" if row["Risk_Level"] == "Medium" else "#4caf50"
            st.markdown(
                f"""
                <div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; background: #fafafa'>
                    <h5>🗓️ Visit on {row['Date'].date()}</h5>
                    <div style='display: flex; justify-content: space-between; font-size: 14px;'>
                        <div>
                            <b>Height:</b> {row['Height_cm']} cm<br>
                            <b>Weight:</b> {row['Weight_kg']} kg<br>
                            <b>BMI:</b> {row['BMI']}<br>
                            <b>Smoking:</b> {row['Smoking_Status']}
                        </div>
                        <div>
                            <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']}<br>
                            <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>
                            <b>Health Score:</b> {row['Health_Score']}<br>
                            <b>Risk:</b> <span style='background-color:{risk_color}; color:white; padding:2px 6px; border-radius:5px;'>{row['Risk_Level']}</span>
                        </div>
                    </div>
                    <div style='margin-top: 10px; font-size: 13px;'>
                        <b>🛡️ Tips:</b><br>
                        {"• BMI outside healthy range – adjust diet & activity.<br>" if row['BMI'] < 18.5 or row['BMI'] > 25 else ""}
                        {"• High Heart Rate – reduce stress, exercise more.<br>" if row['Heart_Rate'] > 90 else ""}
                        {"• High BP – limit sodium, regular monitoring.<br>" if row['Systolic_BP'] > 130 else ""}
                        {"• Smoking – consider cessation support.<br>" if str(row['Smoking_Status']).lower().startswith("current") else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

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

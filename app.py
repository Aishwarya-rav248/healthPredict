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
        # Sidebar calendar
        with st.sidebar:
            st.markdown("## 📅 Book Appointment")
            doctor = st.selectbox("Doctor Type", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
            appt_date = st.date_input("Date", min_value=date.today())
            st.text_input("Notes", key="notes")
            if st.button("Book"):
                st.success(f"✅ Appointment with {doctor} on {appt_date.strftime('%b %d, %Y')}")

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

        # Donut charts
        c3, c4 = st.columns(2)
        with c3:
            score = latest["Health_Score"]
            level = latest["Risk_Level"].lower()
            color = "#4caf50" if "low" in level else "#ffa94d" if "medium" in level else "#ff4d4d"
            st.markdown("### 🧬 Health Score")
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
                label = "High Risk" if prediction == 1 else "Low Risk"
                color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart(label, 50, color, show_score=False), use_container_width=True)
            except Exception as e:
                st.error(f"Model error: {e}")

        # Preventive Measures
        st.markdown("### 🛡️ Preventive Measures")
        bmi, hr, sys = latest["BMI"], latest["Heart_Rate"], latest["Systolic_BP"]
        if bmi < 18.5 or bmi > 25:
            st.write(f"• Adjust BMI ({bmi}) – Diet + Exercise.")
        if hr > 90:
            st.write(f"• High Heart Rate ({hr} bpm) – Relaxation + Activity.")
        if sys > 130:
            st.write(f"• High BP ({sys} mmHg) – Monitor & reduce salt.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            st.write("• Smoking – Consider cessation programs.")

    with tab2:
        st.markdown("## 📅 Visit History")

        total = len(patient_df)
        avg = round(patient_df["Health_Score"].mean(), 2)
        pct = (patient_df["Risk_Level"].str.lower() == "high").mean() * 100
        st.info(f"**Visits:** {total} | **Avg Score:** {avg} | **High Risk:** {pct:.0f}%")

        risk_filter = st.selectbox("Filter Risk", ["All", "Low", "Medium", "High"])
        if risk_filter != "All":
            patient_df = patient_df[patient_df["Risk_Level"].str.lower() == risk_filter.lower()]

        chart_type = st.radio("Chart Type", ["Line", "Bar", "Area"], horizontal=True)
        chart_data = patient_df.set_index(pd.to_datetime(patient_df["date"]))["Health_Score"]
        st.markdown("### 📊 Trend")
        st.line_chart(chart_data) if chart_type == "Line" else (
            st.bar_chart(chart_data) if chart_type == "Bar" else st.area_chart(chart_data)
        )

        for _, row in patient_df.iterrows():
            color = "#ff4d4d" if row["Risk_Level"].lower() == "high" else "#4caf50" if row["Risk_Level"].lower() == "low" else "#ffa94d"
            with st.container():
                st.markdown(
                    f"""
                    <div style='border:1px solid #ccc; border-radius:12px; padding:15px; margin-bottom:15px; background-color:#f8f9fa;'>
                        <h5>🗓️ Visit on {row['date']}</h5>
                        <div style='display:flex; justify-content:space-between; font-size:14px;'>
                            <div>
                                <b>Height:</b> {row['Height_cm']} cm<br>
                                <b>Weight:</b> {row['Weight_kg']} kg<br>
                                <b>BMI:</b> {row['BMI']}<br>
                                <b>Smoking:</b> {row['Smoking_Status']}
                            </div>
                            <div>
                                <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']}<br>
                                <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>
                                <b>Score:</b> {row['Health_Score']}<br>
                                <b>Risk:</b> <span style='background:{color}; color:white; padding:2px 6px; border-radius:4px;'>{row['Risk_Level']}</span>
                            </div>
                        </div>
                        <div style='margin-top:10px; font-size:13px;'>
                            <b>🛡️ Tips:</b><br>
                            {"• Unhealthy BMI – improve diet & activity.<br>" if row['BMI'] < 18.5 or row['BMI'] > 25 else ""}
                            {"• High HR – manage stress.<br>" if row['Heart_Rate'] > 90 else ""}
                            {"• High BP – reduce sodium.<br>" if row['Systolic_BP'] > 130 else ""}
                            {"• Smoking – seek support.<br>" if str(row['Smoking_Status']).lower().startswith("current") else ""}
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

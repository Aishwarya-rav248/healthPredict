import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

# ------------------- Helper -------------------
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

    # ------------------- Overview Tab -------------------
    with tab1:
        with st.sidebar:
            st.markdown("## 📅 Book Appointment")
            doctor = st.selectbox("Choose Doctor", ["Cardiologist", "General Physician", "Endocrinologist"])
            appt_date = st.date_input("Select Date", min_value=date.today())
            notes = st.text_input("Notes (optional)")
            if st.button("Book Appointment"):
                save_appointment(patient_id, doctor, appt_date, notes)
                st.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

        st.markdown("## 👤 Patient Overview")

        # Top Cards
        top1, top2, top3 = st.columns([1.2, 1.2, 1.2])
        with top1:
            st.markdown(f"**Patient ID**: {patient_id}")
            st.markdown(f"**Visit Date**: {latest['Date'].date()}")
            st.markdown(f"**Height**: {latest['Height_cm']} cm")
            st.markdown(f"**Weight**: {latest['Weight_kg']} kg")
        with top2:
            st.metric("BMI", latest["BMI"])
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")
            st.metric("BP", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
        with top3:
            st.markdown(f"**Smoking:** {latest['Smoking_Status']}")
            st.markdown(f"**Diabetes:** {'Yes' if latest['Diabetes'] else 'No'}")
            st.markdown(f"**Hyperlipidemia:** {'Yes' if latest['Hyperlipidemia'] else 'No'}")
            st.markdown(f"**Heart Disease:** {'Yes' if latest['Heart_Disease'] else 'No'}")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### Health Score")
            score = latest["Health_Score"]
            color = "#4caf50" if score >= 80 else "#ffa94d" if score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, color), use_container_width=True)

        with c4:
    st.markdown("### Heart Risk")
    try:
        model = joblib.load("heart_disease_model (1).pkl")
        input_df = pd.DataFrame([{
            "BMI": latest["BMI"],
            "Systolic_BP": latest["Systolic_BP"],
            "Diastolic_BP": latest["Diastolic_BP"],
            "Heart_Rate": latest["Heart_Rate"],
            "Smoking_Status": latest["Smoking_Status"],
            "Diabetes": latest["Diabetes"],
            "Hyperlipidemia": latest["Hyperlipidemia"]
        }])
        le = LabelEncoder()
        le.fit(df["Smoking_Status"].unique())
        input_df["Smoking_Status"] = le.transform(input_df["Smoking_Status"])
        prediction = model.predict(input_df)[0]
        label = "High Risk" if prediction == 1 else "Low Risk"
        risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
        st.plotly_chart(donut_chart(label, 50, risk_color, show_score=False), use_container_width=True)

        # Consistency Check (Updated)
        st.markdown("### 🔍 Consistency Check")
        if prediction == 1 and score >= 85:
            st.warning("⚠️ Your Health Score looks good, but you're at **High Risk**. Consider a detailed checkup soon.")
        elif prediction == 0 and score <= 60:
            st.info("ℹ️ Your Heart Risk is low, but your overall Health Score is poor. Consider improving your lifestyle.")
        elif prediction == 1 and score <= 60:
            st.error("🚨 Both Health Score and Risk are in bad range. Please consult a doctor immediately.")
        elif prediction == 0 and score >= 85:
            st.success("✅ Great! Your Health Score and Risk levels are in sync. Keep maintaining your health!")
    except Exception as e:
        st.error(f"Model error: {e}")


        st.markdown("### Personalized Preventive Measures")
        recommendations = []

        if latest["BMI"] < 18.5:
            recommendations.append(f"• Underweight (BMI: {latest['BMI']:.1f}) – Increase nutritious calorie intake.")
        elif latest["BMI"] > 25:
            recommendations.append(f"• Overweight (BMI: {latest['BMI']:.1f}) – Adopt balanced diet & physical activity.")

        if latest["Heart_Rate"] > 100:
            recommendations.append(f"• High Heart Rate ({latest['Heart_Rate']} bpm) – Practice stress reduction techniques.")
        elif latest["Heart_Rate"] < 60:
            recommendations.append(f"• Low Heart Rate ({latest['Heart_Rate']} bpm) – Monitor if symptoms occur.")

        if latest["Systolic_BP"] >= 130 or latest["Diastolic_BP"] >= 80:
            recommendations.append(f"• Elevated BP ({latest['Systolic_BP']}/{latest['Diastolic_BP']}) – Reduce salt, regular checkups.")

        if str(latest["Smoking_Status"]).lower().startswith("current"):
            recommendations.append("• Smoking – Quit smoking for significant heart and lung health improvements.")

        if latest["Diabetes"]:
            recommendations.append("• Diabetes – Manage sugar levels with diet, medication, and regular monitoring.")

        if latest["Hyperlipidemia"]:
            recommendations.append("• Hyperlipidemia – Avoid fatty foods, follow up on lipid profiles.")

        if not recommendations:
            st.success("🎉 Great job! No major red flags. Keep up the healthy lifestyle.")
        else:
            for rec in recommendations:
                st.write(rec)

    # ------------------- Visit History -------------------
    with tab2:
        st.markdown("## 📅 Visit History")
        st.info(f"Total Visits: {len(patient_df)} | Avg. Score: {round(patient_df['Health_Score'].mean(), 1)}")
        selected_metric = st.selectbox("Metric to View Trend", ["Health_Score", "BMI", "Systolic_BP", "Heart_Rate"])
        st.line_chart(patient_df.set_index("Date")[selected_metric])

        for _, row in patient_df.iterrows():
            risk = "High" if row["Heart_Disease"] == 1 else "Low"
            color = "#ff4d4d" if row["Heart_Disease"] == 1 else "#4caf50"
            tips = []
            if row["BMI"] < 18.5 or row["BMI"] > 25:
                tips.append("• Adjust BMI")
            if row["Heart_Rate"] > 90:
                tips.append("• High Heart Rate")
            if row["Systolic_BP"] > 130:
                tips.append("• High Blood Pressure")
            if str(row["Smoking_Status"]).lower().startswith("current"):
                tips.append("• Smoking Cessation")

            tip_text = "<br>".join(tips)
            st.markdown(
                f"""<div style='border:1px solid #ccc;border-radius:10px;padding:10px;margin:10px 0;background:#f9f9f9;'>
                <b>🗓 Visit Date:</b> {row['Date'].date()}<br>
                <b>Height:</b> {row['Height_cm']} cm | <b>Weight:</b> {row['Weight_kg']} kg | <b>BMI:</b> {row['BMI']}<br>
                <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']} | <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>
                <b>Health Score:</b> {row['Health_Score']} | <b>Heart Risk:</b> <span style='background:{color};color:white;padding:2px 5px;border-radius:4px;'>{risk}</span><br>
                <b>🛡️ Tips:</b><br>{tip_text}
                </div>
                """, unsafe_allow_html=True
            )

    if st.button("🔙 Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# ------------------- Run App -------------------
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

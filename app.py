import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date
import shap
import numpy as np

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
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=160, width=160,
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

def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("Date")
    latest = patient_df.iloc[-1]

    tab1, tab2 = st.tabs(["Overview", "Visit History"])

    with tab1:
        with st.sidebar:
            st.markdown("## Book Appointment")
            doctor = st.selectbox("Choose Doctor", ["Cardiologist", "General Physician", "Endocrinologist", "Dietician"])
            appt_date = st.date_input("Select Date", min_value=date.today())
            notes = st.text_input("Notes (optional)")
            if st.button("Book Appointment"):
                save_appointment(patient_id, doctor, appt_date, notes)
                st.success(f"Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

        st.markdown("""
            <style>
                .card {
                    background-color: #f1f1f1;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }
                .container {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("# Welcome to HealthPredict")
        st.markdown("## 👤 Patient Overview")

        st.markdown("<div class='container'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='card'>
            <b>Patient ID:</b> {patient_id}<br>
            <b>Age:</b> {latest['AGE']}<br>
            <b>Gender:</b> {latest['GENDER']}<br>
            <b>Height:</b> {latest['Height_cm']} cm<br>
            <b>Weight:</b> {latest['Weight_kg']} kg<br>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='card'>
            <b>BMI:</b> {latest['BMI']}<br>
            <b>Heart Rate:</b> {latest['Heart_Rate']} bpm<br>
            <b>BP:</b> {latest['Systolic_BP']}/{latest['Diastolic_BP']}<br>
            <b>Smoking:</b> {latest['Smoking_Status']}<br>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='card'>
            <b>Diabetes:</b> {'Yes' if latest['Diabetes'] else 'No'}<br>
            <b>Hyperlipidemia:</b> {'Yes' if latest['Hyperlipidemia'] else 'No'}<br>
            <b>Heart Disease:</b> {'Yes' if latest['Heart_Disease'] else 'No'}<br>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown("### Health Score")
            health_score = latest["Health Score"]
            color = "#4caf50" if health_score >= 80 else "#ffa94d" if health_score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Health Score", health_score, color), use_container_width=True)

        with c2:
            st.markdown("### Heart Risk Prediction")
            model = joblib.load("Heart_Disease_Risk_Model_XGBoost.pkl")
            input_df = pd.DataFrame([{k: latest[k] for k in [
                "Height_cm", "Weight_kg", "BMI", "Systolic_BP", "Diastolic_BP",
                "Heart_Rate", "Smoking_Status", "Diabetes", "Hyperlipidemia", "AGE", "GENDER"]}])
            prediction_proba = model.predict_proba(input_df)[0][1] * 100
            prediction = model.predict(input_df)[0]
            label = "High Risk" if prediction == 1 else "Low Risk"
            risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
            st.plotly_chart(donut_chart(label, prediction_proba, risk_color), use_container_width=True)

        with c3:
            st.markdown("### Top Risk Contributors")
            preprocessor = model[:-1]
            xgb_model = model.named_steps["classifier"]
            input_df = input_df[model.feature_names_in_]
            input_transformed = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(input_transformed)
            feature_names = preprocessor.get_feature_names_out(model.feature_names_in_)
            base_names = [name.split("__")[1].split("_")[0] if "__" in name else name for name in feature_names]
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                "feature": base_names,
                "importance": shap_abs_mean
            }).groupby("feature")["importance"].sum().sort_values(ascending=False)
            top_k = 4
            top_features = feature_importance.head(top_k)
            others_sum = feature_importance.iloc[top_k:].sum()
            labels = top_features.index.tolist() + (["Others"] if others_sum > 0 else [])
            values = top_features.values.tolist() + ([others_sum] if others_sum > 0 else [])
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Insights & Recommendations")
        if health_score >= 80 and prediction == 0:
            st.success("✅ Your health and risk levels are aligned. Great job maintaining your health!")
        elif health_score < 60 and prediction == 1:
            st.error("🚨 Low health score and high risk. Immediate medical attention is advised.")
        elif health_score >= 80 and prediction == 1:
            st.warning("⚠️ Good health score but elevated heart risk. A full checkup is recommended.")
        elif health_score < 60 and prediction == 0:
            st.info("🔍 Low health score but currently low risk. Focus on building healthy habits.")

        if latest['BMI'] > 25:
            st.write("• Your BMI is high – Work on a balanced diet and regular exercise.")
        if latest['Heart_Rate'] > 90:
            st.write("• High resting heart rate – Consider stress management and aerobic training.")
        if latest['Systolic_BP'] > 130 or latest['Diastolic_BP'] > 85:
            st.write("• Elevated blood pressure – Lower sodium intake, regular checkups suggested.")
        if str(latest['Smoking_Status']).lower().startswith("current"):
            st.write("• Smoking – Strongly advised to quit smoking for heart protection.")
        if latest['Diabetes']:
            st.write("• Diabetes detected – Monitor sugar levels and follow medical advice.")
        if latest['Hyperlipidemia']:
            st.write("• Hyperlipidemia – Adopt a heart-healthy diet and stay active.")

    with tab2:
        st.markdown("## Visit History")
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
                <b>Visit Date:</b> {row['Date'].date()}<br>
                <b>Height:</b> {row['Height_cm']} cm | <b>Weight:</b> {row['Weight_kg']} kg | <b>BMI:</b> {row['BMI']}<br>
                <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']} | <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>
                <b>Health Score:</b> {row['Health Score']} | <b>Heart Risk:</b> <span style='background:{color};color:white;padding:2px 5px;border-radius:4px;'>{risk}</span><br>
                <b>Tips:</b><br>{tip_text}
                </div>
                """, unsafe_allow_html=True
            )

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

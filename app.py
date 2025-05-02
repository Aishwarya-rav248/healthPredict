import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from datetime import date
import shap
import numpy as np

st.set_page_config(page_title="HealthPredict", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

def donut_chart(label, value, color):
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        labels=["", ""],
        hole=0.7,
        marker_colors=[color, "#e0e0e0"],
        textinfo="none"
    ))
    fig.update_layout(
        annotations=[dict(text=f"<b>{label}<br>{int(value)}%</b>", showarrow=False, font_size=14)],
        margin=dict(t=10, b=10, l=10, r=10),
        height=200, width=200
    )
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
        st.markdown("""
            <style>
            .card {
                background-color: #f2f2f2;
                padding: 1.2rem;
                border-radius: 10px;
                box-shadow: 1px 1px 6px #ddd;
                height: 100%;
            }
            .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
            .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
            .full { width: 100%; margin-top: 1rem; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"<h2 style='text-align:center;'>Welcome to HealthPredict</h2>", unsafe_allow_html=True)

        # ➤ Grid 1: Patient Info + Metrics
        st.markdown("<div class='grid3'>", unsafe_allow_html=True)

        st.markdown(f"""
            <div class='card'>
            <h4>Patient Details</h4>
            ID: {patient_id}<br>
            Age: {latest['AGE']}<br>
            Gender: {latest['GENDER']}<br>
            Height: {latest['Height_cm']} cm<br>
            Weight: {latest['Weight_kg']} kg
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='card'>
            <h4>Health Metrics</h4>
            BMI: {latest['BMI']}<br>
            BP: {latest['Systolic_BP']}/{latest['Diastolic_BP']}<br>
            Heart Rate: {latest['Heart_Rate']} bpm<br>
            Smoking: {latest['Smoking_Status']}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='card'>
            <h4>Conditions</h4>
            Diabetes: {'Yes' if latest['Diabetes'] else 'No'}<br>
            Hyperlipidemia: {'Yes' if latest['Hyperlipidemia'] else 'No'}<br>
            Heart Disease: {'Yes' if latest['Heart_Disease'] else 'No'}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ➤ Grid 2: Donut charts + SHAP
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Health Score")
            score = latest["Health Score"]
            color = "#4caf50" if score >= 80 else "#ffa94d" if score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, color), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Heart Risk")
            model = joblib.load("Heart_Disease_Risk_Model_XGBoost.pkl")
            input_df = pd.DataFrame([{k: latest[k] for k in [
                "Height_cm", "Weight_kg", "BMI", "Systolic_BP", "Diastolic_BP",
                "Heart_Rate", "Smoking_Status", "Diabetes", "Hyperlipidemia", "AGE", "GENDER"]}])
            risk = model.predict_proba(input_df)[0][1] * 100
            label = "High Risk" if model.predict(input_df)[0] == 1 else "Low Risk"
            risk_color = "#ff4d4d" if label == "High Risk" else "#4caf50"
            st.plotly_chart(donut_chart(label, risk, risk_color), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Top Risk Contributors")
            preprocessor = model[:-1]
            xgb_model = model.named_steps["classifier"]
            input_df = input_df[model.feature_names_in_]
            transformed = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(transformed)
            feature_names = preprocessor.get_feature_names_out(model.feature_names_in_)
            base_names = [f.split("__")[1].split("_")[0] if "__" in f else f for f in feature_names]
            mean_abs = np.abs(shap_values).mean(axis=0)
            importance = pd.DataFrame({"feature": base_names, "value": mean_abs}) \
                         .groupby("feature")["value"].sum().sort_values(ascending=False)
            top = 4
            labels = importance.head(top).index.tolist() + ["Others"]
            values = importance.head(top).values.tolist() + [importance.iloc[top:].sum()]
            pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
            pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
            st.plotly_chart(pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ➤ Insights card
        st.markdown("<div class='card full'>", unsafe_allow_html=True)
        st.markdown("### 💡 Insights & Recommendations")

        if score >= 80 and label == "Low Risk":
            st.success("✅ Health score and risk are aligned. Keep up the good habits!")
        elif score < 60 and label == "High Risk":
            st.error("🚨 Low score & high risk. Consult a physician ASAP.")
        elif score >= 80 and label == "High Risk":
            st.warning("⚠️ High score but elevated risk. Schedule a full check-up.")
        else:
            st.info("🔍 Low score but currently low risk. Improve your lifestyle.")

        if latest["BMI"] > 25:
            st.write("• High BMI – focus on a balanced diet and exercise.")
        if latest["Heart_Rate"] > 90:
            st.write("• Elevated heart rate – consider stress reduction.")
        if latest["Systolic_BP"] > 130 or latest["Diastolic_BP"] > 85:
            st.write("• High BP – reduce salt and monitor regularly.")
        if "current" in str(latest["Smoking_Status"]).lower():
            st.write("• Smoking detected – strongly consider quitting.")
        if latest["Diabetes"]:
            st.write("• Diabetes – follow medication and monitor sugar.")
        if latest["Hyperlipidemia"]:
            st.write("• Hyperlipidemia – reduce saturated fats and stay active.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ➤ Visit History (unchanged)
    with tab2:
        st.markdown("## Visit History")
        st.info(f"Total Visits: {len(patient_df)} | Avg. Health Score: {round(patient_df['Health Score'].mean(), 1)}")
        metric = st.selectbox("Metric to View Trend", ["Health Score", "BMI", "Systolic_BP", "Heart_Rate"])
        st.line_chart(patient_df.set_index("Date")[metric])

        for _, row in patient_df.iterrows():
            risk = "High" if row["Heart_Disease"] == 1 else "Low"
            color = "#ff4d4d" if risk == "High" else "#4caf50"
            tips = []
            if row["BMI"] < 18.5 or row["BMI"] > 25:
                tips.append("• Maintain healthy BMI.")
            if row["Heart_Rate"] > 90:
                tips.append("• Reduce resting heart rate.")
            if row["Systolic_BP"] > 130 or row["Diastolic_BP"] > 85:
                tips.append("• Manage blood pressure.")
            if "current" in str(row["Smoking_Status"]).lower():
                tips.append("• Stop smoking.")
            if row["Hyperlipidemia"]:
                tips.append("• Monitor cholesterol.")
            if row["Diabetes"]:
                tips.append("• Track glucose levels.")

            st.markdown(
                f"""
                <div style='border:1px solid #ccc; border-radius:10px; padding:10px; background:#f9f9f9; margin:10px 0;'>
                <b>Date:</b> {row['Date'].date()}<br>
                <b>BMI:</b> {row['BMI']}, <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']}, <b>HR:</b> {row['Heart_Rate']}<br>
                <b>Health Score:</b> {row['Health Score']}, <b>Risk:</b> <span style='color:white; background:{color}; padding:2px 6px; border-radius:4px;'>{risk}</span><br>
                <b>Tips:</b> <br>{'<br>'.join(tips)}
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

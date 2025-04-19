import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

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
        margin=dict(t=0, b=0, l=0, r=0),
        height=170,
        width=170,
        annotations=[dict(text=text, font_size=14, showarrow=False)]
    )
    return fig

def plot_feature_importance(model):
    importances = model.named_steps["classifier"].feature_importances_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feature_names, importances, color="#4caf50")
    ax.set_title("🔍 Feature Importance for Heart Risk")
    ax.invert_yaxis()
    st.pyplot(fig)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.patient_id = ""

def show_login():
    st.title("Welcome to HealthPredict 🩺")
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

    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    with tab1:
        with st.sidebar:
            st.markdown("## 📅 Book Appointment")
            doctor = st.selectbox("Choose Doctor", ["Cardiologist", "General Physician", "Endocrinologist"])
            appt_date = st.date_input("Select Date", min_value=date.today())
            notes = st.text_input("Notes (optional)")
            if st.button("Book Appointment"):
                st.success(f"✅ Appointment booked with {doctor} on {appt_date.strftime('%b %d, %Y')}")

        st.markdown("## 👤 Patient Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Patient ID**: {patient_id}")
            st.markdown(f"**Visit Date**: {latest['Date'].date()}")
            st.markdown(f"**Height**: {latest['Height_cm']} cm")
            st.markdown(f"**Weight**: {latest['Weight_kg']} kg")
            st.markdown(f"**Smoking**: {latest['Smoking_Status']}")
        with c2:
            st.metric("BMI", latest["BMI"])
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 🧬 Health Score")
            score = latest["Health_Score"]
            color = "#4caf50" if score >= 80 else "#ffa94d" if score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, color), use_container_width=True)

        with c4:
            st.markdown("### 🧠 Heart Risk")
            try:
                model = joblib.load("heart_disease_model_with_importance.pkl")
                input_df = pd.DataFrame([{
                    "BMI": latest["BMI"],
                    "Systolic_BP": latest["Systolic_BP"],
                    "Diastolic_BP": latest["Diastolic_BP"],
                    "Heart_Rate": latest["Heart_Rate"],
                    "Smoking_Status": latest["Smoking_Status"],
                    "Diabetes": latest["Diabetes"],
                    "Hyperlipidemia": latest["Hyperlipidemia"]
                }])
                prediction = model.predict(input_df)[0]
                risk_label = "High Risk" if prediction == 1 else "Low Risk"
                risk_color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart(risk_label, 50, risk_color, show_score=False), use_container_width=True)

                # Feature importance visualization
                plot_feature_importance(model)

            except Exception as e:
                st.error(f"Model error: {e}")

        st.markdown("### 🛡️ Preventive Measures")
        with st.container():
            if latest["BMI"] < 18.5 or latest["BMI"] > 25:
                st.write(f"• BMI ({latest['BMI']}) – Adjust diet & exercise.")
            if latest["Heart_Rate"] > 90:
                st.write(f"• High Heart Rate – Manage stress, increase activity.")
            if latest["Systolic_BP"] > 130:
                st.write("• High Blood Pressure – Limit salt, monitor regularly.")
            if str(latest["Smoking_Status"]).lower().startswith("current"):
                st.write("• Smoking – Enroll in cessation program.")

    with tab2:
        st.markdown("## 📅 Visit History")
        st.info(f"Total Visits: {len(patient_df)} | Avg. Score: {round(patient_df['Health_Score'].mean(), 1)}")

        metric = st.selectbox("View Trend Over Time", ["Health_Score", "BMI", "Systolic_BP", "Heart_Rate"])
        st.line_chart(patient_df.set_index(pd.to_datetime(patient_df["Date"]))[metric])

        for _, row in patient_df.iterrows():
            risk = "High" if row["Heart_Disease"] == 1 else "Low"
            color = "#ff4d4d" if row["Heart_Disease"] == 1 else "#4caf50"
            tips = []
            if row["BMI"] < 18.5 or row["BMI"] > 25:
                tips.append("• Adjust BMI")
            if row["Heart_Rate"] > 90:
                tips.append("• High Heart Rate")
            if row["Systolic_BP"] > 130:
                tips.append("• High BP")
            if str(row["Smoking_Status"]).lower().startswith("current"):
                tips.append("• Quit Smoking")
            tip_text = "<br>".join(tips)
            st.markdown(
                f"""<div style='border:1px solid #ccc;border-radius:10px;padding:10px;margin:10px 0;background:#f9f9f9;'>
                <b>🗓 Visit Date:</b> {pd.to_datetime(row['Date']).date()}<br>
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

# Run
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

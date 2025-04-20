import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import date
from sklearn.preprocessing import LabelEncoder

# ---------- CONFIG ----------
st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

# ---------- DONUT CHART ----------
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

# ---------- LOGIN ----------
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

# ---------- DASHBOARD ----------
def show_dashboard(patient_id):
    patient_df = df[df["patient"].astype(str) == patient_id].sort_values("Date")
    latest = patient_df.iloc[-1]

    tab1, tab2 = st.tabs(["📊 Overview", "📅 Visit History"])

    # ---------- OVERVIEW ----------
    with tab1:
        st.markdown("## 👤 Patient Overview")
        st.markdown("###")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown("""
            <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px;'>
            <h5>Personal Details</h5>
            <p><b>Patient ID:</b> {}</p>
            <p><b>Date:</b> {}</p>
            <p><b>Height:</b> {} cm</p>
            <p><b>Weight:</b> {} kg</p>
            <p><b>Smoking:</b> {}</p>
            <p><b>Diabetes:</b> {}</p>
            <p><b>Hyperlipidemia:</b> {}</p>
            <p><b>Heart Disease:</b> {}</p>
            </div>
            """.format(
                patient_id,
                latest["Date"].date(),
                latest["Height_cm"],
                latest["Weight_kg"],
                latest["Smoking_Status"],
                latest["Diabetes"],
                latest["Hyperlipidemia"],
                "Yes" if latest["Heart_Disease"] == 1 else "No"
            ), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px;'>
            <h5>Health Metrics</h5>
            """, unsafe_allow_html=True)
            st.metric("BMI", latest["BMI"])
            st.metric("Blood Pressure", f"{latest['Systolic_BP']}/{latest['Diastolic_BP']}")
            st.metric("Heart Rate", f"{latest['Heart_Rate']} bpm")
            st.markdown("</div>", unsafe_allow_html=True)

        # Donut Charts
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("### Health Score")
            score = latest["Health_Score"]
            score_color = "#4caf50" if score >= 80 else "#ffa94d" if score >= 60 else "#ff4d4d"
            st.plotly_chart(donut_chart("Score", score, score_color), use_container_width=True)

        with d2:
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
                le.fit(df["Smoking_Status"])
                input_df["Smoking_Status"] = le.transform(input_df["Smoking_Status"])

                prediction = model.predict(input_df)[0]
                label = "High Risk" if prediction == 1 else "Low Risk"
                color = "#ff4d4d" if prediction == 1 else "#4caf50"
                st.plotly_chart(donut_chart(label, 50, color, show_score=False), use_container_width=True)
            except Exception as e:
                st.error(f"Model error: {e}")

        # Preventive Measures Card
        st.markdown("### Preventive Measures")
        pm = []
        if latest["BMI"] < 18.5 or latest["BMI"] > 25:
            pm.append(f"• BMI ({latest['BMI']}) – Adjust diet & exercise.")
        if latest["Heart_Rate"] > 90:
            pm.append("• High Heart Rate – Manage stress, increase activity.")
        if latest["Systolic_BP"] > 130:
            pm.append("• High BP – Limit salt, regular checkups.")
        if str(latest["Smoking_Status"]).lower().startswith("current"):
            pm.append("• Smoking – Enroll in cessation program.")

        st.markdown(f"""
        <div style='background-color:#f9f9f9; padding: 15px; border-radius: 10px;'>
        {"<br>".join(pm) if pm else "✔️ No critical issues detected at this time."}
        </div>
        """, unsafe_allow_html=True)

    # ---------- VISIT HISTORY ----------
    with tab2:
        st.markdown("## 📅 Visit History")
        st.info(f"Total Visits: {len(patient_df)} | Avg. Score: {round(patient_df['Health_Score'].mean(), 1)}")

        selected_metric = st.selectbox("Choose Metric to View Over Time", ["Health_Score", "BMI", "Systolic_BP", "Heart_Rate"])
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
                tips.append("• High BP")
            if str(row["Smoking_Status"]).lower().startswith("current"):
                tips.append("• Smoking Cessation")

            tip_text = "<br>".join(tips)
            st.markdown(
                f"""<div style='border:1px solid #ccc;border-radius:10px;padding:10px;margin:10px 0;background:#f9f9f9;'>
                <b>🗓 Visit Date:</b> {row['Date'].date()}<br>
                <b>Height:</b> {row['Height_cm']} cm | <b>Weight:</b> {row['Weight_kg']} kg | <b>BMI:</b> {row['BMI']}<br>
                <b>BP:</b> {row['Systolic_BP']}/{row['Diastolic_BP']} | <b>Heart Rate:</b> {row['Heart_Rate']} bpm<br>
                <b>Health Score:</b> {row['Health_Score']} | 
                <b>Heart Risk:</b> <span style='background:{color};color:white;padding:2px 6px;border-radius:4px;'>{risk}</span><br>
                <b>🛡️ Tips:</b><br>{tip_text}
                </div>
                """, unsafe_allow_html=True
            )

    if st.button("🔙 Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = ""
        st.rerun()

# ---------- MAIN ----------
if st.session_state.logged_in:
    show_dashboard(st.session_state.patient_id)
else:
    show_login()

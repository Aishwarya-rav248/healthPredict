
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Health Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("selected_20_final_patients.csv")

df = load_data()

st.sidebar.title("🏥 Health Dashboard")
patient_ids = df["patient"].unique()
selected_patient = st.sidebar.selectbox("Select Patient", patient_ids)

patient_df = df[df["patient"] == selected_patient].sort_values("date")
latest = patient_df.iloc[-1]

st.markdown("<h2 style='text-align:center;'>Patient Overview</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div style='background-color:#fff;padding:1rem 1.5rem;border-radius:10px;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
            <h4>🧍 Patient Info</h4>
            <p><strong>ID:</strong> {selected_patient}</p>
            <p><strong>Date:</strong> {latest["date"]}</p>
            <p><strong>Height:</strong> {latest["Height_cm"]} cm</p>
            <p><strong>Weight:</strong> {latest["Weight_kg"]} kg</p>
            <p><strong>Smoking:</strong> {latest["Smoking_Status"]}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style='background-color:#fff;padding:1rem 1.5rem;border-radius:10px;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
            <h4>📊 Health Metrics</h4>
        </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("BMI", latest["BMI"])
    m2.metric("Blood Pressure", f"{latest["Systolic_BP"]}/{latest["Diastolic_BP"]}")
    m1.metric("Heart Rate", latest["Heart_Rate"])
    m2.metric("Risk Level", latest["Risk_Level"])

    score = latest["Health_Score"]
    color = "green" if score >= 85 else "orange" if score >= 70 else "red"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': color},
               'steps': [{'range': [0, 100], 'color': color}]},
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<h4>📈 Health Score Over Time</h4>", unsafe_allow_html=True)
trend_df = patient_df[["date", "Health_Score"]].copy()
trend_df["date"] = pd.to_datetime(trend_df["date"])
st.line_chart(trend_df.set_index("date"))

st.markdown("<h4>🕒 Visit History</h4>", unsafe_allow_html=True)
for _, row in patient_df.iterrows():
    with st.expander(f"Visit on {row['date']}"):
        st.write(f"**Height:** {row['Height_cm']} cm")
        st.write(f"**Weight:** {row['Weight_kg']} kg")
        st.write(f"**BMI:** {row['BMI']}")
        st.write(f"**Blood Pressure:** {row['Systolic_BP']}/{row['Diastolic_BP']}")
        st.write(f"**Heart Rate:** {row['Heart_Rate']}")
        st.write(f"**Smoking Status:** {row['Smoking_Status']}")
        st.write(f"**Health Score:** {row['Health_Score']}")
        st.write(f"**Risk Level:** {row['Risk_Level']}")

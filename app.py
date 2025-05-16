import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Page Config ----------
st.set_page_config(page_title="Diabetes Predictor", layout="wide", page_icon="🩺")

# ---------- Custom CSS for Alerts ----------
hide_streamlit_style = """
<style>
/* Hide hamburger menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom alert styles */
.alert {
    padding: 20px;
    color: white;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    border-radius: 8px;
    margin-top: 20px;
    position: relative;
}
.success {background-color: #2ecc71;}
.warning {background-color: #e74c3c;}
.closebtn {
    position: absolute;
    top: 5px;
    right: 20px;
    color: white;
    font-weight: bold;
    cursor: pointer;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# ---------- Sidebar ----------
st.sidebar.title("🧪 Input Features")
st.sidebar.markdown("Enter patient information:")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 2)
glucose = st.sidebar.slider("Glucose", 0, 200, 100)
bp = st.sidebar.slider("Blood Pressure", 0, 122, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 846, 79)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 33)

input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

# ---------- Train Model ----------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
input_scaled = scaler.transform(input_data)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1] * 100

# ---------- Main Area ----------
st.title("💉 Diabetes Prediction Dashboard")

# ---------- Preview Data ----------
st.subheader("📋 Sample Data")
st.dataframe(df.sample(5), use_container_width=True)

# ---------- Charts ----------
col1, col2 = st.columns(2)

with col1:
    pie_fig = px.pie(df, names="Outcome", title="Diabetes Outcome Distribution", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(pie_fig, use_container_width=True)

with col2:
    hist_fig = px.histogram(df, x="Age", color="Outcome", barmode="overlay", title="Age Distribution by Outcome")
    st.plotly_chart(hist_fig, use_container_width=True)

# ---------- Prediction Result ----------
st.subheader("🩺 Prediction Result")

if "hide_alert" not in st.session_state:
    st.session_state.hide_alert = False

if st.button("🔍 Predict"):
    st.session_state.hide_alert = False

if not st.session_state.hide_alert:
    if probability > 50:
        st.markdown(f"""
        <div class="alert warning">
            <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span>
            🚨 High Risk: You have a {probability:.2f}% chance of having diabetes. Please consult your doctor and take preventive measures.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert success">
            <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span>
            ✅ Low Risk: You have only a {probability:.2f}% chance of having diabetes. Keep up your healthy lifestyle!
        </div>
        """, unsafe_allow_html=True)

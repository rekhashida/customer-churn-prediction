import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction App")
st.markdown("**Predict whether a telecom customer will churn using Machine Learning**")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",
    ["No phone service", "No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service",
    ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security",
    ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup",
    ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection",
    ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support",
    ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV",
    ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",
    ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract Type",
    ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 119.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 8685.0, 1500.0)

# Encode inputs
def encode(val, mapping):
    return mapping[val]

input_data = {
    'gender': encode(gender, {'Female': 0, 'Male': 1}),
    'SeniorCitizen': encode(senior, {'No': 0, 'Yes': 1}),
    'Partner': encode(partner, {'No': 0, 'Yes': 1}),
    'Dependents': encode(dependents, {'No': 0, 'Yes': 1}),
    'tenure': tenure,
    'PhoneService': encode(phone_service, {'No': 0, 'Yes': 1}),
    'MultipleLines': encode(multiple_lines,
        {'No phone service': 0, 'No': 1, 'Yes': 2}),
    'InternetService': encode(internet_service,
        {'DSL': 0, 'Fiber optic': 1, 'No': 2}),
    'OnlineSecurity': encode(online_security,
        {'No': 0, 'Yes': 1, 'No internet service': 2}),
    'OnlineBackup': encode(online_backup,
        {'No': 0, 'Yes': 1, 'No internet service': 2}),
    'DeviceProtection': encode(device_protection,
        {'No': 0, 'Yes': 1, 'No internet service': 2}),
    'TechSupport': encode(tech_support,
        {'No': 0, 'Yes': 1, 'No internet service': 2}),
    'StreamingTV': encode(streaming_tv,
        {'No': 0, 'Yes': 1, 'No internet service': 2}),
    'StreamingMovies': encode(streaming_movies,
        {'No': 0, 'Yes': 1, 'No internet service': 2}),
    'Contract': encode(contract,
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2}),
    'PaperlessBilling': encode(paperless, {'No': 0, 'Yes': 1}),
    'PaymentMethod': encode(payment,
        {'Electronic check': 0, 'Mailed check': 1,
         'Bank transfer (automatic)': 2,
         'Credit card (automatic)': 3}),
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Main area — show customer summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tenure", f"{tenure} months")
with col2:
    st.metric("Monthly Charges", f"${monthly_charges}")
with col3:
    st.metric("Contract Type", contract)

st.markdown("---")

# Predict button
if st.button("🔮 Predict Churn", use_container_width=True):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("## Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH CHURN RISK — This customer is likely to churn!")
        st.metric("Churn Probability", f"{probability:.1%}")
        st.markdown("### 💡 Recommended Actions:")
        st.markdown("- Offer a discount to switch to annual contract")
        st.markdown("- Assign a dedicated customer success manager")
        st.markdown("- Provide free upgrade on current services")
        st.markdown("- Send personalized retention offer within 24 hours")
    else:
        st.success(f"✅ LOW CHURN RISK — This customer is likely to stay!")
        st.metric("Churn Probability", f"{probability:.1%}")
        st.markdown("### 💡 Recommended Actions:")
        st.markdown("- Continue regular engagement")
        st.markdown("- Offer loyalty rewards")
        st.markdown("- Upsell premium services")

    # Show probability bar
    st.markdown("### Churn Probability")
    st.progress(float(probability))

st.markdown("---")
st.markdown("**Model:** Logistic Regression | **Accuracy:** 79.84% | **ROC-AUC:** 0.8404")
st.markdown("**Built by:** Rekha Sida | github.com/rekhashida")
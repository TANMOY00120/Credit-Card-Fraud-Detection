import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the model
model = joblib.load("credit_card_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Transaction Checker", layout="centered")
st.title("💳 Credit Card Transaction Checker")

st.markdown("Enter your transaction details to check if it's fraudulent or not.")

# Input section
col1, col2 = st.columns(2)
with col1:
    amount = st.slider("Transaction Amount (₹)", 0.0, 5000.0, 100.0, step=1.0)
with col2:
    time = st.slider("Transaction Time (in seconds from start of the day)", 0, 172800, 60000)

# Optionally simulate background transaction features (V1–V28)
# These are normally hidden in real systems
auto_generate = st.checkbox("Auto-generate background transaction data (recommended)", value=True)

if auto_generate:
    v_features = np.random.normal(0, 1, 28)
else:
    st.markdown("Enter technical transaction features (V1 to V28):")
    v_features = []
    for i in range(1, 29):
        v = st.number_input(f"V{i}", value=0.0)
        v_features.append(v)

# Final input vector
transaction_data = list(v_features) + [amount]
input_df = pd.DataFrame([transaction_data], columns=[f"V{i}" for i in range(1, 29)] + ["Amount"])

# Prediction
if st.button("🔍 Check Transaction"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔎 Result")
    if prediction == 1:
        st.error(f"❗ Fraudulent Transaction Detected\n🧪 Probability: {proba:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction\n🧪 Probability of Fraud: {proba:.2%}")

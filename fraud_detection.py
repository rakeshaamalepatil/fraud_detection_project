import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model pipeline
# The model file 'fraud_detection_pipeline.pkl' should be in the same directory
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except FileNotFoundError:
    st.error("Model file 'fraud_detection_pipeline.pkl' not found. Please upload it to the same directory.")
    st.stop()

# --- Streamlit App UI ---
st.title("Fraud Detection Prediction App")

st.markdown("Please enter the transaction details and use the predictor button below.")

st.divider()

# Input fields for user data
# These must match the features used to train the model
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Prediction button
if st.button("Predict"):
    # Create a DataFrame from the user's input
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error("This transaction is predicted to be a **fraudulent transaction**.")
    else:
        st.success("This transaction is predicted to be a **non-fraudulent transaction**.")
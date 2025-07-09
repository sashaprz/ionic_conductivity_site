import streamlit as st
import requests

st.title("Ionic Conductivity Predictor")

composition = st.text_input("Enter composition (e.g. Li2O):")

if st.button("Predict"):
    response = requests.post(
        "http://localhost:5000/predict",
        json={"composition": composition}
    )
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error("Error: Could not get prediction.")

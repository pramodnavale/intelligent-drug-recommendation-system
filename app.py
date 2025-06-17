import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open("ML2.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Drug Type Prediction")

# Input fields
age = st.slider("Age", 15, 100)
sex = st.selectbox("Sex", ["F", "M"])
bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
cholesterol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na = st.number_input("Sodium (Na) level")
k = st.number_input("Potassium (K) level")

# Convert categorical to numerical if needed
def encode_input(sex, bp, cholesterol):
    return [
        0 if sex == "F" else 1,
        {"LOW": 0, "NORMAL": 1, "HIGH": 2}[bp],
        {"NORMAL": 0, "HIGH": 1}[cholesterol],
    ]

sex_num, bp_num, chol_num = encode_input(sex, bp, cholesterol)

# Make prediction
input_data = [[age, sex_num, bp_num, chol_num, na, k]]
if st.button("Predict Drug Type"):
    result = model.predict(input_data)
    st.success(f"Predicted Drug Type: {result[0]}")

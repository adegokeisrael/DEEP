import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load("random_forest_iris_model.pkl")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification App")
st.write("Predict the species of an Iris flower using a Random Forest model.")

# ============================================================
# USER INPUTS
# ============================================================

st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create DataFrame
input_data = pd.DataFrame({
    "SepalLengthCm": [sepal_length],
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width]
})

st.subheader("Input Data")
st.write(input_data)

# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict"):

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    st.subheader("Prediction")
    st.success(f"Predicted Species: {prediction[0]}")

    st.subheader("Prediction Probability")
    st.write(probabilities)
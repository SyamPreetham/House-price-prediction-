import streamlit as st
import joblib
import numpy as np
import os

# --- Load Model Safely ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # absolute path
    model_path = os.path.join(BASE_DIR, "model.pkl")

    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# --- App UI ---
st.title("🏠 House Price Predictions")
st.divider()

st.write(
    "This app uses a trained Machine Learning model to predict house prices. "
    "Enter the features below and click **Predict** to see the estimated price."
)
st.divider()

# Inputs
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=0)
livingarea = st.number_input("Living Area (sq ft)", min_value=0, value=2000)
condition = st.number_input("Condition (1-5)", min_value=0, value=3)
numberofschools = st.number_input("Number of Schools Nearby", min_value=0, value=0)

st.divider()

# Prepare input
x = [[bedrooms, bathrooms, livingarea, condition, numberofschools]]

# Predict button
if st.button("Predict"):
    st.balloons()
    x_array = np.array(x)

    try:
        prediction = model.predict(x_array)  # ✅ predict method
        st.success(f"💰 Estimated House Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
else:
    st.info("Please enter values and then click the **Predict** button.")

       
       
       
       
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('lr_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="UK Accident Severity Predictor", page_icon="", layout="centered")

st.title("UK Accident Severity Prediction App")
st.write("Enter the accident details below to predict the severity.")

# Input fields for user data
speed_limit = st.number_input("Speed Limit", min_value=10, max_value=70, value=30)
urban_or_rural_area = st.selectbox("Area Type", options=[1, 2], format_func=lambda x: "Urban" if x == 1 else "Rural")
light_conditions = st.selectbox("Light Conditions", options=[1, 4, 5, 6, 7], format_func=lambda x: "Daylight" if x == 1 else "Darkness")
weather_conditions = st.selectbox("Weather Conditions", options=[1, 2, 4, 5, 6, 7], format_func=lambda x: "Fine" if x == 1 else "Other")

if st.button("Predict Severity"):
    # Prepare input array
    input_data = np.array([[speed_limit, urban_or_rural_area, light_conditions, weather_conditions]])
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.error("Prediction: High Accident Severity")
    else:
        st.success("Prediction: Low/Moderate Accident Severity")

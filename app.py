import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="UK Accident Severity Predictor",
                   page_icon="🚗", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

pipe = load_model()

cats = pipe.named_steps['prep'].named_transformers_['cat'].categories_
light_options = list(cats[0])
weather_options = list(cats[1])

labels = {1: "Fatal", 2: "Serious", 3: "Slight"}

st.title("UK Accident Severity Prediction App")
st.write("Enter the accident details below to predict the severity.")

speed_limit = st.selectbox("Speed Limit", [20, 30, 40, 50, 60, 70], index=1)
area = st.selectbox("Area Type", [1, 2],
                    format_func=lambda x: "Urban" if x == 1 else "Rural")
light = st.selectbox("Light Conditions", light_options)
weather = st.selectbox("Weather Conditions", weather_options)

if st.button("Predict Severity"):
    row = pd.DataFrame([{
        'Speed_limit': speed_limit,
        'Urban_or_Rural_Area': area,
        'Light_Conditions': light,
        'Weather_Conditions': weather
    }])
    pred = pipe.predict(row)[0]
    probs = pipe.predict_proba(row)[0]

    if pred == 3:
        st.success("Prediction: Slight severity")
    elif pred == 2:
        st.warning("Prediction: Serious severity")
    else:
        st.error("Prediction: Fatal severity")

    st.write("Probabilities:")
    for cls, p in zip(pipe.classes_, probs):
        st.write(f"{labels[cls]}: {p:.1%}")
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='UK Accident Severity Predictor', page_icon='', layout='centered'
)

# --- LOAD SAVED MODEL & SCALER ---
# Make sure 'lr_model.pkl' and 'scaler.pkl' are in the same folder as app.py


@st.cache_resource
def load_assets():
  model = joblib.load('lr_model.pkl')
  scaler = joblib.load('scaler.pkl')
  return model, scaler


try:
  model, scaler = load_assets()
except Exception as e:
  st.error(
      f'Error loading model or scaler files. Make sure they are uploaded: {e}'
  )

# --- APP HEADER ---
st.title(' UK Accident Severity Predictor')
st.markdown(
    'Enter the accident details below to predict its severity class (ranging'
    ' from Slight to Serious).'
)
st.divider()

# --- USER INPUT FORM ---
st.subheader('Accident Parameters')

col1, col2 = st.columns(2)

with col1:
  hour = st.slider('Hour of Day', min_value=0, max_value=23, value=12)
  longitude = st.number_input('Longitude', value=-0.1278, format='%.5f')
  location_easting = st.number_input(
      'Location Easting (OSGR)', value=529000.0, format='%.1f'
  )

with col2:
  police_attended = st.selectbox(
      'Did Police Officer Attend Scene?', ['Yes', 'No']
  )
  weather = st.selectbox(
      'Weather Conditions',
      [
          'Fine without high winds',
          'Raining without high winds',
          'Snowing',
          'Other',
      ],
  )
  light_cond = st.selectbox(
      'Light Conditions',
      [
          'Daylight: Street light present',
          'Darkness: Lighting unknown',
          'Darkness: No lighting',
      ],
  )

road_type = st.selectbox(
    'Road Type', ['Single carriageway', 'One way street', 'Dual carriageway']
)

# --- PREDICTION LOGIC ---
if st.button('Predict Accident Severity', type='primary'):
  input_data = pd.DataFrame(
      {
          'Hour': [hour],
          'Longitude': [longitude],
          'Location_Easting_OSGR': [location_easting],
          'Did_Police_Officer_Attend_Scene_of_Accident_Yes': [
              1 if police_attended == 'Yes' else 0
          ],
          'Weather_Conditions_Raining without high winds': [
              1 if weather == 'Raining without high winds' else 0
          ],
          'Light_Conditions_Daylight: Street light present': [
              1 if light_cond == 'Daylight: Street light present' else 0
          ],
          'Road_Type_One way street': [1 if road_type == 'One way street' else 0],
      }
  )

  try:
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader('Prediction Result')

    class_labels = {1: 'Fatal (Class 1)', 2: 'Serious (Class 2)', 3: 'Slight (Class 3)'}
    predicted_label = class_labels.get(prediction, f'Class {prediction}')

    if prediction == 2:
      st.warning(f' Predicted Severity: **{predicted_label}**')
    elif prediction == 1:
      st.error(f' Predicted Severity: **{predicted_label}**')
    else:
      st.success(f' Predicted Severity: **{predicted_label}**')

    st.write('### Confidence Breakdown')
    prob_df = pd.DataFrame(
        {
            'Severity Class': ['Fatal (1)', 'Serious (2)', 'Slight (3)'],
            'Probability': probabilities,
        }
    )
    st.bar_chart(prob_df.set_index('Severity Class'))

  except Exception as e:
    st.error(
        f'An error occurred during prediction. Check feature alignment: {e}'
    )

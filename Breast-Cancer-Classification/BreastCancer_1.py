import streamlit as st
import pandas as pd
import numpy as np

import joblib

# load data
data = pd.read_csv('data_cancer.csv')

# load model & Scaler
model = joblib.load('Breast_Cancer_model.pkl')
scaler = joblib.load('Scaler.pkl')

X = data.drop('diagnosis', axis=1)

st.title('Breast Cancer Detection')
st.write('Enter your values of the features below...')

input_data = []
for feature in X.columns:
    value = st.number_input(f"{feature}, min_value = 0.0 , max_value = 1000")
    input_data.append(value)

if st.button('predict'):
    input_arr = np.array(input_data).reshape(1, -1)
    input_std = scaler.transform(input_arr)
    pred = model.predict(input_std)

    if pred [0] == 1:
        st.error('Prediction: Malignant (Cancer Detected)')
    else:
        st.success('Prediction: Benignant  (No Cancer Detected)')


import streamlit as st
import requests
import pandas as pd

st.title('COVID-19 Diagnosis from Breath Analysis')

st.header('Enter Patient Data')

# Create input fields for each feature
input_data = {}
for i in range(1, 65):
    input_data[f'D{i}'] = st.number_input(f'D{i}')

# Button for making the prediction
if st.button('Predict'):
    response = requests.post('http://localhost:8000/predict/', json=input_data)
    if response.status_code == 200:
        st.write(f'The predicted class is: {response.json()["prediction"]}')
    else:
        st.write('Error in prediction. Please check the input values.')

st.header('Upload CSV File for Batch Prediction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'Patient_ID' in data.columns:
        data = data.drop(columns=['Patient_ID'])
        
    if st.button('Predict for CSV'):
        response = requests.post('http://localhost:8000/predict_batch/', json=data.to_dict(orient='records'))
        if response.status_code == 200:
            st.write('Predictions:')
            st.write(response.json()['predictions'])
        else:
            st.write('Error in prediction. Please check the input values.')

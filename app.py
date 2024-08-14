import streamlit as st
import requests
import pandas as pd

st.title('COVID-19 Diagnosis from Breath Analysis')

st.header('Entrez les informations de patient')

# Create input fields for each feature
input_data = {}
for i in range(1, 65):
    input_data[f'D{i}'] = st.number_input(f'D{i}')

# Button for making the prediction
if st.button('Predict'):
    response = requests.post('https://covid-19-api3.onrender.com/predict/', json=input_data)
    if response.status_code == 200:
        st.write(f'RESULTAT DU DIAGNOSTIQUE : {response.json()["prediction"]}')
    else:
        st.write('Error in prediction. Please check the input values.')

st.header('FAIRE LE DIAGNOSTIQUE DE PLUSIEUR PATIENT ')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'Patient_ID' in data.columns:
        data = data.drop(columns=['Patient_ID'])
        
    if st.button('Predict for CSV'):
        response = requests.post('https://covid-19-api3.onrender.com/predict_batch/', json=data.to_dict(orient='records'))
        if response.status_code == 200:
            st.write('Predictions:')
            st.write(response.json()['predictions'])
        else:
            st.write('Error in prediction. Please check the input values.')

with st.sidebar:
    st.write("**Mes Coordonnées :**")
    st.write("**Nom:** MAMA Moussinou")
    st.write("**Email:** mamamouhsinou@gmail.com")
    st.write("**Téléphone:** +229 95231680")
    st.write("**LinkedIn:** [moussinou-mama-8b6270284](https://www.linkedin.com/in/moussinou-mama-8b6270284/)")
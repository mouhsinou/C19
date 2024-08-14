import streamlit as st
import requests
import pandas as pd

st.title('COVID-19 Diagnosis from Breath Analysis')

# Sidebar with user info
with st.sidebar:
    st.write("**Mes Coordonnées :**")
    st.write("**Nom:** MAMA Moussinou")
    st.write("**Email:** mamamouhsinou@gmail.com")
    st.write("**Téléphone:** +229 95231680")
    st.write("**LinkedIn:** [moussinou-mama-8b6270284](https://www.linkedin.com/in/moussinou-mama-8b6270284/)")

# User choice for what they want to do
st.header('Que voulez-vous faire ?')
option = st.selectbox(
    "Choisissez une option",
    ('Préparation des données', 'Prédiction pour un patient', 'Prédiction pour plusieurs patients')
)

# Data Preparation: Convert .txt to .csv
if option == 'Préparation des données':
    st.subheader("Convertir un fichier .txt en fichier .csv")
    uploaded_txt = st.file_uploader("Choisissez un fichier .txt", type="txt")
    
    if uploaded_txt is not None:
        txt_content = uploaded_txt.read().decode("utf-8")
        # Assuming the .txt file contains data in a structured format
        # Convert the txt content into a dataframe (customize based on your .txt format)
        lines = txt_content.splitlines()
        data = [line.split() for line in lines]
        df = pd.DataFrame(data)
        
        st.write("Voici un aperçu des données converties :")
        st.dataframe(df)
        
        # Save as CSV
        csv_file = df.to_csv(index=False)
        st.download_button(label="Télécharger en CSV", data=csv_file, file_name="data.csv", mime="text/csv")

# Single patient prediction
elif option == 'Prédiction pour un patient':
    st.subheader("Entrez les informations du patient")

    # Create input fields for each feature
    input_data = {}
    col1, col2 = st.columns(2)
    for i in range(1, 33):
        input_data[f'D{i}'] = col1.number_input(f'D{i}')
    for i in range(33, 65):
        input_data[f'D{i}'] = col2.number_input(f'D{i}')
    
    # Button for making the prediction
    if st.button('Faire la prédiction'):
        response = requests.post('https://covid-19-api3.onrender.com/predict/', json=input_data)
        if response.status_code == 200:
            st.success(f'RESULTAT DU DIAGNOSTIQUE : {response.json()["prediction"]}')
        else:
            st.error('Erreur dans la prédiction. Veuillez vérifier les valeurs saisies.')

# Batch prediction with CSV
elif option == 'Prédiction pour plusieurs patients':
    st.subheader("Prédiction pour plusieurs patients avec un fichier CSV")
    
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Check if 'Patient_ID' column exists
        if 'Patient_ID' in data.columns:
            patient_ids = data['Patient_ID']
            data = data.drop(columns=['Patient_ID'])
        else:
            patient_ids = pd.Series([f'Patient_{i+1}' for i in range(len(data))])

        if st.button('Faire la prédiction pour CSV'):
            response = requests.post('https://covid-19-api3.onrender.com/predict_batch/', json=data.to_dict(orient='records'))
            if response.status_code == 200:
                predictions = response.json()['predictions']
                
                # Create a DataFrame with Patient_ID and predictions
                result_df = pd.DataFrame({
                    'Patient_ID': patient_ids,
                    'Prediction': predictions
                })
                
                st.write("Résultats des prédictions :")
                st.dataframe(result_df)
            else:
                st.error('Erreur dans la prédiction. Veuillez vérifier les valeurs saisies.')

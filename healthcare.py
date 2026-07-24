import streamlit as st
import pandas as pd
import pickle
import time 

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Health Predict",
    page_icon="🩺",
    layout="wide"
)

# --- CHARGEMENT DU DATASET ET DU MODÈLE (avec mise en cache) ---
@st.cache_data
def load_data():
    return pd.read_csv('healthcare_dataset.csv')

@st.cache_resource
def load_model():
    with open('maladie.pkl', 'rb') as file:
        return pickle.load(file)

health = load_data()
modele = load_model()

# Extract pipeline and label encoder
pipeline = modele['pipeline'] 
la = modele['label_encoder']

# --- EN-TÊTE DE LA PAGE ---
st.title("🩺 Health Predict")
st.caption("Application d'assistance au diagnostic médical basée sur le Machine Learning")

st.markdown("""
Bienvenue sur **Health Predict**. Renseignez les données cliniques du patient dans le panneau de gauche 
afin de prédire la condition médicale probable.
""")

st.divider()

st.sidebar.header("📋 Profil du Patient")

with st.sidebar.form("patient_form"):
    st.subheader("Informations Générales")
    age = st.number_input('Âge', min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Sexe", options=['Male', 'Female'])
    blood = st.selectbox("Groupe Sanguin", options=health['Blood Type'].unique())
    insurance = st.selectbox('Assurance Santé', options=health['Insurance Provider'].unique())
    
    st.subheader("Détails Médicaux")
    admission = st.selectbox("Type d'admission", options=health['Admission Type'].unique())  
    medication = st.selectbox("Traitement prescrit", options=health['Medication'].unique())
    tests = st.selectbox("Résultat du test", options=health['Test Results'].unique())
    
    # Bouton de soumission dans la barre latérale
    submit_button = st.form_submit_button("Lancer la Prédiction", type="primary")

# --- CONSTRUCTION DU DATAFRAME DES ENTRÉES ---
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Blood Type': blood,
    'Insurance Provider': insurance,
    'Admission Type': admission,
    'Medication': medication,
    'Test Results': tests
}])

# --- PRÉSENTATION DES DONNÉES SAISIES ET RÉSULTAT ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Données à analyser")
    # Affichage propre des données saisies
    st.dataframe(input_data.T.rename(columns={0: "Valeur"}), use_container_width=True)

with col_right:
    st.subheader("Résultat du Diagnostic")
    
    if submit_button:
        with st.spinner("Analyse des données en cours..."):
            # Prédiction
            
            maladie_ch = pipeline.predict(input_data)
            maladie_txt = la.inverse_transform(maladie_ch)[0]
            
        # Résultat mis en avant avec st.metric
        st.success("Analyse terminée avec succès !")
        st.metric(label="Diagnostic Estimé", value=str(maladie_txt))
        
        st.info("💡 **Note :** Ce résultat est fourni à titre indicatif par un modèle prédictif et ne remplace pas un avis médical professionnel.")
    else:
        st.warning("Veuillez renseigner les données à gauche et cliquer sur **'Lancer la Prédiction'**.")
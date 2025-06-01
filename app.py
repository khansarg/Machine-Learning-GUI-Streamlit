import streamlit as st
import numpy as np
import joblib


# Load models and features
rf_model = joblib.load('random_forest_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Optional: hardcoded list of unique diseases
unique_diseases = sorted([
    'Acne', 'Alzheimer\'s Disease', 'Allergic Rhinitis', 'Anemia', 'Anxiety Disorders',
    'Appendicitis', 'Asthma', 'Atherosclerosis', 'Autism Spectrum Disorder (ASD)', 
    'Bipolar Disorder', 'Bladder Cancer', 'Brain Tumor', 'Breast Cancer', 'Bronchitis',
    'Cataracts', 'Cerebral Palsy', 'Chickenpox', 'Cholecystitis', 'Cholera', 'Chronic Kidney Disease',
    'Chronic Obstructive Pulmonary Disease (COPD)', 'Cirrhosis', 'Colorectal Cancer',
    'Common Cold', 'Conjunctivitis (Pink Eye)', 'Coronary Artery Disease', 'Crohn\'s Disease',
    'Cystic Fibrosis', 'Dementia', 'Dengue Fever', 'Depression', 'Diabetes', 'Diverticulitis',
    'Down Syndrome', 'Eating Disorders (Anorexia, ...)', 'Ebola Virus', 'Eczema',
    'Endometriosis', 'Epilepsy', 'Esophageal Cancer', 'Fibromyalgia', 'Gastroenteritis',
    'Glaucoma', 'Gout', 'Hemophilia', 'Hemorrhoids', 'Hepatitis', 'Hepatitis B',
    'HIV/AIDS', 'Hypertension', 'Hyperglycemia', 'Hyperthyroidism', 'Hypoglycemia',
    'Hypothyroidism', 'Influenza', 'Kidney Cancer', 'Kidney Disease', 'Klinefelter Syndrome',
    'Liver Cancer', 'Liver Disease', 'Lung Cancer', 'Lymphoma', 'Malaria', 'Marfan Syndrome',
    'Measles', 'Melanoma', 'Migraine', 'Multiple Sclerosis', 'Mumps', 'Muscular Dystrophy',
    'Myocardial Infarction (Heart...)', 'Obsessive-Compulsive Disorder', 'Osteoarthritis',
    'Osteomyelitis', 'Osteoporosis', 'Otitis Media (Ear Infection)', 'Ovarian Cancer',
    'Pancreatic Cancer', 'Pancreatitis', 'Parkinson\'s Disease', 'Polio',
    'Polycystic Ovary Syndrome (PCOS)', 'Pneumocystis Pneumonia (PCP)', 'Pneumonia',
    'Pneumothorax', 'Prader-Willi Syndrome', 'Prostate Cancer', 'Psoriasis', 'Rabies',
    'Rheumatoid Arthritis', 'Rubella', 'Schizophrenia', 'Scoliosis', 'Sepsis',
    'Sickle Cell Anemia', 'Sinusitis', 'Sleep Apnea', 'Spina Bifida', 'Stroke',
    'Systemic Lupus Erythematosus', 'Testicular Cancer', 'Tetanus', 'Thyroid Cancer',
    'Tonsillitis', 'Tourette Syndrome', 'Tuberculosis', 'Turner Syndrome',
    'Typhoid Fever', 'Ulcerative Colitis', 'Urinary Tract Infection', 
    'Urinary Tract Infection (UTI)', 'Williams Syndrome', 'Zika Virus'
])


# UI
st.title("Prediksi Penyakit - Random Forest")
st.text("Khansa Resqi Ghassani")
st.text("Fadhilah Kartika Firdausi")
st.text("Khaulah Qurota Ain")
# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)

gender = st.radio("Gender", ("Male", "Female"))

blood_pressure = st.selectbox("Blood Pressure", ("Low", "Normal", "High"))
cholesterol = st.selectbox("Cholesterol", ("Low", "Normal", "High"))

fever = st.radio("Fever", ("Yes", "No"))
cough = st.radio("Cough", ("Yes", "No"))
fatigue = st.radio("Fatigue", ("Yes", "No"))
difficulty_breathing = st.radio("Difficulty Breathing", ("Yes", "No"))

disease = st.selectbox("Disease", unique_diseases)

# Mapping categorical to numeric or one-hot
def map_yes_no(value): return 1 if value == "Yes" else 0
bp_map = {"Low": 0, "Normal": 1, "High": 2}
chol_map = {"Low": 0, "Normal": 1, "High": 2}

# One-hot encode gender
gender_encoded = {
    "Gender_Male": int(gender == "Male"),
    "Gender_Female": int(gender == "Female")
}

# One-hot encode disease
disease_encoded = {f"Disease_{d}": int(d == disease) for d in unique_diseases}

# Combine all into one input dictionary
input_dict = {
    "Age": age,
    "Fever": map_yes_no(fever),
    "Cough": map_yes_no(cough),
    "Fatigue": map_yes_no(fatigue),
    "Difficulty Breathing": map_yes_no(difficulty_breathing),
    "Blood Pressure": bp_map[blood_pressure],
    "Cholesterol Level": chol_map[cholesterol],
    **gender_encoded,
    **disease_encoded
}

# Ensure all features are present
user_input = [input_dict.get(f, 0) for f in feature_names]

# Predict
if st.button("Predict"):

    input_array = np.array(user_input).reshape(1, -1)
    prediction = rf_model.predict(input_array)[0]

    result = "Positif" if prediction == 1 else "Negatif"
    st.success(f"Hasil Prediksi: {result}")

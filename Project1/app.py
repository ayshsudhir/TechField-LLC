import streamlit as st
import pandas as pd
import joblib

# ────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit command
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥", layout="centered")

# ────────────────────────────────────────────────────────────
# LOAD MODEL, SCALER, FEATURE COLUMNS (cached so it only loads once)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_readmission_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ────────────────────────────────────────────────────────────
# TITLE
# ────────────────────────────────────────────────────────────
st.title("🏥 Hospital Readmission Risk Predictor")
st.write("Enter patient details to predict 30-day readmission risk.")

# ────────────────────────────────────────────────────────────
# INPUT FORM
# ────────────────────────────────────────────────────────────
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=8.0, max_value=70.0, value=25.0)
    primary_diagnosis = st.selectbox("Primary Diagnosis", 
        ["Diabetes", "Heart Failure", "Pneumonia", "COPD", 
         "Renal Failure", "Sepsis", "AMI", "Stroke"])
    num_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=3)
    num_medications = st.number_input("Number of Medications", min_value=0, max_value=40, value=8)
    num_prev_admissions = st.number_input("Prior Admissions (last 12mo)", min_value=0, max_value=20, value=1)
    length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=120, value=4)

with col2:
    emergency_admission = st.selectbox("Emergency Admission?", ["Yes", "No"])
    hba1c_level = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=20.0, value=6.0)
    glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=40.0, max_value=700.0, value=110.0)
    creatinine_level = st.number_input("Creatinine Level (mg/dL)", min_value=0.1, max_value=15.0, value=1.0)
    blood_pressure_systolic = st.number_input("Systolic BP (mmHg)", min_value=40, max_value=280, value=125)
    insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Uninsured"])
    discharged_to = st.selectbox("Discharged To", ["Home", "Rehab", "SNF", "Home Health", "AMA"])
    follow_up_scheduled = st.selectbox("Follow-up Scheduled?", ["Yes", "No"])
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

# ────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ────────────────────────────────────────────────────────────
if st.button("⚡ Predict Readmission Risk", type="primary"):

    # 1. Build a raw input dictionary matching ORIGINAL column names
    raw_input = {
        'age': age,
        'bmi': bmi,
        'num_diagnoses': num_diagnoses,
        'num_medications': num_medications,
        'num_prev_admissions': num_prev_admissions,
        'length_of_stay': length_of_stay,
        'emergency_admission': 1 if emergency_admission == "Yes" else 0,
        'hba1c_level': hba1c_level,
        'glucose_level': glucose_level,
        'creatinine_level': creatinine_level,
        'blood_pressure_systolic': blood_pressure_systolic,
        'follow_up_scheduled': 1 if follow_up_scheduled == "Yes" else 0,
        'labs_missing_flag': 0,   # new patient input -> no missing labs
        'gender': gender,
        'primary_diagnosis': primary_diagnosis,
        'insurance_type': insurance_type,
        'discharged_to': discharged_to,
        'smoking_status': smoking_status,
    }

    # 2. Convert to single-row DataFrame
    input_df = pd.DataFrame([raw_input])

    # 3. One-hot encode the SAME way training data was encoded
    categorical_cols = ['gender', 'primary_diagnosis', 'insurance_type',
                         'discharged_to', 'smoking_status']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    # 4. Align columns to match training data EXACTLY
    #    (adds any missing dummy columns as 0, drops any extras, fixes order)
    input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # 5. Scale using the SAME scaler from training
    input_scaled = scaler.transform(input_aligned)

    # 6. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of "Yes"

    # ────────────────────────────────────────────────────────
    # DISPLAY RESULT
    # ────────────────────────────────────────────────────────
    st.header("Prediction Result")

    if prediction == 1:
        st.error("YES")
    else:
        st.success("NO")

    # Probability bar visualization
    st.progress(float(probability))

    # ────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE CHART (satisfies "at least one visualization" requirement)
    # ────────────────────────────────────────────────────────
    st.subheader("What drives this model's predictions?")
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    st.bar_chart(importance_df.set_index('Feature'))
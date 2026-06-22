# Hospital Readmission Risk Dataset

This dataset contains inpatient encounter records used to predict whether a patient will be readmitted to the hospital within 30 days of discharge. It covers patient demographics, clinical measurements, diagnosis history, and discharge details.

---

## Columns

| Column | Description |
|---|---|
| `patient_id` | Unique patient identifier |
| `age` | Patient age in years |
| `gender` | Male / Female |
| `bmi` | Body Mass Index at admission |
| `primary_diagnosis` | Main condition driving the hospitalization |
| `num_diagnoses` | Total number of diagnoses recorded |
| `num_medications` | Number of medications prescribed |
| `num_prev_admissions` | Prior hospital admissions in the last 12 months |
| `length_of_stay` | Duration of stay in days |
| `emergency_admission` | 1 if admitted via emergency, 0 otherwise |
| `hba1c_level` | Glycated hemoglobin (%) — blood sugar control indicator |
| `glucose_level` | Blood glucose level (mg/dL) |
| `creatinine_level` | Serum creatinine (mg/dL) — kidney function indicator |
| `blood_pressure_systolic` | Systolic blood pressure at admission (mmHg) |
| `insurance_type` | Medicare / Medicaid / Private / Uninsured |
| `discharged_to` | Discharge destination (Home, Rehab, SNF, Home Health, AMA) |
| `follow_up_scheduled` | 1 if follow-up appointment was arranged, 0 otherwise |
| `smoking_status` | Current / Former / Never |
| `readmitted` | **Target** — Yes if readmitted within 30 days, No otherwise |

# Hospital Readmission Risk Prediction
## ML Capstone Project

A machine learning pipeline to predict 30-day hospital readmission risk using clinical, 
demographic, and administrative patient data. Includes a live Streamlit app for 
interactive predictions.

---

## Problem Statement

Hospital readmissions within 30 days of discharge cost the US healthcare system over 
$26 billion annually, with CMS penalizing hospitals up to $200K per year for excessive 
rates. Current clinical workflows rely on intuition and basic checklists — no 
data-driven early warning system exists at the point of discharge. This project builds 
a machine learning classifier to flag high-risk patients before they leave, enabling 
targeted intervention.

---

## Dataset

- **File:** `hospital_readmission_dataset.csv`
- **Size:** 15,000 patient records, 19 columns
- **Target:** `readmitted` — Yes if readmitted within 30 days, No otherwise
- **Features:** Patient demographics, clinical lab values, utilization metrics, 
  and discharge information
- **Note:** This is a synthetic dataset. The target variable appears to have been 
  generated using a near-deterministic function of clinical features (individual 
  feature-target correlations of 0.68–0.81, vs 0.05–0.25 in real EHR data). 
  The pipeline and methodology are valid — real-world performance would be lower 
  and would require revalidation on genuine hospital data.

---

## Project Structure
---

## Pipeline Overview

| Stage | Description |
|---|---|
| EDA | Target distribution, univariate/bivariate analysis, correlation heatmap, outlier detection |
| Missing Data | MCAR/MAR diagnosed via missingness correlation analysis |
| Imputation | MICE for MAR lab cluster, median for MCAR numerics, 'Unknown' for categoricals |
| Encoding | One-hot encoding for categoricals, binary encoding for target |
| Scaling | StandardScaler fit on train only — no data leakage |
| Training | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Evaluation | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix |
| Deployment | Random Forest saved via joblib, served via Streamlit |

---

## Key Findings

- **Class imbalance:** 92% No / 8% Yes — F1-Score used as primary metric
- **Strongest predictors:** Creatinine level, HbA1c, glucose, prior admissions (r=0.68–0.81)
- **High-signal features from EDA:**
  - AMA discharge: 30.6% readmission rate vs Home: 2.4%
  - Emergency admission: 22.1% vs elective: 2.1%
  - No follow-up scheduled: 27.8% vs follow-up arranged: 3.4%
- **Outliers retained:** Clinical extremes (creatinine max=11.82, glucose max=695) 
  represent genuine high-acuity patients — retained deliberately, not data errors
- **Multicollinearity noted:** Prior admissions correlates with lab values (r=0.50–0.53) 
  — tree-based models selected partly for robustness to this

---

## Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree | 1.00 | 0.99 | 0.99 | 0.99 | 0.996 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gradient Boosting | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |

**Selected model: Random Forest** — tied for highest performance, robust to outliers 
and multicollinearity, provides interpretable feature importances.

> ⚠️ Near-perfect scores reflect synthetic data characteristics, not real-world 
> clinical performance. See Dataset note above.

---

## How to Run the Notebook

1. Open `notebook.ipynb` in Google Colab or Jupyter
2. Upload `hospital_readmission_dataset.csv` to the session
3. Run all cells top to bottom (Runtime → Run All)

---

## How to Run the Streamlit App

**Prerequisites:**
```bash
pip install streamlit pandas scikit-learn joblib
```

**Run:**
```bash
cd readmission_app
streamlit run app.py
```

App opens at `http://localhost:8501`

All 3 `.pkl` files must be in the same folder as `app.py`.

---

## Limitations & Future Work

- **Synthetic data:** Real-world validation on genuine EHR data required before 
  any clinical use
- **Gradient Boosting:** No native `class_weight` support — may underperform on 
  minority class with real imbalanced data
- **Threshold tuning:** Default 0.5 classification threshold used — in a clinical 
  setting, lowering the threshold would increase recall at the cost of precision
- **Future:** Validate on prospective patient data, A/B test intervention protocols, 
  expand to 60-day readmission prediction

---

## Tech Stack

Python · scikit-learn · pandas · numpy · matplotlib · seaborn · Streamlit · joblib
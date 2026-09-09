# Student Performance Prediction and Early Warning System using Machine Learning

A complete end-to-end Machine Learning project that predicts whether a student will **Pass or Fail**, estimates their **Final Marks**, and identifies **at-risk students** early using academic and behavioral data.

---

## Project Overview

Educational institutions collect large amounts of student data, but identifying struggling students early remains a challenge. This project solves that problem by building a practical Machine Learning system that:

- Predicts student performance (Pass/Fail)
- Predicts final marks (Regression)
- Classifies students into Risk Levels (Low / Medium / High)
- Provides actionable insights through an Early Warning System
- Offers an interactive web interface using Streamlit

---

## Key Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Classification Models (Pass/Fail Prediction)
- Regression Models (Final Marks Prediction)
- Model Evaluation & Comparison
- Hyperparameter Tuning
- Cross-Validation
- Class Imbalance Handling
- Underfitting / Overfitting Analysis
- Early Warning System (Risk Classification)
- Interactive Streamlit Web App
- Model Persistence (Save & Load)

---

## Dataset

- **Total Records:** 2000 students
- **Features:** Academic + Behavioral

### Features Used:

| Category              | Features                                      |
|-----------------------|-----------------------------------------------|
| Student Information   | Age, Gender, Department, Year                 |
| Academic              | Hours Studied, Attendance, Previous Marks, Assignment Score, Internal Marks, Quiz Score, Practical Score |
| Behavioral            | Study Hours Per Day, Sleep Hours, Class Participation, Internet Usage |
| Engineered Features   | Study Efficiency, Attendance Level, Academic Consistency |
| Target Variables      | `Passed` (Classification), `Final_Marks` (Regression) |

---

## Project Structure
Student-Performance-Prediction/
│
├── data/
│   ├── student_performance_dataset.csv
│   └── cleaned_student_data.csv
│
├── notebooks/
│   └── Student_Performance_Prediction.ipynb
│
├── models/
│   ├── best_student_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── app.py
│
├── images/
│   └── (graphs and app screenshots)
│
├── requirements.txt
└── README.md
text


---

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Saving:** Joblib
- **Web App:** Streamlit
- **Version Control:** Git & GitHub

---

## Machine Learning Models

### Classification (Pass / Fail)

| Model                  | Purpose                     |
|------------------------|-----------------------------|
| Logistic Regression    | Baseline linear model       |
| Decision Tree          | Interpretable tree model    |
| Random Forest          | Ensemble model (Best)       |

### Regression (Final Marks)

| Model                       | Purpose                     |
|-----------------------------|-----------------------------|
| Linear Regression           | Baseline                    |
| Decision Tree Regressor     | Non-linear modeling         |
| Random Forest Regressor     | Best performing regressor   |

---

## Model Evaluation Metrics

### Classification
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Early Warning System

Students are classified into three risk categories based on the predicted probability of passing:

| Pass Probability | Risk Level     |
|------------------|----------------|
| ≥ 75%            | Low Risk       |
| 45% – 74%        | Medium Risk    |
| < 45%            | High Risk      |

This helps teachers and administrators take early action for students who are likely to fail.

---

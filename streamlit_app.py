import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Student Performance Prediction & Early Warning System")
st.markdown("### Predict Pass/Fail + Risk Level using Machine Learning")

# ------------------------------------------------------------------
# Load Model & Scaler
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("best_student_model.pkl")
    scaler = joblib.load("scaler.pkl")  # only needed for Logistic Regression
    return model, scaler

try:
    model, scaler = load_models()
    model_loaded = True
except Exception:
    st.error("Model files not found! Please make sure 'best_student_model.pkl' "
             "and 'scaler.pkl' are in the same folder as this script.")
    model_loaded = False
    st.stop()

# ------------------------------------------------------------------
# Constants — MUST match how the data was preprocessed in the notebook
# ------------------------------------------------------------------
# In the notebook: pd.get_dummies(columns=["Department"], drop_first=True)
# => the FIRST department in alphabetical order (AIML) was dropped and is
#    represented by ALL ZEROS. CSE keeps a flag "Department_CSE".
DEPARTMENTS = ["AIML", "CSE", "Civil", "ECE", "EEE", "IT", "ME"]

# The exact column order the Random Forest was trained on.
# We rely on model.feature_names_in_ at runtime, so this list is only a
# fallback and a way to see the correct Department columns.
EXPECTED_COLUMNS = list(model.feature_names_in_)


def build_feature_row(
    age, gender, year, department,
    hours_studied, attendance, previous_marks, assignment_score,
    internal_marks, quiz_score, practical_score, study_hours_per_day,
    sleep_hours, class_participation, internet_usage,
):
    gender_num = 0 if gender == "Male" else 1
    participation_num = {"Low": 0, "Medium": 1, "High": 2}[class_participation]

    data = {
        "Age": float(age),
        "Gender": gender_num,
        "Year": int(year),
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score,
        "Internal_Marks": internal_marks,
        "Quiz_Score": quiz_score,
        "Practical_Score": practical_score,
        "Study_Hours_Per_Day": study_hours_per_day,
        "Sleep_Hours": sleep_hours,
        "Class_Participation": participation_num,
        "Internet_Usage": internet_usage,
    }

    # ---- Engineered features (same formulas as the notebook) ----
    # NOTE: during training "Study_Efficiency" was computed from
    # Final_Marks, which the model cannot know at prediction time.
    # We approximate it with Previous_Marks here (same as your original).
    data["Study_Efficiency"] = previous_marks / (hours_studied + 1)
    data["Attendance_Level"] = 0 if attendance < 60 else (1 if attendance < 75 else 2)
    data["Academic_Consistency"] = (
        previous_marks + assignment_score + internal_marks + quiz_score
    ) / 4

    # ---- Department one-hot encoding ----
    # FIX: AIML is the reference class (all zeros), NOT CSE.
    for dept in DEPARTMENTS:
        if dept == "AIML":
            continue  # reference column was dropped by drop_first=True
        data[f"Department_{dept}"] = 1 if department == dept else 0

    return data


# ------------------------------------------------------------------
# Input Form
# ------------------------------------------------------------------
st.sidebar.header("Enter Student Details")

age = st.sidebar.slider("Age", 18, 25, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
year = st.sidebar.selectbox("Year", [1, 2, 3, 4])
department = st.sidebar.selectbox("Department", DEPARTMENTS)

hours_studied = st.sidebar.slider("Hours Studied (per week)", 0.5, 17.0, 6.0, 0.1)
attendance = st.sidebar.slider("Attendance (%)", 30.0, 100.0, 75.0, 0.5)
previous_marks = st.sidebar.slider("Previous Marks", 25.0, 98.0, 65.0, 0.5)
assignment_score = st.sidebar.slider("Assignment Score", 20.0, 100.0, 70.0, 0.5)
internal_marks = st.sidebar.slider("Internal Marks", 25.0, 100.0, 68.0, 0.5)
quiz_score = st.sidebar.slider("Quiz Score", 15.0, 100.0, 65.0, 0.5)
practical_score = st.sidebar.slider("Practical Score", 25.0, 100.0, 70.0, 0.5)
study_hours_per_day = st.sidebar.slider("Study Hours Per Day", 0.3, 7.7, 2.5, 0.1)
sleep_hours = st.sidebar.slider("Sleep Hours", 3.0, 10.3, 7.0, 0.1)
class_participation = st.sidebar.selectbox("Class Participation", ["Low", "Medium", "High"])
internet_usage = st.sidebar.slider("Internet Usage (hours/day)", 0.5, 9.3, 3.5, 0.1)

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
if st.sidebar.button("Predict Performance", type="primary"):
    if not model_loaded:
        st.stop()

    data = build_feature_row(
        age, gender, year, department,
        hours_studied, attendance, previous_marks, assignment_score,
        internal_marks, quiz_score, practical_score, study_hours_per_day,
        sleep_hours, class_participation, internet_usage,
    )

    input_df = pd.DataFrame([data])
    # Reorder to the exact columns the model saw during training
    input_df = input_df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    try:
        # Logistic Regression needs scaled input; tree models do not.
        if "LogisticRegression" in str(type(model)):
            probability = model.predict_proba(scaler.transform(input_df))[0][1]
            prediction = (probability >= 0.5).astype(int)
        else:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

        # Risk classification (same thresholds as the notebook)
        if probability >= 0.75:
            risk, risk_color = "Low Risk", "green"
        elif probability >= 0.45:
            risk, risk_color = "Medium Risk", "orange"
        else:
            risk, risk_color = "High Risk", "red"

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", "PASS" if prediction == 1 else "FAIL")
        with col2:
            st.metric("Pass Probability", f"{probability * 100:.1f}%")
        with col3:
            st.markdown("**Risk Level**")
            st.markdown(f"<h3 style='color:{risk_color}'>{risk}</h3>", unsafe_allow_html=True)

        if risk == "High Risk":
            st.warning("This student is at **High Risk** of failing.")
            st.markdown("""
            **Suggested Actions:**
            - Improve attendance immediately
            - Increase daily study hours
            - Complete all assignments on time
            - Seek help from teachers / mentors
            - Reduce excessive internet usage
            """)
        elif risk == "Medium Risk":
            st.info("This student is at **Medium Risk**. Regular monitoring is recommended.")
        else:
            st.success("This student is at **Low Risk**. Performance looks good!")

    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
        st.info("Make sure the feature columns match exactly with the training data.")

else:
    st.info("← Enter student details in the sidebar and click **Predict Performance**")

st.markdown("---")
st.caption("Student Performance Prediction System | Machine Learning Project")
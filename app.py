import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    import joblib
except ImportError:
    joblib = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_student_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

DEPARTMENTS = ["AIML", "CSE", "Civil", "ECE", "EEE", "IT", "ME"]

app = Flask(__name__)

_model = None
_scaler = None


def get_risk_level(probability):
    if probability >= 0.75:
        return "Low Risk"
    if probability >= 0.45:
        return "Medium Risk"
    return "High Risk"


def build_feature_vector(data):
    attendance = float(data["attendance"])
    hours_studied = float(data["hours_studied"])
    previous_marks = float(data["previous_marks"])
    assignment_score = float(data["assignment_score"])
    internal_marks = float(data["internal_marks"])
    quiz_score = float(data["quiz_score"])
    department = data["department"]

    features = {
        "Age": float(data["age"]),
        "Gender": 0 if data["gender"] == "Male" else 1,
        "Year": int(data["year"]),
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score,
        "Internal_Marks": internal_marks,
        "Quiz_Score": quiz_score,
        "Practical_Score": float(data["practical_score"]),
        "Study_Hours_Per_Day": float(data["study_hours_per_day"]),
        "Sleep_Hours": float(data["sleep_hours"]),
        "Class_Participation": {"Low": 0, "Medium": 1, "High": 2}[data["class_participation"]],
        "Internet_Usage": float(data["internet_usage"]),
        "Study_Efficiency": previous_marks / (hours_studied + 1),
        "Attendance_Level": 0 if attendance < 60 else (1 if attendance < 75 else 2),
        "Academic_Consistency": (
            previous_marks + assignment_score + internal_marks + quiz_score
        ) / 4,
    }

    for dept in DEPARTMENTS:
        if dept == "AIML":
            continue
        features[f"Department_{dept}"] = 1 if department == dept else 0

    return features


@app.route("/")
def index():
    return render_template(
        "index.html",
        departments=DEPARTMENTS,
        model_ready=_model is not None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500

    try:
        data = request.get_json(silent=True) or request.form
        features = build_feature_vector(data)
        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(columns=list(_model.feature_names_in_), fill_value=0)

        probability = float(_model.predict_proba(input_df)[0][1])
        prediction = int(_model.predict(input_df)[0])

        return jsonify(
            {
                "ok": True,
                "prediction": "PASS" if prediction == 1 else "FAIL",
                "probability": round(probability * 100, 1),
                "risk_level": get_risk_level(probability),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def load_models():
    global _model, _scaler
    if joblib is None:
        print("joblib not available — model cannot be loaded.")
        return
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        print(f"Loaded model: {type(_model).__name__}")
    else:
        print(f"Model file not found: {MODEL_PATH}")
    if os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
        print("Loaded scaler.")


load_models()

if __name__ == "__main__":
    app.run(debug=True)
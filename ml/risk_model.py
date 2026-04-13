"""ML-based dropout risk prediction model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


# Columns used for risk prediction
FEATURE_COLUMNS = [
    "admission_year", "age", "current_semester", "study_hours", "study_sessions",
    "social_media_hours", "attendance_pct", "previous_sgpa", "current_cgpa",
    "credits_completed", "skill_dev_hours", "family_income",
    "scholarship_enc", "probation_enc", "suspension_enc", "consultancy_enc",
    "cocurricular_enc", "health_issues_enc", "transportation_enc",
]

LABEL_ENCODERS = {}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features from raw dataframe."""
    encodings = {
        "scholarship": ("Do you have meritorious scholarship ?", ["Yes", "No"]),
        "probation": ("Did you ever fall in probation?", ["No", "Yes"]),
        "suspension": ("Did you ever got suspension?", ["No", "Yes"]),
        "consultancy": ("Do you attend in teacher consultancy for any kind of academical problems?", ["No", "Yes"]),
        "cocurricular": ("Are you engaged with any co-curriculum activities?", ["No", "Yes"]),
        "health_issues": ("Do you have any health issues?", ["No", "Yes"]),
        "transportation": ("Do you use University transportation?", ["No", "Yes"]),
    }

    col_map = {
        "University Admission year": "admission_year",
        "Age": "age",
        "Current Semester": "current_semester",
        "How many hour do you study daily?": "study_hours",
        "How many times do you seat for study in a day?": "study_sessions",
        "How many hour do you spent daily in social media?": "social_media_hours",
        "What was your previous SGPA?": "previous_sgpa",
        "What is your current CGPA?": "current_cgpa",
        "How many Credit did you have completed?": "credits_completed",
        "How many hour do you spent daily on your skill development?": "skill_dev_hours",
        "What is your monthly family income?": "family_income",
    }

    X = df.copy()
    for new_col, (orig_col, _) in encodings.items():
        orig = [c for c in df.columns if orig_col in c or orig_col.replace("?", "").strip() in c.replace("?", "").strip()]
        if orig:
            col = orig[0]
            X[f"{new_col}_enc"] = (X[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Attendance
    def parse_att(x):
        if pd.isna(x):
            return 0.0
        s = str(x).replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    if "Average attendance on class" in X.columns:
        X["attendance_pct"] = X["Average attendance on class"].apply(parse_att)
    else:
        X["attendance_pct"] = 80.0

    for old, new in col_map.items():
        if old in X.columns:
            X = X.rename(columns={old: new})

    return X


def create_target(df: pd.DataFrame) -> np.ndarray:
    """Create synthetic at-risk target: CGPA<2.5 OR attendance<70 OR probation/suspension."""
    cgpa = df["What is your current CGPA?"].fillna(3.0) if "What is your current CGPA?" in df.columns else df.get("current_cgpa", pd.Series([3.0] * len(df)))
    def parse_att(x):
        if pd.isna(x): return 70.0
        s = str(x).replace("%", "").strip()
        try: return float(s)
        except: return 70.0
    att_col = "Average attendance on class" if "Average attendance on class" in df.columns else "attendance"
    att = df[att_col].apply(parse_att) if att_col in df.columns else pd.Series([80.0] * len(df))
    prob = df["Did you ever fall in probation?"].astype(str).str.lower().str.contains("yes", na=False) if "Did you ever fall in probation?" in df.columns else False
    susp = df["Did you ever got suspension?"].astype(str).str.lower().str.contains("yes", na=False) if "Did you ever got suspension?" in df.columns else False
    return ((cgpa < 2.5) | (att < 70) | prob | susp).astype(int).values


def train_model(excel_path: str, model_path: str = "ml_models/risk_model.joblib"):
    """Train and save the risk prediction model."""
    df = pd.read_excel(excel_path)
    X_df = prepare_features(df)
    y = create_target(df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
    X = X_df[feature_cols].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.2%}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    return model, feature_cols


def _raw_to_features(raw: dict) -> dict:
    """Map raw_data dict to model feature columns. Handles both original and renamed keys."""
    yes_no = lambda v: 1 if str(v).strip().lower() == "yes" else 0
    def parse_att(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return 80.0
        s = str(x).replace("%", "").strip()
        try: return float(s)
        except: return 80.0
    def safe_float(v):
        if v is None: return 0.0
        try: return float(v)
        except: return 0.0
    # Maps: (original_excel_key, renamed_key) -> model_col
    col_map = [
        (["University Admission year", "admission_year"], "admission_year"),
        (["Age", "age"], "age"),
        (["Current Semester", "current_semester"], "current_semester"),
        (["How many hour do you study daily?", "study_hours"], "study_hours"),
        (["How many times do you seat for study in a day?", "study_sessions"], "study_sessions"),
        (["How many hour do you spent daily in social media?", "social_media_hours"], "social_media_hours"),
        (["Average attendance on class", "attendance"], "attendance_pct"),
        (["What was your previous SGPA?", "previous_sgpa"], "previous_sgpa"),
        (["What is your current CGPA?", "current_cgpa"], "current_cgpa"),
        (["How many Credit did you have completed?", "credits_completed"], "credits_completed"),
        (["How many hour do you spent daily on your skill development?", "skill_dev_hours"], "skill_dev_hours"),
        (["What is your monthly family income?", "family_income"], "family_income"),
    ]
    enc_cols = [
        (["Do you have meritorious scholarship ?", "scholarship"], "scholarship_enc"),
        (["Did you ever fall in probation?", "probation"], "probation_enc"),
        (["Did you ever got suspension?", "suspension"], "suspension_enc"),
        (["Do you attend in teacher consultancy for any kind of academical problems?", "teacher_consultancy"], "consultancy_enc"),
        (["Are you engaged with any co-curriculum activities?", "cocurricular"], "cocurricular_enc"),
        (["Do you have any health issues?", "health_issues"], "health_issues_enc"),
        (["Do you use University transportation?", "transportation"], "transportation_enc"),
    ]
    out = {}
    for keys, new in col_map:
        v = next((raw.get(k) for k in keys if k in raw), 0)
        if new == "attendance_pct":
            out[new] = parse_att(v)
        else:
            out[new] = safe_float(v)
    for keys, new in enc_cols:
        v = next((raw.get(k) for k in keys if k in raw), "No")
        out[new] = yes_no(v)
    return out


def predict_risk(student_data: dict, model_path: str = "ml_models/risk_model.joblib") -> tuple[float, str]:
    """Predict risk score (0-100) and level for a student."""
    path = Path(model_path)
    if not path.exists():
        return 25.0, "low"

    data = joblib.load(path)
    model = data["model"]
    feature_cols = data["feature_cols"]

    raw = student_data.get("raw_data", student_data)
    row = _raw_to_features(raw)
    for col in feature_cols:
        if col not in row:
            row[col] = student_data.get(col, 0)

    X = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])[feature_cols].fillna(0)
    proba = model.predict_proba(X)[0]
    risk_score = float(proba[1] * 100) if len(proba) > 1 else 25.0

    if risk_score >= 75: level = "critical"
    elif risk_score >= 50: level = "high"
    elif risk_score >= 25: level = "medium"
    else: level = "low"

    return round(risk_score, 1), level

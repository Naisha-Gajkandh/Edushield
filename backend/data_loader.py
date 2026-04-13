"""Load student performance data from Excel and populate database."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _safe_int(x, default=0):
    """Convert to int, handling NaN and None."""
    if pd.isna(x) or x is None:
        return default
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default


def _safe_float(x, default=0.0):
    """Convert to float, handling NaN and None."""
    if pd.isna(x) or x is None:
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def parse_attendance(x):
    """Parse attendance value (e.g. 90, '90%')."""
    if pd.isna(x):
        return 0.0
    s = str(x).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_and_transform_data(excel_path: str) -> pd.DataFrame:
    """Load Excel data and transform for risk scoring."""
    df = pd.read_excel(excel_path)

    # Rename columns for easier access
    col_map = {
        "University Admission year": "admission_year",
        "Gender": "gender",
        "Age": "age",
        "H.S.C passing year": "hsc_year",
        "Program": "program",
        "Current Semester": "current_semester",
        "Do you have meritorious scholarship ?": "scholarship",
        "Do you use University transportation?": "transportation",
        "How many hour do you study daily?": "study_hours",
        "How many times do you seat for study in a day?": "study_sessions",
        "What is your preferable learning mode?": "learning_mode",
        "Do you use smart phone?": "smartphone",
        "Do you have personal Computer?": "personal_computer",
        "How many hour do you spent daily in social media?": "social_media_hours",
        "Status of your English language proficiency": "english_proficiency",
        "Average attendance on class": "attendance",
        "Did you ever fall in probation?": "probation",
        "Did you ever got suspension?": "suspension",
        "Do you attend in teacher consultancy for any kind of academical problems?": "teacher_consultancy",
        "What are the skills do you have ?": "skills",
        "How many hour do you spent daily on your skill development?": "skill_dev_hours",
        "What is you interested area?": "interested_area",
        "What is your relationship status?": "relationship_status",
        "Are you engaged with any co-curriculum activities?": "cocurricular",
        "With whom you are living with?": "living_with",
        "Do you have any health issues?": "health_issues",
        "What was your previous SGPA?": "previous_sgpa",
        "Do you have any physical disabilities?": "physical_disability",
        "What is your current CGPA?": "current_cgpa",
        "How many Credit did you have completed?": "credits_completed",
        "What is your monthly family income?": "family_income",
    }
    df = df.rename(columns=col_map)

    df["attendance_pct"] = df["attendance"].apply(parse_attendance)

    return df


def create_student_dict(row: pd.Series, index: int) -> dict:
    """Convert a dataframe row to student dict for API/database."""
    drop_cols = [c for c in ["attendance_pct"] if c in row.index]
    raw = row.drop(drop_cols).to_dict() if drop_cols else row.to_dict()
    att = row.get("attendance_pct", None)
    if pd.isna(att):
        att = parse_attendance(row.get("attendance", 0))
    return {
        "student_id": f"STU{1000 + index:05d}",
        "name": f"Student {1000 + index}",
        "email": f"student{1000 + index}@university.edu",
        "gender": str(row.get("gender", "")) if pd.notna(row.get("gender")) else "",
        "age": _safe_int(row.get("age"), 0),
        "program": str(row.get("program", "")) if pd.notna(row.get("program")) else "",
        "current_semester": _safe_int(row.get("current_semester"), 0),
        "admission_year": _safe_int(row.get("admission_year"), 0),
        "current_cgpa": _safe_float(row.get("current_cgpa"), 0),
        "previous_sgpa": _safe_float(row.get("previous_sgpa"), 0),
        "credits_completed": _safe_int(row.get("credits_completed"), 0),
        "attendance": _safe_float(att, 0),
        "family_income": _safe_int(row.get("family_income"), 0),
        "raw_data": raw,
    }


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    excel_path = base / "Students_Performance_data_set.xlsx"
    df = load_and_transform_data(str(excel_path))
    print(f"Loaded {len(df)} records")
    print(df.columns.tolist())
    print(df.head(2))

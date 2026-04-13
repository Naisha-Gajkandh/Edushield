"""Add 6 subject columns with varied marks to the student performance dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = ROOT / "Students_Performance_data_set.xlsx"

# Credits per course (Creativity has fewer credits)
SUBJECT_CREDITS = {
    "Python": 4,
    "Java": 4,
    "Data Science": 4,
    "DBMS": 4,
    "Digital Electronics": 3,
    "Creativity and Creation": 2,
}

SUBJECTS = list(SUBJECT_CREDITS.keys())


def add_subject_marks(df: pd.DataFrame) -> pd.DataFrame:
    """CGPA <= 2: below 60 in 2-3 subjects, ~20 in one, high in 1 only. CGPA 3-4: above 75 in many, below 50 in 1."""
    cgpa_col = "What is your current CGPA?"
    if cgpa_col not in df.columns:
        cgpa_col = [c for c in df.columns if "CGPA" in c][0]
    cgpa = df[cgpa_col].fillna(2.5).values

    for subj in SUBJECTS:
        df[subj] = 0

    for j in range(len(df)):
        c = float(cgpa[j])
        np.random.seed(j * 47 + 11)

        if c <= 2.0:
            # CGPA <= 2: below 60 in 2-3 subjects, ~20 in one, high (75+) in only one
            strong_subj = j % 6
            weak_subj = (j + 2) % 6
            low_subjs = [i for i in range(6) if i != strong_subj and i != weak_subj]
            np.random.shuffle(low_subjs)
            low_subjs = low_subjs[:2]

            for i, subj in enumerate(SUBJECTS):
                np.random.seed(j * 17 + i * 31 + 7)
                if i == strong_subj:
                    mark = 75 + np.random.uniform(0, 18)
                elif i == weak_subj:
                    mark = 18 + np.random.uniform(0, 8)
                elif i in low_subjs:
                    mark = 42 + np.random.uniform(0, 16)
                else:
                    mark = 55 + np.random.uniform(0, 5)
                df.loc[df.index[j], subj] = int(np.clip(round(mark), 18, 95))

        elif c < 2.5:
            strong_subj = j % 6
            weak_subj = (j + 3) % 6
            low_subjs = [i for i in range(6) if i != strong_subj and i != weak_subj]
            np.random.shuffle(low_subjs)
            low_subjs = low_subjs[:2]

            for i, subj in enumerate(SUBJECTS):
                np.random.seed(j * 17 + i * 31 + 7)
                if i == strong_subj:
                    mark = 78 + np.random.uniform(0, 15)
                elif i == weak_subj:
                    mark = 22 + np.random.uniform(0, 12)
                elif i in low_subjs:
                    mark = 48 + np.random.uniform(0, 12)
                else:
                    mark = 58 + np.random.uniform(0, 8)
                df.loc[df.index[j], subj] = int(np.clip(round(mark), 20, 95))

        elif c >= 3.0 and c <= 4.0:
            # CGPA 3-4: above 75 in many (4-5), below 50 in one subject
            weak_subj = j % 6
            for i, subj in enumerate(SUBJECTS):
                np.random.seed(j * 17 + i * 31 + 7)
                if i == weak_subj:
                    mark = 38 + np.random.uniform(0, 12)
                else:
                    mark = 76 + np.random.uniform(0, 18)
                df.loc[df.index[j], subj] = int(np.clip(round(mark), 35, 98))

        else:
            # CGPA 2.5-3: transitional
            weak_subj = j % 6
            for i, subj in enumerate(SUBJECTS):
                np.random.seed(j * 17 + i * 31 + 7)
                if i == weak_subj:
                    mark = 45 + np.random.uniform(0, 15)
                else:
                    mark = 65 + np.random.uniform(0, 18)
                df.loc[df.index[j], subj] = int(np.clip(round(mark), 42, 95))

    return df


def add_credits_row(df: pd.DataFrame) -> pd.DataFrame:
    """Add a metadata row or store credits. We'll store in a separate sheet."""
    return df


def main():
    df = pd.read_excel(EXCEL_PATH)
    df = add_subject_marks(df)

    # Save - write to same file, backup first
    backup = ROOT / "Students_Performance_data_set_backup.xlsx"
    if not backup.exists():
        import shutil
        shutil.copy(EXCEL_PATH, backup)
        print(f"Backup saved to {backup.name}")

    df.to_excel(EXCEL_PATH, index=False)
    print(f"Added columns: {SUBJECTS}")
    print(f"Credits: {SUBJECT_CREDITS}")
    print(f"Sample marks (first 3 rows):")
    print(df[SUBJECTS].head(3).to_string())


if __name__ == "__main__":
    main()

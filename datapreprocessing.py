# data_preprocessing.py
# Loads and cleans the student performance dataset
# Variables: Hours_Studied, Attendance, Parental_Involvement,
#            Access_to_Resources, Extracurricular_Activities, Sleep_Hours,
#            Previous_Scores, Motivation_Level, Internet_Access,
#            Tutoring_Sessions, Family_Income, Teacher_Quality, School_Type,
#            Peer_Influence, Physical_Activity, Learning_Disabilities,
#            Parental_Education_Level, Distance_from_Home, Gender, Exam_Score

import pandas as pd
import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Numeric columns
NUM_COLS = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions",
    "Physical_Activity", "Exam_Score",
]

# Categorical columns
CAT_COLS = [
    "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Motivation_Level",
    "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]

# Features used for regression
FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()
    # cap Exam_Score at 100
    df["Exam_Score"] = df["Exam_Score"].clip(upper=100)
    df = df.reset_index(drop=True)
    return df


def get_clean_data() -> pd.DataFrame:
    return clean_data(load_data())


if __name__ == "__main__":
    df = get_clean_data()
    print(f"Shape: {df.shape}")
    print(df.dtypes)
    print(df.head(3))

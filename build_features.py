# builds the feature set from the real Kaggle sleep/lifestyle dataset.
# 374 real people, sourced from:
# https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
#
# what comes from the real data:
#   sleep_hours        <- Sleep Duration
#   sleep_quality_score <- Quality of Sleep
#   physical_activity_minutes <- Physical Activity Level
#   stress_level       <- Stress Level
#   age                <- Age
#   occupation_type    <- Occupation (all map to "worker" in this dataset -
#                         no students, athletes or unemployed in the source)
#
# what's still simulated (these columns don't exist in any public version
# of this dataset - documented here so it's never a surprise):
#   screen_time_hours   - generated based on activity/sleep quality
#   caffeine_intake_mg  - generated based on stress level
#   diet_quality_score  - generated based on activity/sleep quality
#
# target (also derived, not measured):
#   cognitive_performance_score - weighted formula, see build_target_score()

import os
import numpy as np
import pandas as pd

np.random.seed(7)

JOB_TO_CATEGORY = {
    "Software Engineer": "worker", "Engineer": "worker", "Doctor": "worker",
    "Nurse": "worker", "Teacher": "worker", "Lawyer": "worker",
    "Accountant": "worker", "Salesperson": "worker", "Scientist": "worker",
    "Manager": "worker", "Freelancer": "worker",
    "Sales Representative": "worker",
    "Student": "student", "Athlete": "athlete",
    "Unemployed": "unemployed", "Retired": "unemployed",
}


def load_raw_data():
    real_path = "data/sleep_data_real.csv"
    synth_path = "data/sleep_data_synthetic.csv"
    if os.path.exists(real_path):
        print("Using the REAL Kaggle dataset (374 rows)")
        return pd.read_csv(real_path), "real"
    print("Real dataset not found - falling back to synthetic")
    return pd.read_csv(synth_path), "synthetic"


def build_features(raw, source):
    df = pd.DataFrame()
    df["age"] = raw["Age"]
    df["stress_level"] = raw["Stress Level"]
    df["sleep_quality_score"] = raw["Quality of Sleep"]
    df["physical_activity_minutes"] = raw["Physical Activity Level"]
    df["sleep_hours"] = raw["Sleep Duration"]
    df["occupation_type"] = raw["Occupation"].map(JOB_TO_CATEGORY).fillna("worker")
    df["_source"] = source

    # simulated columns - not in the real dataset, generated from real columns
    screen_base = 9 - 0.3 * (df["sleep_quality_score"] - 5) - 0.02 * (df["physical_activity_minutes"] - 50)
    df["screen_time_hours"] = np.clip(screen_base + np.random.normal(0, 1.5, len(df)), 0, 16).round(1)

    caffeine_base = 80 + 25 * (df["stress_level"] - 5)
    df["caffeine_intake_mg"] = np.clip(caffeine_base + np.random.normal(0, 60, len(df)), 0, 600).round()

    diet_base = 5 + 0.3 * (df["physical_activity_minutes"] / 30) + 0.2 * (df["sleep_quality_score"] - 5)
    df["diet_quality_score"] = np.clip(diet_base + np.random.normal(0, 1.2, len(df)), 1, 10).round(1)

    df["cognitive_performance_score"] = build_target_score(df)
    return df


def build_target_score(df):
    sleep_duration_penalty = -1.2 * (df["sleep_hours"] - 7.5) ** 2
    score = (
        50
        + 2.8 * (df["sleep_quality_score"] - 5)
        + sleep_duration_penalty
        - 1.8 * (df["stress_level"] - 5)
        + 1.8 * (df["diet_quality_score"] - 5)
        + 0.06 * (df["physical_activity_minutes"] - 60)
        - 1.0 * np.clip(df["screen_time_hours"] - 6, 0, None)
        - 0.012 * np.clip(df["caffeine_intake_mg"] - 400, 0, None)
        + np.random.normal(0, 4.5, len(df))
    )
    return np.clip(score, 0, 100).round(1)


if __name__ == "__main__":
    raw, source = load_raw_data()
    df = build_features(raw, source)
    df.drop(columns=["_source"], inplace=True)
    df.to_csv("data/features_raw.csv", index=False)
    print(df.head())
    print(f"\nshape: {df.shape}")
    print(f"\noccupation breakdown:\n{df['occupation_type'].value_counts()}")
    print(f"\ntarget stats:\n{df['cognitive_performance_score'].describe()}")

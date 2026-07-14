# 03_clean_and_validate.py

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from enum import Enum

class Occupation(str, Enum):
    student = "student"
    worker = "worker"
    athlete = "athlete"
    unemployed = "unemployed"

class Record(BaseModel):
    sleep_hours: float = Field(ge=3, le=12)
    screen_time_hours: float = Field(ge=0, le=16)
    physical_activity_minutes: float = Field(ge=0, le=180)
    stress_level: int = Field(ge=1, le=10)
    diet_quality_score: float = Field(ge=1, le=10)
    caffeine_intake_mg: float = Field(ge=0, le=600)
    sleep_quality_score: float = Field(ge=1, le=10)
    age: int = Field(ge=18, le=60)
    occupation_type: Occupation
    cognitive_performance_score: float = Field(ge=0, le=100)
    class Config:
        extra = "forbid"

BOUNDS = {
    "sleep_hours": (3, 12), "screen_time_hours": (0, 16),
    "physical_activity_minutes": (0, 180), "stress_level": (1, 10),
    "diet_quality_score": (1, 10), "caffeine_intake_mg": (0, 600),
    "sleep_quality_score": (1, 10), "age": (18, 60),
}

def validate_rows(df):
    good_rows = []
    bad_count = 0
    for _, row in df.iterrows():
        try:
            checked = Record(**row.to_dict())
            good_rows.append(checked.model_dump())
        except ValidationError:
            bad_count += 1
    print(f"validated {len(df)} rows, kept {len(good_rows)}, rejected {bad_count}")
    return pd.DataFrame(good_rows)

def clean_data(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"removed {before - len(df)} duplicate rows")
    for col in BOUNDS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col, (lo, hi) in BOUNDS.items():
        df[col] = df[col].clip(lo, hi)
    return df

def check_bias(df):
    print("\n--- bias check ---")
    age_groups = pd.cut(df["age"], bins=[18, 28, 38, 48, 60], include_lowest=True)
    print("age distribution:")
    print(age_groups.value_counts(normalize=True).round(3))
    occ_dist = df["occupation_type"].value_counts(normalize=True)
    print(f"\noccupation distribution:")
    print(occ_dist.round(3))
    if occ_dist.max() > 0.6:
        print(f"-> WARNING: '{occ_dist.idxmax()}' is over 60% - real Kaggle dataset only has 'worker' type occupations")
    corrs = df[[c for c in BOUNDS] + ["cognitive_performance_score"]].corr()["cognitive_performance_score"]
    corrs = corrs.drop("cognitive_performance_score").sort_values(key=abs, ascending=False)
    print(f"\nfeature correlations with target:")
    print(corrs.round(3))
    age_corr = corrs.get("age", 0)
    print(f"\nage correlation: {age_corr:.3f} -", "fine" if abs(age_corr) < 0.3 else "elevated, check this")

if __name__ == "__main__":
    df = pd.read_csv("data/features_raw.csv")
    df = clean_data(df)
    df = validate_rows(df)
    check_bias(df)
    df.to_csv("data/cpis_clean.csv", index=False)
    print(f"\nsaved -> data/cpis_clean.csv ({len(df)} rows)")

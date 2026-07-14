# 04_train_model.py

import joblib, json, time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUMERIC_COLS = ["sleep_hours", "screen_time_hours", "physical_activity_minutes",
                 "stress_level", "diet_quality_score", "caffeine_intake_mg",
                 "sleep_quality_score", "age"]
CATEGORICAL_COLS = ["occupation_type"]
TARGET = "cognitive_performance_score"

def make_pipeline(model):
    preprocessing = ColumnTransformer([
        ("scale", StandardScaler(), NUMERIC_COLS),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", model)])

def evaluate(pipeline, X_train, y_train, X_test, y_test):
    start = time.time()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return {
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "test_mae": round(mean_absolute_error(y_test, preds), 3),
        "test_rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 3),
        "test_r2": round(r2_score(y_test, preds), 4),
        "seconds": round(time.time() - start, 2),
    }

if __name__ == "__main__":
    df = pd.read_csv("data/cpis_clean.csv")
    print(f"training on {len(df)} rows (real Kaggle data)")
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear_regression": make_pipeline(LinearRegression()),
        "random_forest": make_pipeline(RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1)),
        "gradient_boosting": make_pipeline(GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)),
    }

    results = {}
    fitted = {}
    for name, pipe in models.items():
        print(f"training {name}...")
        results[name] = evaluate(pipe, X_train, y_train, X_test, y_test)
        fitted[name] = pipe
        r = results[name]
        print(f"  cv r2={r['cv_r2_mean']} (+/-{r['cv_r2_std']})  test r2={r['test_r2']}  mae={r['test_mae']}  rmse={r['test_rmse']}")

    best_name = "random_forest"
    best_by_cv = max(results, key=lambda n: results[n]["cv_r2_mean"])
    if best_by_cv != best_name:
        gap = results[best_by_cv]["cv_r2_mean"] - results[best_name]["cv_r2_mean"]
        print(f"\nnote: {best_by_cv} scored higher by {gap:.4f} r2, but keeping random_forest as planned")

    joblib.dump(fitted[best_name], "models/cpis_model.pkl")
    with open("models/model_results.json", "w") as f:
        json.dump({"chosen_model": best_name, "all_results": results, "trained_on": "real_kaggle_data"}, f, indent=2)
    print(f"\nsaved random_forest -> models/cpis_model.pkl")

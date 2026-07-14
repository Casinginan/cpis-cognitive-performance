import json, joblib, numpy as np, pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERIC_COLS = ["sleep_hours", "screen_time_hours", "physical_activity_minutes",
                 "stress_level", "diet_quality_score", "caffeine_intake_mg",
                 "sleep_quality_score", "age"]
CATEGORICAL_COLS = ["occupation_type"]

NICE_NAMES = {
    "sleep_hours": "Sleep duration", "screen_time_hours": "Screen time",
    "physical_activity_minutes": "Physical activity", "stress_level": "Stress level",
    "diet_quality_score": "Diet quality", "caffeine_intake_mg": "Caffeine intake",
    "sleep_quality_score": "Sleep quality", "age": "Age",
}

def nice_name(raw_name):
    for prefix in ("scale__", "onehot__"):
        if raw_name.startswith(prefix):
            raw_name = raw_name[len(prefix):]
    for key, label in NICE_NAMES.items():
        if raw_name.startswith(key):
            return label
    if raw_name.startswith("occupation_type_"):
        return f"Occupation ({raw_name.replace('occupation_type_', '')})"
    return raw_name.replace("_", " ").title()

def get_feature_names(pipeline):
    return list(pipeline.named_steps["preprocessing"].get_feature_names_out())

def plot_feature_importance(pipeline):
    model = pipeline.named_steps["model"]
    names = [nice_name(n) for n in get_feature_names(pipeline)]
    importances = model.feature_importances_
    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    names_sorted, vals_sorted = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names_sorted[::-1], vals_sorted[::-1], color="#5C7A52")
    ax.set_xlabel("importance")
    ax.set_title("Feature Importance (Random Forest)")
    fig.tight_layout()
    fig.savefig("reports/figures/feature_importance.png", dpi=130)
    plt.close(fig)
    return dict(pairs)

def get_shap_values(pipeline, X):
    model = pipeline.named_steps["model"]
    preprocessing = pipeline.named_steps["preprocessing"]
    X_transformed = preprocessing.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    return explainer, shap_values

def plot_shap_summary(pipeline, X_sample):
    explainer, shap_values = get_shap_values(pipeline, X_sample)
    names = [nice_name(n) for n in get_feature_names(pipeline)]
    mean_abs = np.abs(shap_values).mean(axis=0)
    pairs = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)
    names_sorted, vals_sorted = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names_sorted[::-1], vals_sorted[::-1], color="#B85C5C")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP Feature Impact")
    fig.tight_layout()
    fig.savefig("reports/figures/shap_summary.png", dpi=130)
    plt.close(fig)
    return dict(pairs)

def explain_one_prediction(pipeline, row):
    explainer, shap_values = get_shap_values(pipeline, row)
    names = [nice_name(n) for n in get_feature_names(pipeline)]
    row_shap = shap_values[0]
    total = np.abs(row_shap).sum()
    order = np.argsort(-np.abs(row_shap))
    explanations = []
    contributions = {}
    for i in order:
        pct = abs(row_shap[i]) / total * 100 if total > 0 else 0
        direction = "increases" if row_shap[i] > 0 else "reduces"
        sign = "+" if row_shap[i] > 0 else "-"
        explanations.append(f"{names[i]} contributes {sign}{pct:.0f}% to the score ({direction} it)")
        contributions[names[i]] = round(float(row_shap[i]), 3)
    return explanations[:5], contributions

if __name__ == "__main__":
    df = pd.read_csv("data/cpis_clean.csv")
    pipeline = joblib.load("models/cpis_model.pkl")
    sample = df.sample(n=min(200, len(df)), random_state=42)
    X_sample = sample[NUMERIC_COLS + CATEGORICAL_COLS]

    print("computing feature importance...")
    importance = plot_feature_importance(pipeline)
    for name, val in list(importance.items())[:5]:
        print(f"  {name}: {val:.3f}")

    print("\ncomputing SHAP values...")
    shap_importance = plot_shap_summary(pipeline, X_sample)

    print("\nexample explanation:")
    explanations, _ = explain_one_prediction(pipeline, X_sample.iloc[[0]])
    for line in explanations:
        print(f"  - {line}")

    with open("reports/explainability.json", "w") as f:
        json.dump({"feature_importance": importance, "shap_importance": shap_importance}, f, indent=2)
    print("\nsaved charts + reports/explainability.json")

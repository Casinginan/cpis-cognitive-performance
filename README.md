# Cognitive Performance Intelligence System (CPIS)

ML project that predicts a cognitive performance score (0-100) from real
sleep and lifestyle data. Built as a portfolio project demonstrating a full
data science pipeline - not just "train a model and call it done."

**Dataset:** Sleep Health and Lifestyle Dataset, Kaggle
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
374 real people, self-reported sleep/lifestyle survey data.

> Disclaimer: predictive insight tool only, not a medical diagnosis.

---

## Honest notes about the data

Columns from the real dataset:
- sleep_hours, sleep_quality_score, physical_activity_minutes,
  stress_level, age, occupation_type

Columns that don't exist in the dataset and were simulated:
- screen_time_hours, caffeine_intake_mg, diet_quality_score
  (generated using correlated formulas - documented in 02_build_features.py)

Target (also derived, not measured):
- cognitive_performance_score - a weighted proxy formula, not a real
  psychometric test result. Formula is in 02_build_features.py.

One known bias: the real Kaggle dataset only has working professionals
(nurses, doctors, engineers etc.) - no students, athletes, or unemployed
people. The model is flagged for this in the bias checker.

---

## How to run

```
pip install -r requirements.txt
```

Run in order (each script saves output for the next one):
```
python 01_get_data.py
python 02_build_features.py
python 03_clean_and_validate.py
python 04_train_model.py
python 05_stats_and_charts.py
python 06_explainability.py
```

Then either:
```
streamlit run app/dashboard.py
```
or:
```
cd api && uvicorn main:app --reload
```
(API docs at http://localhost:8000/docs)

---

## Dashboard tabs

1. The Population  - charts of the real sample first (pie, bar, histograms)
2. New Entry       - predict your score + see where you land vs the sample
3. Model Logic     - feature importance, SHAP, model comparison table
4. Sample & Bias   - raw data, correlation matrix, bias check
5. Project Plan    - Gantt chart of how this project was built

---

## What's in the pipeline

- Pydantic validation - rejects out-of-range or extra fields (no PII)
- Bias detection - age balance, occupation balance, feature dominance check
- 3 models compared - Linear Regression, Random Forest, Gradient Boosting
- All inside one sklearn Pipeline - scaling/encoding travels with the model
- 5-fold cross validation + 80/20 train/test split
- SHAP explainability - per-prediction plain-english explanations
- FastAPI wrapper - /predict and /health endpoints

# api/main.py
# run with: cd api && uvicorn main:app --reload

import os, logging, joblib
import numpy as np
import pandas as pd
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("cpis")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cpis_model.pkl")
NUMERIC_COLS = ["sleep_hours", "screen_time_hours", "physical_activity_minutes",
                 "stress_level", "diet_quality_score", "caffeine_intake_mg",
                 "sleep_quality_score", "age"]
CATEGORICAL_COLS = ["occupation_type"]

app = FastAPI(title="CPIS API")
model = None

class Occupation(str, Enum):
    student = "student"
    worker = "worker"
    athlete = "athlete"
    unemployed = "unemployed"

class PredictRequest(BaseModel):
    sleep_hours: float = Field(ge=3, le=12)
    screen_time_hours: float = Field(ge=0, le=16)
    physical_activity_minutes: float = Field(ge=0, le=180)
    stress_level: int = Field(ge=1, le=10)
    diet_quality_score: float = Field(ge=1, le=10)
    caffeine_intake_mg: float = Field(ge=0, le=600)
    sleep_quality_score: float = Field(ge=1, le=10)
    age: int = Field(ge=18, le=60)
    occupation_type: Occupation
    class Config:
        extra = "forbid"

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        log.info("model loaded")
    else:
        log.warning("no model found - run the training scripts first")

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "no model loaded"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    row = pd.DataFrame([req.model_dump()])
    score = float(np.clip(model.predict(row)[0], 0, 100))
    log.info(f"prediction: {score:.1f}")
    return {
        "cognitive_performance_score": round(score, 1),
        "disclaimer": "Predictive insight only - not a medical diagnosis."
    }

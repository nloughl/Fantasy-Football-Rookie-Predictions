"""
FastAPI application for rookie fantasy football predictions.

Serves the trained Random Forest model via REST endpoints.

Usage:
    uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/model.joblib"

# Feature columns in the order the model expects
CATEGORICAL_COLS = ["position", "team", "C_conference", "C_team"]
NUMERIC_COLS = [
    "C_season", "C_passing_TD", "C_passing_YDS", "C_passing_INT",
    "C_rushing_TD", "C_rushing_YDS", "C_receiving_REC", "C_receiving_TD",
    "C_receiving_YDS", "C_fumbles_LOST", "C_passing_ATT", "C_passing_COMPLETIONS",
    "C_passing_PCT", "C_passing_YPA", "C_rushing_CAR", "C_rushing_YPC",
    "C_rushing_LONG", "C_receiving_YPR", "C_receiving_LONG", "C_fumbles_FUM",
    "C_conference_strength",
]
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


# --- Pydantic schemas ---

class RookieInput(BaseModel):
    """Input schema for a single rookie prediction."""
    player_name: str
    position: str
    team: str
    C_conference: str
    C_team: str
    C_season: float
    C_passing_TD: float = 0.0
    C_passing_YDS: float = 0.0
    C_passing_INT: float = 0.0
    C_rushing_TD: float = 0.0
    C_rushing_YDS: float = 0.0
    C_receiving_REC: float = 0.0
    C_receiving_TD: float = 0.0
    C_receiving_YDS: float = 0.0
    C_fumbles_LOST: float = 0.0
    C_passing_ATT: float = 0.0
    C_passing_COMPLETIONS: float = 0.0
    C_passing_PCT: float = 0.0
    C_passing_YPA: float = 0.0
    C_rushing_CAR: float = 0.0
    C_rushing_YPC: float = 0.0
    C_rushing_LONG: float = 0.0
    C_receiving_YPR: float = 0.0
    C_receiving_LONG: float = 0.0
    C_fumbles_FUM: float = 0.0
    C_conference_strength: float = 5.0

    model_config = {"json_schema_extra": {
        "examples": [{
            "player_name": "Cam Ward",
            "position": "QB",
            "team": "TEN",
            "C_conference": "ACC",
            "C_team": "Miami (FL)",
            "C_season": 2024,
            "C_passing_TD": 36.0,
            "C_passing_YDS": 4313.0,
            "C_passing_INT": 7.0,
            "C_rushing_TD": 3.0,
            "C_rushing_YDS": 152.0,
            "C_receiving_REC": 0.0,
            "C_receiving_TD": 0.0,
            "C_receiving_YDS": 0.0,
            "C_fumbles_LOST": 2.0,
            "C_passing_ATT": 493.0,
            "C_passing_COMPLETIONS": 328.0,
            "C_passing_PCT": 66.5,
            "C_passing_YPA": 8.7,
            "C_rushing_CAR": 89.0,
            "C_rushing_YPC": 1.7,
            "C_rushing_LONG": 25.0,
            "C_receiving_YPR": 0.0,
            "C_receiving_LONG": 0.0,
            "C_fumbles_FUM": 4.0,
            "C_conference_strength": 8.0,
        }]
    }}


class PredictionOutput(BaseModel):
    player_name: str
    position: str
    predicted_fantasy_points: float


# --- App setup ---

app = FastAPI(
    title="Rookie Fantasy Football Predictor",
    description="Predict rookie NFL fantasy points (Half-PPR + TE Premium) from college stats.",
    version="1.0.0",
)

# Load model at startup
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


def _predict_single(rookie: RookieInput) -> PredictionOutput:
    """Run prediction for a single rookie."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first.",
        )

    row = {col: getattr(rookie, col) for col in ALL_FEATURE_COLS}
    df = pd.DataFrame([row])
    prediction = model.predict(df)[0]

    return PredictionOutput(
        player_name=rookie.player_name,
        position=rookie.position,
        predicted_fantasy_points=round(float(prediction), 2),
    )


# --- Endpoints ---

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(rookie: RookieInput):
    """Predict fantasy points for a single rookie."""
    return _predict_single(rookie)


@app.post("/predict/batch", response_model=list[PredictionOutput])
def predict_batch(rookies: list[RookieInput]):
    """Predict fantasy points for a batch of rookies."""
    return [_predict_single(r) for r in rookies]

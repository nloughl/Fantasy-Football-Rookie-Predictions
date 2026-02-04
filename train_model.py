"""
Train the Random Forest model and save it to disk.

Replicates the pipeline from 02_Model.ipynb:
  - Loads df_master.csv
  - Filters to college stats + fantasy points target
  - Builds a StandardScaler + OneHotEncoder + RandomForest pipeline
  - Trains on the full dataset and saves to models/model.joblib

Usage:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/processed/df_master.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

TARGET = "R_fantasy_points_halfppr_tep"

CATEGORICAL_COLS = ["position", "team", "C_conference", "C_team"]

COLUMNS_TO_DROP = [
    "player_name", "player_id_x", "player_id_y",
    "draft_year", "draft_round", "draft_pick_overall", "age_on_draft_day",
]


def load_and_prepare_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the master CSV and return (X, y) ready for training."""
    df = pd.read_csv(path)

    # Keep only college stats + target (drop individual rookie stat columns)
    columns_to_keep = [
        col for col in df.columns
        if not col.startswith("R_") or col == TARGET
    ]
    df = df[columns_to_keep].copy()
    df.drop(columns=COLUMNS_TO_DROP, inplace=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_pipeline(numeric_cols: list[str]) -> Pipeline:
    """Build the sklearn pipeline matching the notebook configuration."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
    ])
    return pipeline


def main():
    print(f"Loading data from {DATA_PATH}...")
    X, y = load_and_prepare_data(DATA_PATH)

    numeric_cols = [col for col in X.columns if col not in CATEGORICAL_COLS]
    print(f"Features: {len(numeric_cols)} numeric, {len(CATEGORICAL_COLS)} categorical")
    print(f"Samples: {len(X)}")

    pipeline = build_pipeline(numeric_cols)

    # Cross-validate before final training
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    print(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Train on full dataset for production
    print("\nTraining on full dataset...")
    pipeline.fit(X, y)

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

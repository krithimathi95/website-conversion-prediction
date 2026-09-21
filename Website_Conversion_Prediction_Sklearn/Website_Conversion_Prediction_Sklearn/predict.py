import os
import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "website_conversion_model.pkl"
model = joblib.load(MODEL_PATH)

visitor = pd.DataFrame([{
    "session_duration_seconds": 420,
    "pages_viewed": 8,
    "previous_visits": 3,
    "is_returning_visitor": 1,
    "cart_items": 2,
    "traffic_source": 2,
    "device_type": 1
}])

prediction = model.predict(visitor)[0]
probability = model.predict_proba(visitor)[0][1]

print("Conversion Prediction:", "Will Convert" if prediction == 1 else "Will Not Convert")
print(f"Conversion probability: {probability:.2%}")

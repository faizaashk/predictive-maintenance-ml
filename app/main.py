
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import shap
import numpy as np
import json

# 1. Load the saved ML model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_model.joblib"

model = joblib.load(MODEL_PATH)

THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = json.load(file)["threshold"]


# Initialize SHAP TreeExplainer
rf_classifier = model.named_steps["classifier"]

explainer = shap.TreeExplainer(rf_classifier)

# Get names of transformed features
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()


# 2. Create the FastAPI application
app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction using Random Forest",
    version="1.0.0"
)


# 3. Define the input format
class MachineData(BaseModel):
    Type: Literal["L", "M", "H"]
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float



@app.post("/predict")
def predict(data: MachineData):

    # Convert incoming readings into a DataFrame
    input_data = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    # Get failure probability
    probability = float(
    model.predict_proba(input_data)[0][1]
    )

    prediction = int(probability >= THRESHOLD)
    

    # Transform input using the fitted preprocessing pipeline
    transformed_input = model.named_steps[
        "preprocessor"
    ].transform(input_data)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    # Calculate SHAP values
    raw_shap_values = explainer.shap_values(transformed_input)

    # Handle different SHAP output formats
    if isinstance(raw_shap_values, list):
        contributions = np.asarray(raw_shap_values[1])[0]

    else:
        shap_array = np.asarray(raw_shap_values)

        if shap_array.ndim == 3:
            contributions = shap_array[0, :, 1]

        elif shap_array.ndim == 2:
            contributions = shap_array[0]

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_array.shape}"
            )


    base_values = np.asarray(explainer.expected_value)

    if base_values.size > 1:
        base_value = float(base_values.reshape(-1)[-1])
    else:
        base_value = float(base_values.item())

    print("SHAP expected value:", base_value)
    print("Sum of SHAP values:", float(np.sum(contributions)))
    print("Baseline + SHAP:", base_value + float(np.sum(contributions)))
    print("Model failure probability:", probability)

    # Prepare feature explanations
    explanations = []

    for feature, value in zip(feature_names, contributions):

        # Remove preprocessing prefixes
        readable_name = (
            feature.replace("num__", "")
                   .replace("cat__", "")
        )

        explanations.append({
            "feature": readable_name,
            "shap_value": round(float(value), 6),
            "direction": (
                "Increases failure prediction"
                if value > 0
                else "Decreases failure prediction"
                if value < 0
                else "No contribution"
            )
        })

    # Sort by absolute contribution
    explanations.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    # Return API response
    return {
        "prediction": prediction,
        "machine_status": (
            "Failure" if prediction == 1 else "Healthy"
        ),
        "failure_probability": round(probability * 100, 2),
        "shap_explanations": explanations
    }



# 5. Create a home endpoint
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running!"
    }

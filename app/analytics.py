
from pathlib import Path
import pandas as pd

# Find project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load original dataset
DATA_PATH = BASE_DIR / "data" / "pm.csv"

df = pd.read_csv(DATA_PATH)

# Sensor features
SENSOR_FEATURES = [
    "Torque [Nm]",
    "Tool wear [min]",
    "Rotational speed [rpm]",
    "Air temperature [K]",
    "Process temperature [K]"
]

# Calculate average readings for healthy and failed machines
sensor_averages = (
    df.groupby("Machine failure")[SENSOR_FEATURES]
    .mean()
    .T
)

# Rename columns
sensor_averages.columns = ["Healthy Average", "Failed Average"]

# Function to retrieve reference averages
def get_sensor_averages():
    return sensor_averages.copy()

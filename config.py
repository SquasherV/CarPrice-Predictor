# config.py

import os

# --- File Paths ---
DATA_PATH = 'cars_data_clean.csv'

# --- Directories ---
# Using os.path.join for better cross-platform compatibility
MODEL_DIR = "models"
REPORTS_DIR = "reports"

# --- File Paths for Artifacts ---
# We'll create the reports directory if it doesn't exist
METRICS_FILE = os.path.join(REPORTS_DIR, "metrics.json")


# --- Feature Definitions ---
# Using the EXACT column names from your file
TARGET = 'listed_price'

# Features for the model
CATEGORICAL_FEATURES = ['fuel', 'transmission', 'owner_type', 'oem', 'Seats']
NUMERICAL_FEATURES = ['myear', 'km']

# Engineered features that will be created
ENGINEERED_FEATURES = ['car_age']
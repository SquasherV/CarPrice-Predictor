# config.py

# --- File Paths ---
DATA_PATH = 'cars_data_clean.csv'
MODEL_PATH = 'car_price_model.joblib'

# --- Feature Definitions ---
# Using the EXACT column names from your file
TARGET = 'listed_price'

# Features for the model
CATEGORICAL_FEATURES = ['fuel', 'transmission', 'owner_type', 'oem', 'Seats']
NUMERICAL_FEATURES = ['myear', 'km']

# Engineered features that will be created
ENGINEERED_FEATURES = ['car_age']
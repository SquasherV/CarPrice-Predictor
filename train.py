# train.py

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import logging
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import config
from pipeline import create_pipeline

# --- Configure Logging ---
# Sets up logging to show INFO level messages with a timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def ensure_dir_exists(path):
    """Checks if a directory exists, and if not, creates it."""
    if not os.path.exists(path):
        logging.info(f"Directory not found. Creating directory: {path}")
        os.makedirs(path)
    else:
        logging.info(f"Directory already exists: {path}")

def run_training():
    """Loads data, processes it, trains the model, and saves artifacts."""
    logging.info("--- Starting Model Training ---")

    # 0. Create artifact directories
    ensure_dir_exists(config.MODEL_DIR)
    ensure_dir_exists(config.REPORTS_DIR)

    # 1. Load Data
    try:
        df = pd.read_csv(config.DATA_PATH)
        logging.info(f"Successfully loaded data from {config.DATA_PATH}")
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {config.DATA_PATH}")
        return

    # The 'Seats' column might have missing values, let's fill them
    df['Seats'].fillna(df['Seats'].mode()[0], inplace=True)
    logging.info("Handled missing 'Seats' values using mode.")


    # 2. Feature Engineering
    logging.info("Engineering 'car_age' feature using 'myear' column...")
    current_year = datetime.now().year
    df['car_age'] = current_year - df['myear']

    # 3. Prepare data for training
    features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES + config.ENGINEERED_FEATURES
    X = df[features]
    y = df[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples).")

    # 4. Create and Train Pipeline
    model_pipeline = create_pipeline()
    logging.info("Training the model...")
    model_pipeline.fit(X_train, y_train)
    logging.info("Model training complete.")

    # 5. Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logging.info("--- Model Evaluation ---")
    logging.info(f"R-squared (R2) Score: {r2:.4f}")
    logging.info(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")

    # 6. Save Evaluation Metrics
    logging.info(f"Saving metrics to: {config.METRICS_FILE}")
    metrics = {
        "r2_score": r2,
        "rmse": rmse,
        "training_date": datetime.now().isoformat(),
        "data_shape": df.shape,
        "test_size": X_test.shape[0]
    }
    
    try:
        with open(config.METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics successfully saved.")
    except IOError as e:
        logging.error(f"Failed to save metrics: {e}")

    # 7. Save the trained model with versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"car_price_predictor_{timestamp}.joblib"
    model_save_path = os.path.join(config.MODEL_DIR, model_filename)
    
    logging.info(f"Saving versioned model to: {model_save_path}")
    try:
        joblib.dump(model_pipeline, model_save_path)
        logging.info("Model training finished and all artifacts saved.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    run_training()

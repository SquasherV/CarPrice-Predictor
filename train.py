# train.py

import pandas as pd
import numpy as np
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import config
from pipeline import create_pipeline

def run_training():
    """Loads data, processes it, trains the model, and saves it."""
    print("--- Starting Model Training ---")

    # 1. Load Data
    df = pd.read_csv(config.DATA_PATH)
    # The 'Seats' column might have missing values, let's fill them with the most common value (mode)
    df['Seats'].fillna(df['Seats'].mode()[0], inplace=True)


    # 2. Feature Engineering
    print("Engineering 'car_age' feature using 'myear' column...")
    current_year = datetime.now().year
    # This now uses the correct column name: 'myear'
    df['car_age'] = current_year - df['myear']

    # 3. Prepare data for training
    features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES + config.ENGINEERED_FEATURES
    X = df[features]
    y = df[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Create and Train Pipeline
    model_pipeline = create_pipeline()
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\nModel Evaluation Complete!")
    print(f"   - R-squared ($R^2$) Score: {r2:.4f}")
    # The prices are in actual value, so the RMSE will be a larger number
    print(f"   - Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")

    # 6. Save the trained model
    print(f"\nSaving model to: {config.MODEL_PATH}")
    joblib.dump(model_pipeline, config.MODEL_PATH)
    print("✅ Model training finished and model saved.")

if __name__ == "__main__":
    run_training()
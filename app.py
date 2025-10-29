# app.py

import pandas as pd
import joblib
from datetime import datetime
import config
import os
import glob

def load_latest_model():
    """Finds and loads the latest .joblib model from the MODEL_DIR"""
    try:
        # Check if model directory exists
        if not os.path.exists(config.MODEL_DIR):
            print(f"Error: Model directory not found at '{config.MODEL_DIR}'.")
            print("Please run train.py first to train and save a model")
            return None
        
        # Find all model files in the directory
        list_of_models = glob.glob(os.path.join(config.MODEL_DIR, '*.joblib'))
        
        # Check if any models were found
        if not list_of_models:
            print(f"Error: No models (.joblib files) found in '{config.MODEL_DIR}'.")
            print("Please run train.py first to train and save a model")
            return None
        
        # Find the most recent model file based on creation time
        latest_model_path = max(list_of_models, key=os.path.getctime)
        print(f"Loading latest model: {os.path.basename(latest_model_path)}")
        
        # Load the model
        model_pipeline = joblib.load(latest_model_path)
        print("Model loaded successfully!")
        return model_pipeline

    except Exception as e:
        print(f"An error occurred while loading model: {e}")
        return None

# --- Load Model ---
model_pipeline = load_latest_model()

def predict_car_price(details: dict) -> str:
    """Predicts the price of a car using the loaded model"""
    if model_pipeline is None:
        return "Error: Model is not loaded. Cannot make prediction"
        
    try:
        input_df = pd.DataFrame([details])
        predicted_price = model_pipeline.predict(input_df)[0]
        return f"The estimated price for the car is ₹{predicted_price:,.2f}"
    except Exception as e:
        return f"Error during prediction: {e}"

# --- Example Usage with correct keys ---
if __name__ == "__main__":
    print("\n--- Car Price Predictor Tool ---")
    
    if model_pipeline is not None:
        # Example dictionary with all the correct feature names
        car_to_predict = {
            'myear': 2018,
            'km': 45000,
            'fuel': 'Petrol',
            'transmission': 'MANUAL',
            'owner_type': 'First Owner',
            'oem': 'Maruti', # 'oem' is the brand
            'Seats': 5.0,
            # We must also provide the engineered feature
            'car_age': datetime.now().year - 2018 
        }

        estimated_price = predict_car_price(car_to_predict)
        print(f"\nPrediction for the selected car: {estimated_price}")
    else:
        print("\nExiting application because no model could be loaded")

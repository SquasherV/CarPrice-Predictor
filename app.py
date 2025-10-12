# app.py

import pandas as pd
import joblib
from datetime import datetime
import config

try:
    model_pipeline = joblib.load(config.MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at '{config.MODEL_PATH}'. Please run train.py first.")
    exit()

def predict_car_price(details: dict) -> str:
    """Predicts the price of a car using the loaded model."""
    input_df = pd.DataFrame([details])
    predicted_price = model_pipeline.predict(input_df)[0]
    return f"The estimated price for the car is ₹{predicted_price:,.2f}"

# --- Example Usage with correct keys ---
if __name__ == "__main__":
    print("\n--- Car Price Predictor Tool ---")
    
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
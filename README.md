# Car Price Predictor 🚗

A machine learning project that predicts the selling price of used cars based on their features. This repository contains the complete pipeline for data cleaning, feature engineering, model training, and prediction.

## Overview

The goal of this project is to build a reliable tool for estimating the value of a used car. It uses an **XGBoost Regressor** model, a powerful and popular algorithm for tabular data, to learn the complex relationships between a car's attributes and its market price.



## Features

- **Data Cleaning:** Handles messy, real-world data to prepare it for modeling.
- **Feature Engineering:** Creates a new `car_age` feature for improved model performance.
- **Advanced Modeling:** Utilizes an XGBoost model for high prediction accuracy.
- **Modular Code:** The project is split into separate files for configuration, pipeline, training, and prediction, making it easy to understand and maintain.
- **Model Persistence:** Uses `joblib` to save the trained model, allowing for instant predictions without retraining.

## Project Structure

```
CAR MODEL/
├── cars_data_clean.csv      # The dataset used for training
├── config.py                # Stores file paths and feature lists
├── pipeline.py              # Defines the preprocessing and model pipeline
├── train.py                 # Script to train the model and save it
├── app.py                   # The final application to make predictions
└── car_price_model.joblib   # The saved, pre-trained model
```

## How to Use

### Prerequisites

- Python 3.x
- Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  Install the required libraries:
    ```bash
    pip install pandas scikit-learn xgboost joblib
    ```

### Running the Application

1.  **Train the Model:**
    First, run the training script. This will process the data and create the `car_price_model.joblib` file.
    ```bash
    python train.py
    ```

2.  **Get a Prediction:**
    Once the model is trained, run the application script to get a price estimate for the example car.
    ```bash
    python app.py
    ```
    You can customize the car's details in `app.py` to get predictions for different vehicles.

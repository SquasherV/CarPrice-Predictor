# pipeline.py

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ENGINEERED_FEATURES

def create_pipeline() -> Pipeline:
    """Creates and returns the scikit-learn pipeline for preprocessing and modeling."""
    
    numeric_features_total = NUMERICAL_FEATURES + ENGINEERED_FEATURES

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_total),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    return model_pipeline
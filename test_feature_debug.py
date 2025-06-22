#!/usr/bin/env python3
"""
Quick script to debug the feature mismatch issue by testing predictions
with manual feature preparation that matches training features.
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def test_feature_matching():
    """Test if adding missing categorical features fixes the prediction issue."""
    
    # Load the trained model
    with open('test_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully")
    
    # Create features that match training (11 features total)
    # Based on training: 9 numeric + 2 categorical (dataset_uuid, rule_code) = 11
    
    # Numeric features (from predictor.py)
    numeric_features = [
        1,  # day_of_week (Monday=0, Sunday=6, so 1=Tuesday)
        15,  # day_of_month  
        16,  # week_of_year
        4,   # month (April)
        0.0, # lag_1_day (default for missing)
        0.0, # lag_2_day (default for missing)
        0.0, # lag_7_day (default for missing)
        0.0, # ma_3_day (default for missing)
        0.0  # ma_7_day (default for missing)
    ]
    
    # Add categorical features (these are missing in prediction!)
    # Based on pipeline.py, these should be encoded as categorical
    # For LightGBM, we can pass string/categorical values directly
    categorical_features = [
        'dataset-001',  # dataset_uuid
        '1'             # rule_code (as string)
    ]
    
    # Combine all features
    all_features = numeric_features + categorical_features
    
    print(f"Total features prepared: {len(all_features)}")
    print(f"Features: {all_features}")
    
    # Try prediction with all 11 features
    try:
        # For LightGBM with categorical features, we need to create a DataFrame
        # with proper column names that match training
        feature_data = pd.DataFrame([all_features], columns=[
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'lag_1_day', 'lag_2_day', 'lag_7_day',
            'ma_3_day', 'ma_7_day',
            'dataset_uuid', 'rule_code'
        ])
        
        prediction = model.predict(feature_data)[0]
        print(f"SUCCESS! Prediction with 11 features: {prediction:.2f}%")
        
    except Exception as e:
        print(f"ERROR with 11 features: {e}")
    
    # Test with just numeric features (current predictor approach)
    try:
        numeric_only = np.array([numeric_features])
        prediction = model.predict(numeric_only)[0]
        print(f"Prediction with 9 numeric features: {prediction:.2f}%")
        
    except Exception as e:
        print(f"ERROR with 9 numeric features: {e}")

if __name__ == '__main__':
    test_feature_matching()
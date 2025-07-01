#!/usr/bin/env python3
"""
Debug the feature engineering and model input pipeline.
Investigate why predictions are 81x higher than expected.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_quality_summarizer.ml.data_loader import load_and_validate_csv, parse_results_column, create_binary_pass_column
from data_quality_summarizer.ml.aggregator import aggregate_pass_percentages
from data_quality_summarizer.ml.feature_engineer import extract_time_features, create_lag_features, calculate_moving_averages
from data_quality_summarizer.ml.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Debug the feature pipeline and model predictions."""
    print("🔍 Debugging Model Feature Pipeline\n")
    
    data_path = "test_data/large_sample.csv"
    model_path = "resources/reports/demo_model.pkl"
    
    print("=" * 60)
    print("1. TRAINING FEATURE PIPELINE")
    print("=" * 60)
    
    # Recreate the exact training pipeline
    raw_data = load_and_validate_csv(data_path)
    parsed_results = parse_results_column(raw_data['results'].tolist())
    raw_data['is_pass'] = create_binary_pass_column(parsed_results)
    
    # Aggregate data
    aggregated_data = aggregate_pass_percentages(raw_data)
    print(f"Aggregated shape: {aggregated_data.shape}")
    
    # Feature engineering (same as training)
    feature_data = extract_time_features(aggregated_data)
    feature_data = create_lag_features(feature_data)
    feature_data = calculate_moving_averages(feature_data)
    
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Feature columns: {list(feature_data.columns)}")
    
    # Show feature statistics for training data
    exclude_cols = ['pass_percentage', 'business_date', 'source', 'tenant_id', 'dataset_name']
    feature_cols = [col for col in feature_data.columns if col not in exclude_cols]
    
    print(f"\nTraining features: {feature_cols}")
    print(f"Training target (pass_percentage) stats:")
    print(f"  Min: {feature_data['pass_percentage'].min():.1f}")
    print(f"  Max: {feature_data['pass_percentage'].max():.1f}")
    print(f"  Mean: {feature_data['pass_percentage'].mean():.1f}")
    
    print("=" * 60)
    print("2. PREDICTION FEATURE PIPELINE")
    print("=" * 60)
    
    # Load the trained model and make a prediction
    historical_data = pd.read_csv(data_path)
    predictor = Predictor(model_path, historical_data)
    
    # Make a prediction (this will show us what features are generated)
    test_prediction = predictor.predict(
        dataset_uuid="dataset-001",
        rule_code="1", 
        business_date="2024-01-05"
    )
    
    print(f"\nPrediction result: {test_prediction:.2f}%")
    
    print("=" * 60)
    print("3. MODEL INSPECTION")
    print("=" * 60)
    
    # Load and inspect the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model)}")
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        print(f"\nFeature importance:")
        for i, (feat, imp) in enumerate(zip(feature_cols, feature_importance)):
            print(f"  {feat}: {imp:.4f}")
    
    # Check if there are any extreme feature values in prediction
    print("=" * 60)
    print("4. HYPOTHESIS TESTING")
    print("=" * 60)
    
    print("Possible causes of 81x higher predictions:")
    print("1. Model was trained on pass_percentage (0-100) but predicting on different scale")
    print("2. Feature values during prediction differ drastically from training")
    print("3. Missing feature normalization/scaling")
    print("4. Categorical encoding mismatch between training and prediction")
    
    # Test with known training data point
    print(f"\n5. TESTING WITH KNOWN TRAINING POINT")
    print("=" * 50)
    
    # Get a sample from training data
    sample_row = feature_data.iloc[0]
    print(f"Training sample: {sample_row['dataset_uuid']}, {sample_row['rule_code']}, {sample_row['business_date']}")
    print(f"Training target: {sample_row['pass_percentage']:.1f}%")
    
    # Make prediction for the same data point
    prediction_sample = predictor.predict(
        dataset_uuid=sample_row['dataset_uuid'],
        rule_code=str(sample_row['rule_code']),
        business_date=sample_row['business_date']
    )
    
    print(f"Prediction result: {prediction_sample:.2f}%")
    print(f"Ratio (prediction/actual): {prediction_sample/sample_row['pass_percentage']:.1f}x")

if __name__ == "__main__":
    main()
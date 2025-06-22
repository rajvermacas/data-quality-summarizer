#!/usr/bin/env python3
"""
Debug script to investigate why the ML model is predicting 0.0 for all inputs.
"""

import pickle
import pandas as pd
import numpy as np
import json
import sys
sys.path.append('src')

from data_quality_summarizer.ml.data_loader import load_and_validate_csv, parse_results_column, create_binary_pass_column
from data_quality_summarizer.ml.aggregator import aggregate_pass_percentages
from data_quality_summarizer.ml.feature_engineer import extract_time_features, create_lag_features, calculate_moving_averages

def load_model():
    """Load the trained model."""
    print("🔍 Loading trained model...")
    with open('demo_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ Model type: {type(model)}")
    print(f"   ✓ Model features: {model.num_feature()}")
    return model

def process_training_data():
    """Process training data through the same pipeline used during training."""
    print("🔄 Processing training data through ML pipeline...")
    
    # Load data
    df = pd.read_csv('large_test_data.csv')
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Parse results
    parsed_results = parse_results_column(df['results'].tolist())
    df['is_pass'] = create_binary_pass_column(parsed_results)
    
    # Aggregate
    aggregated = aggregate_pass_percentages(df)
    print(f"   ✓ Aggregated to {len(aggregated)} groups")
    
    # Feature engineering
    featured = extract_time_features(aggregated)
    featured = create_lag_features(featured, lag_days=[1, 2, 7])
    featured = calculate_moving_averages(featured, windows=[3, 7])
    
    print(f"   ✓ Final feature set shape: {featured.shape}")
    print(f"   ✓ Columns: {list(featured.columns)}")
    
    return featured

def analyze_features_and_target(data):
    """Analyze the features and target variable."""
    print("📊 Analyzing features and target...")
    
    target_col = 'pass_percentage' 
    print(f"   ✓ Target column '{target_col}' exists: {target_col in data.columns}")
    
    if target_col in data.columns:
        print(f"   ✓ Target range: {data[target_col].min():.3f} to {data[target_col].max():.3f}")
        print(f"   ✓ Target mean: {data[target_col].mean():.3f}")
        print(f"   ✓ Target distribution:")
        print(f"     - Zeros: {(data[target_col] == 0).sum()}")
        print(f"     - Non-zeros: {(data[target_col] > 0).sum()}")
        print(f"     - Ones: {(data[target_col] == 1).sum()}")
    
    # Check for categorical columns
    categorical_cols = [col for col in ['dataset_uuid', 'rule_code'] if col in data.columns]
    # Only include numeric feature columns (exclude strings and dates)
    exclude_cols = categorical_cols + [target_col] + ['source', 'tenant_id', 'dataset_name', 'business_date']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"   ✓ Categorical columns: {categorical_cols}")
    print(f"   ✓ Feature columns: {feature_cols}")
    
    return feature_cols, categorical_cols

def test_model_prediction(model, data, feature_cols):
    """Test model predictions on the processed data."""
    print("🎯 Testing model predictions...")
    
    # Prepare features (same as model training)
    X = data[feature_cols].fillna(0)
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Feature matrix range: {X.min().min():.3f} to {X.max().max():.3f}")
    
    # Check for any obvious issues
    print(f"   ✓ NaN values: {X.isna().sum().sum()}")
    print(f"   ✓ Infinite values: {np.isinf(X).sum().sum()}")
    
    # Make predictions
    predictions = model.predict(X)
    print(f"   ✓ Generated {len(predictions)} predictions")
    print(f"   ✓ Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
    print(f"   ✓ Prediction mean: {predictions.mean():.6f}")
    print(f"   ✓ Unique predictions: {len(np.unique(predictions))}")
    
    # Show first few predictions vs actual
    if 'pass_percentage' in data.columns:
        print(f"\n   📋 Sample predictions vs actual:")
        for i in range(min(10, len(predictions))):
            actual = data.iloc[i]['pass_percentage']
            pred = predictions[i]
            print(f"     Sample {i+1}: Actual={actual:.3f}, Predicted={pred:.6f}")
    
    return predictions

def investigate_model_internals(model):
    """Investigate model internals to understand why it's predicting 0.0."""
    print("🔬 Investigating model internals...")
    
    try:
        # Get feature importance
        importance = model.feature_importance()
        print(f"   ✓ Feature importance shape: {importance.shape}")
        print(f"   ✓ Feature importance range: {importance.min():.3f} to {importance.max():.3f}")
        print(f"   ✓ Non-zero importance features: {(importance > 0).sum()}")
        
        # Check model parameters
        print(f"   ✓ Model objective: {model.params.get('objective', 'unknown')}")
        print(f"   ✓ Number of trees: {model.num_trees()}")
        print(f"   ✓ Number of features: {model.num_feature()}")
        
    except Exception as e:
        print(f"   ❌ Error investigating model: {e}")

def main():
    """Main debugging function."""
    print("🐛 Starting ML Model Debugging")
    print("="*50)
    
    try:
        # Load model
        model = load_model()
        
        # Process data
        processed_data = process_training_data()
        
        # Analyze features
        feature_cols, categorical_cols = analyze_features_and_target(processed_data)
        
        # Test predictions
        predictions = test_model_prediction(model, processed_data, feature_cols)
        
        # Investigate model internals
        investigate_model_internals(model)
        
        print("\n✅ Debugging completed!")
        
        # Provide recommendations
        print("\n💡 RECOMMENDATIONS:")
        if predictions.max() == predictions.min() == 0.0:
            print("   • All predictions are 0.0 - model may be overfitting or underfitting")
            print("   • Check if target variable has sufficient variation")
            print("   • Consider increasing regularization or changing model parameters")
            print("   • Verify feature engineering is creating meaningful features")
        
    except Exception as e:
        print(f"\n❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
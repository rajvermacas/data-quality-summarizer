#!/usr/bin/env python3
"""
ML Pipeline Analysis and Visualization Script.
Creates comprehensive analysis of model performance and prediction accuracy.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import pickle
import json
from pathlib import Path

# Import ML pipeline components
import sys
sys.path.append('src')
from data_quality_summarizer.ml.data_loader import load_and_validate_csv, parse_results_column, create_binary_pass_column
from data_quality_summarizer.ml.aggregator import aggregate_pass_percentages
from data_quality_summarizer.ml.feature_engineer import extract_time_features, create_lag_features, calculate_moving_averages
from data_quality_summarizer.ml.data_splitter import split_data_chronologically
from data_quality_summarizer.ml.predictor import Predictor

def load_model_and_data():
    """Load trained model and test data."""
    # Load the trained model
    with open('demo_trained_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Load test data
    test_data = pd.read_csv('large_test_data.csv')
    
    return model_data, test_data

def prepare_ml_data_pipeline(csv_data):
    """Run the ML data preparation pipeline."""
    # Step 1: Parse results and create binary pass column
    print("📊 Parsing results column...")
    success_rate = parse_results_column(csv_data)
    print(f"   ✓ Parsed {success_rate*100:.1f}% of results successfully")
    
    binary_success_rate = create_binary_pass_column(csv_data)
    print(f"   ✓ Created binary pass column (success rate: {binary_success_rate*100:.1f}%)")
    
    # Step 2: Aggregate by dataset/rule/date
    print("🔄 Aggregating data by dataset/rule/date...")
    aggregated_data = aggregate_pass_percentages(csv_data)
    print(f"   ✓ Aggregated to {len(aggregated_data)} groups")
    
    # Step 3: Feature engineering
    print("⚙️ Engineering features...")
    feature_data = extract_time_features(aggregated_data)
    print(f"   ✓ Added time features: {['day_of_week', 'day_of_month', 'week_of_year', 'month']}")
    
    feature_data = create_lag_features(feature_data, lag_periods=[1, 2, 7])
    print(f"   ✓ Created lag features for periods: [1, 2, 7]")
    
    feature_data = calculate_moving_averages(feature_data, windows=[3, 7])
    print(f"   ✓ Calculated moving averages for windows: [3, 7]")
    
    # Step 4: Split data chronologically
    print("📈 Splitting data chronologically...")
    train_data, test_data = split_data_chronologically(feature_data, test_size=0.2)
    print(f"   ✓ Training samples: {len(train_data)}")
    print(f"   ✓ Testing samples: {len(test_data)}")
    
    return train_data, test_data

def analyze_model_predictions(model_data, test_data):
    """Analyze model predictions vs actual values."""
    print("🔍 Analyzing model predictions...")
    
    # Extract model and feature columns
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    # Prepare test features
    X_test = test_data[feature_columns].fillna(0)
    y_test = test_data['pass_percentage']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    print(f"   ✓ Generated {len(y_pred)} predictions")
    print(f"   ✓ Test set actual pass percentage range: {y_test.min():.3f} to {y_test.max():.3f}")
    print(f"   ✓ Predicted pass percentage range: {y_pred.min():.3f} to {y_pred.max():.3f}")
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"   📊 Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   📊 Mean Squared Error (MSE): {mse:.4f}")
    print(f"   📊 Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {
        'y_test': y_test,
        'y_pred': y_pred,
        'test_data': test_data,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

def create_comprehensive_visualizations(analysis_results):
    """Create comprehensive visualizations of model performance."""
    y_test = analysis_results['y_test']
    y_pred = analysis_results['y_pred']
    test_data = analysis_results['test_data']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Actual vs Predicted Scatter Plot
    plt.subplot(3, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Pass Percentage')
    plt.ylabel('Predicted Pass Percentage')
    plt.title('Actual vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    plt.subplot(3, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Pass Percentage')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution of Actual Values
    plt.subplot(3, 3, 3)
    plt.hist(y_test, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Actual Pass Percentage')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual Values')
    plt.grid(True, alpha=0.3)
    
    # 4. Distribution of Predicted Values
    plt.subplot(3, 3, 4)
    plt.hist(y_pred, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Predicted Pass Percentage')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 5. Time Series Plot (if business_date is available)
    plt.subplot(3, 3, 5)
    if 'business_date' in test_data.columns:
        test_data_sorted = test_data.copy()
        test_data_sorted['y_test'] = y_test.values
        test_data_sorted['y_pred'] = y_pred
        test_data_sorted = test_data_sorted.sort_values('business_date')
        
        plt.plot(pd.to_datetime(test_data_sorted['business_date']), test_data_sorted['y_test'], 
                'o-', label='Actual', alpha=0.7, markersize=3)
        plt.plot(pd.to_datetime(test_data_sorted['business_date']), test_data_sorted['y_pred'], 
                's-', label='Predicted', alpha=0.7, markersize=3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.title('Time Series: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No time data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Time Series Not Available')
    
    # 6. Error Distribution
    plt.subplot(3, 3, 6)
    errors = np.abs(y_test - y_pred)
    plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Errors')
    plt.grid(True, alpha=0.3)
    
    # 7. Performance by Rule Code
    plt.subplot(3, 3, 7)
    if 'rule_code' in test_data.columns:
        rule_performance = []
        for rule in test_data['rule_code'].unique():
            mask = test_data['rule_code'] == rule
            rule_mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
            rule_performance.append({'rule_code': rule, 'mae': rule_mae})
        
        rule_df = pd.DataFrame(rule_performance)
        plt.bar(rule_df['rule_code'], rule_df['mae'], alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Rule Code')
        plt.ylabel('Mean Absolute Error')
        plt.title('Performance by Rule Code')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No rule code data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Rule Performance Not Available')
    
    # 8. Cumulative Error Distribution
    plt.subplot(3, 3, 8)
    sorted_errors = np.sort(errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative_prob, linewidth=2)
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 9. Model Performance Summary
    plt.subplot(3, 3, 9)
    metrics = ['MAE', 'MSE', 'RMSE']
    values = [analysis_results['mae'], analysis_results['mse'], analysis_results['rmse']]
    
    bars = plt.bar(metrics, values, alpha=0.7, color=['red', 'blue', 'green'], edgecolor='black')
    plt.ylabel('Error Value')
    plt.title('Model Performance Metrics')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ml_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comprehensive visualization saved as 'ml_performance_analysis.png'")

def main():
    """Main analysis function."""
    print("🚀 Starting ML Pipeline Analysis & Visualization")
    print("=" * 60)
    
    try:
        # Load model and data
        print("📂 Loading model and test data...")
        model_data, raw_test_data = load_model_and_data()
        print(f"   ✓ Loaded model with {len(model_data['feature_columns'])} features")
        print(f"   ✓ Loaded test data with {len(raw_test_data)} samples")
        
        # Process data through ML pipeline
        print("\n🔄 Processing data through ML pipeline...")
        train_data, test_data = prepare_ml_data_pipeline(raw_test_data)
        
        # Analyze predictions
        print("\n🎯 Analyzing model predictions...")
        analysis_results = analyze_model_predictions(model_data, test_data)
        
        # Create visualizations
        print("\n📈 Creating comprehensive visualizations...")
        create_comprehensive_visualizations(analysis_results)
        
        print("\n✅ Analysis completed successfully!")
        print("   📁 Visualization saved: ml_performance_analysis.png")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
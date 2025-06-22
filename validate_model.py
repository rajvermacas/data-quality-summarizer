#!/usr/bin/env python3
"""
Comprehensive model validation script to assess prediction accuracy.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys
import logging

# Add src to path
sys.path.append('src')

from data_quality_summarizer.ml.predictor import Predictor
from data_quality_summarizer.ml.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model_predictions():
    """Run comprehensive model validation on test data."""
    
    # Load the trained model and test data
    model_path = "demo_model.pkl"
    test_data_path = "demo_subset.csv"
    
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not Path(test_data_path).exists():
        logger.error(f"Test data file not found: {test_data_path}")
        return
    
    # Load test data
    logger.info("Loading test data...")
    data_loader = DataLoader()
    df = data_loader.load_csv_data(test_data_path)
    
    # Parse results and create actual pass percentages
    df['pass'] = df['results'].str.contains('Pass', na=False).astype(int)
    
    # Group by dataset_uuid, rule_code, and business_date to get actual pass rates
    actual_data = df.groupby(['dataset_uuid', 'rule_code', 'business_date']).agg({
        'pass': 'mean'  # This gives us the pass percentage (0-100)
    }).reset_index()
    actual_data['actual_pass_percentage'] = actual_data['pass'] * 100
    
    logger.info(f"Created {len(actual_data)} actual data points")
    
    # Load predictor
    logger.info("Loading predictor...")
    predictor = Predictor(model_path=model_path)
    
    # Make predictions for all test cases
    predictions = []
    actuals = []
    
    logger.info("Making predictions...")
    for idx, row in actual_data.iterrows():
        try:
            prediction = predictor.predict(
                dataset_uuid=row['dataset_uuid'],
                rule_code=str(row['rule_code']),
                business_date=row['business_date']
            )
            predictions.append(prediction)
            actuals.append(row['actual_pass_percentage'])
            
            if idx < 10:  # Show first 10 for debugging
                logger.info(f"Row {idx}: UUID={row['dataset_uuid']}, Rule={row['rule_code']}, "
                          f"Date={row['business_date']}, Actual={row['actual_pass_percentage']:.1f}%, "
                          f"Predicted={prediction:.1f}%")
                
        except Exception as e:
            logger.error(f"Failed to predict row {idx}: {e}")
            continue
    
    # Calculate evaluation metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions_clean = predictions[valid_mask]
    actuals_clean = actuals[valid_mask]
    
    if len(predictions_clean) == 0:
        logger.error("No valid predictions found!")
        return
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions_clean - actuals_clean))
    mse = np.mean((predictions_clean - actuals_clean) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE - handle division by zero
    non_zero_mask = actuals_clean != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((actuals_clean[non_zero_mask] - predictions_clean[non_zero_mask]) / actuals_clean[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # R-squared
    ss_res = np.sum((actuals_clean - predictions_clean) ** 2)
    ss_tot = np.sum((actuals_clean - np.mean(actuals_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Create detailed report
    report = f"""
    
🔍 MODEL VALIDATION REPORT
==========================

📊 Dataset Statistics:
  • Total predictions: {len(predictions_clean)}
  • Actual pass % range: {actuals_clean.min():.1f}% - {actuals_clean.max():.1f}%
  • Predicted pass % range: {predictions_clean.min():.1f}% - {predictions_clean.max():.1f}%
  • Actual mean: {actuals_clean.mean():.1f}%
  • Predicted mean: {predictions_clean.mean():.1f}%

🎯 Accuracy Metrics:
  • MAE (Mean Absolute Error): {mae:.2f}%
  • MSE (Mean Squared Error): {mse:.2f}
  • RMSE (Root Mean Squared Error): {rmse:.2f}%
  • MAPE (Mean Absolute Percentage Error): {mape:.2f}%
  • R² (Coefficient of Determination): {r2:.3f}

📈 Prediction Quality Assessment:
"""
    
    # Quality assessment
    if mae < 5:
        report += "  • EXCELLENT: Predictions are very accurate (MAE < 5%)\n"
    elif mae < 10:
        report += "  • GOOD: Predictions are reasonably accurate (MAE < 10%)\n"
    elif mae < 20:
        report += "  • FAIR: Predictions have moderate accuracy (MAE < 20%)\n"
    else:
        report += "  • POOR: Predictions need improvement (MAE > 20%)\n"
    
    if r2 > 0.8:
        report += "  • HIGH CORRELATION: Model captures data patterns well (R² > 0.8)\n"
    elif r2 > 0.5:
        report += "  • MODERATE CORRELATION: Model captures some patterns (R² > 0.5)\n"
    else:
        report += "  • LOW CORRELATION: Model needs better features (R² < 0.5)\n"
    
    # Sample comparisons
    report += "\n🔢 Sample Predictions vs Actuals:\n"
    sample_indices = np.random.choice(len(predictions_clean), min(10, len(predictions_clean)), replace=False)
    for i, idx in enumerate(sample_indices):
        actual = actuals_clean[idx]
        predicted = predictions_clean[idx]
        error = abs(predicted - actual)
        report += f"  • Sample {i+1}: Actual={actual:.1f}%, Predicted={predicted:.1f}%, Error={error:.1f}%\n"
    
    print(report)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'actual_pass_percentage': actuals_clean,
        'predicted_pass_percentage': predictions_clean,
        'absolute_error': np.abs(predictions_clean - actuals_clean),
        'percentage_error': np.abs((actuals_clean - predictions_clean) / np.maximum(actuals_clean, 1)) * 100
    })
    
    results_df.to_csv('model_validation_results.csv', index=False)
    logger.info("Detailed results saved to model_validation_results.csv")
    
    return {
        'mae': mae,
        'mse': mse, 
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'total_predictions': len(predictions_clean)
    }

if __name__ == "__main__":
    validate_model_predictions()
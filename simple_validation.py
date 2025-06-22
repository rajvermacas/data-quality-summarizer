#!/usr/bin/env python3
"""
Simple model validation to check prediction accuracy.
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_quality_summarizer.ml.predictor import Predictor

def analyze_predictions():
    """Analyze model predictions against actual data."""
    
    print("🔍 MODEL VALIDATION ANALYSIS")
    print("=" * 40)
    
    # Check if model exists
    if not Path("demo_model.pkl").exists():
        print("❌ Model file 'demo_model.pkl' not found")
        return
    
    # Load test data  
    if not Path("demo_subset.csv").exists():
        print("❌ Test data file 'demo_subset.csv' not found")
        return
    
    print("📊 Loading test data...")
    df = pd.read_csv("demo_subset.csv")
    print(f"   • Loaded {len(df)} records")
    
    # Parse JSON results to get actual pass/fail
    print("🔄 Parsing actual results...")
    def parse_result(result_str):
        try:
            result_dict = json.loads(result_str)
            return 1 if result_dict.get('result') == 'Pass' else 0
        except:
            return 0
    
    df['actual_pass'] = df['results'].apply(parse_result)
    
    # Group by dataset_uuid, rule_code to get actual pass rates
    actual_rates = df.groupby(['dataset_uuid', 'rule_code', 'business_date']).agg({
        'actual_pass': 'mean'
    }).reset_index()
    actual_rates['actual_pass_percentage'] = actual_rates['actual_pass'] * 100
    
    print(f"   • Found {len(actual_rates)} unique dataset/rule/date combinations")
    
    # Load predictor
    print("🤖 Loading trained model...")
    predictor = Predictor(model_path="demo_model.pkl", historical_data=pd.DataFrame())
    
    # Make predictions and compare
    print("🎯 Making predictions and comparing...")
    results = []
    
    for idx, row in actual_rates.iterrows():
        try:
            prediction = predictor.predict(
                dataset_uuid=row['dataset_uuid'],
                rule_code=str(row['rule_code']),
                business_date=row['business_date']
            )
            
            actual = row['actual_pass_percentage']
            error = abs(prediction - actual)
            
            results.append({
                'dataset_uuid': row['dataset_uuid'],
                'rule_code': row['rule_code'],
                'date': row['business_date'],
                'actual': actual,
                'predicted': prediction,
                'error': error
            })
            
            if idx < 10:  # Show first 10
                print(f"   • {row['dataset_uuid'][:8]}... Rule {row['rule_code']}: "
                      f"Actual={actual:.1f}%, Predicted={prediction:.1f}%, Error={error:.1f}%")
                
        except Exception as e:
            print(f"   ❌ Failed to predict row {idx}: {e}")
            continue
    
    if not results:
        print("❌ No successful predictions made")
        return
    
    # Calculate metrics
    actuals = [r['actual'] for r in results]
    predictions = [r['predicted'] for r in results]
    errors = [r['error'] for r in results]
    
    mae = np.mean(errors)
    mse = np.mean([e**2 for e in errors])
    rmse = np.sqrt(mse)
    
    # Check prediction behavior
    unique_predictions = len(set(predictions))
    all_same = unique_predictions == 1
    
    print(f"\n📈 VALIDATION RESULTS:")
    print(f"   • Total predictions: {len(results)}")
    print(f"   • Unique predictions: {unique_predictions}")
    print(f"   • All predictions same: {all_same}")
    print(f"   • Actual range: {min(actuals):.1f}% - {max(actuals):.1f}%")
    print(f"   • Predicted range: {min(predictions):.1f}% - {max(predictions):.1f}%")
    print(f"   • MAE (Mean Absolute Error): {mae:.2f}%")
    print(f"   • RMSE (Root Mean Squared Error): {rmse:.2f}%")
    
    # Analyze results
    print(f"\n🚨 ANALYSIS:")
    if all_same:
        print("   ❌ CRITICAL ISSUE: All predictions are identical!")
        print("   → This suggests the model is not learning from features")
        print("   → Possible causes: insufficient training data, feature scaling issues, or overfitting")
    elif mae < 5:
        print("   ✅ EXCELLENT: Model predictions are very accurate")
    elif mae < 15:
        print("   ✅ GOOD: Model predictions are reasonably accurate")
    elif mae < 30:
        print("   ⚠️ FAIR: Model has moderate accuracy, could be improved")
    else:
        print("   ❌ POOR: Model predictions need significant improvement")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('validation_report.csv', index=False)
    print(f"\n📁 Detailed results saved to: validation_report.csv")
    
    return results

if __name__ == "__main__":
    analyze_predictions()
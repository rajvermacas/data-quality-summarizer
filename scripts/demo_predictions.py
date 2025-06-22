#!/usr/bin/env python3
"""
Demo script to showcase ML prediction capabilities.
Trains a model and demonstrates both single and batch predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add src directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_quality_summarizer.ml.pipeline import MLPipeline
from data_quality_summarizer.ml.predictor import Predictor
from data_quality_summarizer.ml.batch_predictor import BatchPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the prediction demo."""
    logger.info("Starting ML Prediction Demo")
    
    # File paths - use larger dataset for training
    data_path = Path("test_data/large_sample.csv")
    rules_path = Path("test_data/sample_rules.json")
    model_path = Path("resources/reports/demo_model.pkl")
    
    # Ensure directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Train a model using the pipeline
        logger.info("Step 1: Training ML model...")
        pipeline = MLPipeline()
        
        # Load rule metadata
        import json
        with open(rules_path, 'r') as f:
            rule_metadata = json.load(f)
        
        # Train the model
        results = pipeline.train_model(
            str(data_path), 
            rule_metadata, 
            str(model_path)
        )
        
        logger.info(f"Model training results: {results}")
        
        # Check if training was successful
        if not results.get('success', False):
            logger.error(f"Model training failed: {results.get('error', 'Unknown error')}")
            return 1
        
        # Step 2: Load historical data for predictions
        logger.info("Step 2: Preparing historical data...")
        historical_data = pd.read_csv(data_path)
        
        # Step 3: Initialize predictor
        logger.info("Step 3: Initializing predictor...")
        predictor = Predictor(model_path, historical_data)
        
        # Step 4: Single prediction example
        logger.info("Step 4: Single prediction example...")
        single_prediction = predictor.predict(
            dataset_uuid="uuid-123",
            rule_code="101", 
            business_date="2024-01-20"  # Predict for future date
        )
        
        print("\n=== SINGLE PREDICTION EXAMPLE ===")
        print(f"Dataset UUID: uuid-123")
        print(f"Rule Code: 101 (ROW_COUNT_CHECK)")
        print(f"Prediction Date: 2024-01-20")
        print(f"Predicted Pass Percentage: {single_prediction:.2f}%")
        
        # Step 5: Batch prediction example
        logger.info("Step 5: Batch prediction example...")
        
        # Create sample batch requests using real dataset UUIDs from our data
        batch_requests = [
            {"dataset_uuid": "dataset-001", "rule_code": "1", "business_date": "2024-01-05"},
            {"dataset_uuid": "dataset-002", "rule_code": "2", "business_date": "2024-01-05"},
            {"dataset_uuid": "dataset-003", "rule_code": "3", "business_date": "2024-01-05"},
        ]
        
        # Make individual predictions (simulating batch processing)
        batch_predictions = []
        for req in batch_requests:
            try:
                pred = predictor.predict(
                    dataset_uuid=req["dataset_uuid"],
                    rule_code=req["rule_code"],
                    business_date=req["business_date"]
                )
                batch_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {req}: {e}")
                batch_predictions.append(0.0)
        
        print("\n=== BATCH PREDICTION EXAMPLE ===")
        print("Input Batch:")
        for req in batch_requests:
            print(f"  Dataset: {req['dataset_uuid']}, Rule: {req['rule_code']}, Date: {req['business_date']}")
        print("\nPredictions:")
        for i, result in enumerate(batch_predictions):
            req = batch_requests[i]
            pred = result.get('prediction', 0.0) if isinstance(result, dict) else result
            print(f"Dataset {req['dataset_uuid']}, Rule {req['rule_code']}: {pred:.2f}% pass rate")
        
        # Step 6: Historical trend analysis
        logger.info("Step 6: Historical trend analysis...")
        
        print("\n=== HISTORICAL TREND ANALYSIS ===")
        print("Recent historical performance for comparison:")
        
        # Show recent performance from historical data
        recent_data = historical_data.tail(5)[['dataset_uuid', 'rule_code', 'business_date', 'results']]
        print(recent_data.to_string(index=False))
        
        logger.info("ML Prediction Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
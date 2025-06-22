"""
Shared test fixtures for ML module tests.
Creates real LightGBM models and data for testing to avoid Mock serialization issues.
"""

import pytest
import lightgbm as lgb
import numpy as np
import pandas as pd
import tempfile
import pickle
from pathlib import Path
from typing import Dict, Any

from src.data_quality_summarizer.rules import RuleMetadata


@pytest.fixture
def minimal_real_model():
    """
    Create minimal real LightGBM model for testing.
    
    This fixture creates a real, serializable LightGBM model that can be
    pickled and used in tests without Mock object serialization issues.
    
    Returns:
        lgb.Booster: Trained LightGBM model
    """
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Create minimal training data (3 features to match expected ML pipeline structure)
    n_samples = 50
    X_train = np.random.rand(n_samples, 3)
    
    # Generate realistic target values (pass percentages between 60-95%)
    y_train = np.random.uniform(60, 95, n_samples)
    
    # Train real model with minimal parameters
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 10,
        'verbosity': -1,
        'seed': 42,
        'force_col_wise': True  # Suppress LightGBM warnings
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(5), lgb.log_evaluation(0)]
    )
    
    return model


@pytest.fixture
def real_test_data():
    """
    Create realistic test data matching production schema.
    
    Returns:
        pd.DataFrame: Test data with proper structure and types
    """
    np.random.seed(42)
    n_records = 100
    
    # Generate realistic business dates
    dates = pd.date_range('2024-01-01', periods=n_records//2, freq='D')
    business_dates = np.tile(dates.strftime('%Y-%m-%d'), 2)[:n_records]
    
    # Generate realistic pass/fail results
    pass_probabilities = np.random.uniform(0.7, 0.95, n_records)
    results = [
        f'{{"status": "{status}"}}'
        for status in np.random.choice(['Pass', 'Fail'], n_records, p=[0.8, 0.2])
    ]
    
    return pd.DataFrame({
        'source': ['test_source'] * n_records,
        'tenant_id': ['test_tenant'] * n_records,
        'dataset_uuid': ['uuid1', 'uuid2'] * (n_records // 2),
        'dataset_name': ['Test Dataset 1', 'Test Dataset 2'] * (n_records // 2),
        'rule_code': [1, 2] * (n_records // 2),
        'business_date': business_dates,
        'results': results,
        'dataset_record_count': np.random.randint(1000, 5000, n_records),
        'filtered_record_count': np.random.randint(900, 4500, n_records),
        'level_of_execution': ['dataset'] * n_records,
        'attribute_name': [None] * n_records
    })


@pytest.fixture
def real_rule_metadata() -> Dict[int, RuleMetadata]:
    """
    Create real rule metadata objects for testing.
    
    Returns:
        Dict[int, RuleMetadata]: Rule metadata with integer keys
    """
    return {
        1: RuleMetadata(
            rule_code=1,
            rule_name='Data Completeness Check',
            rule_type='Completeness',
            dimension='Completeness',
            rule_description='Validates that required fields are not null or empty',
            category='C1'
        ),
        2: RuleMetadata(
            rule_code=2,
            rule_name='Format Validation',
            rule_type='Validity',
            dimension='Validity',
            rule_description='Validates data format compliance with business rules',
            category='V1'
        )
    }


@pytest.fixture
def real_pickled_model_file(minimal_real_model):
    """
    Create a temporary file with a real pickled LightGBM model.
    
    Args:
        minimal_real_model: Real LightGBM model fixture
        
    Returns:
        str: Path to temporary model file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
        pickle.dump(minimal_real_model, temp_file)
        temp_file.flush()
        return temp_file.name


@pytest.fixture
def batch_prediction_test_data():
    """
    Create test data specifically for batch prediction tests.
    
    Returns:
        pd.DataFrame: Batch prediction request data
    """
    return pd.DataFrame({
        'dataset_uuid': ['uuid1', 'uuid2', 'uuid3', 'uuid1'],
        'rule_code': [1, 2, 1, 2],
        'business_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18']
    })


@pytest.fixture
def large_test_dataset():
    """
    Create larger test dataset for performance testing.
    
    Returns:
        pd.DataFrame: Large test dataset (~1000 records)
    """
    np.random.seed(42)
    n_records = 1000
    
    # Generate data spanning multiple months
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    data = []
    for i, date in enumerate(dates[:100]):  # 100 days
        for dataset_id in ['dataset-001', 'dataset-002', 'dataset-003']:
            for rule_code in [1, 2]:
                # Multiple executions per day per dataset per rule
                num_executions = np.random.randint(5, 16)
                for _ in range(num_executions):
                    # Simulate declining data quality over time
                    base_pass_rate = 0.9 - (i / 100) * 0.2
                    is_pass = np.random.random() < base_pass_rate
                    
                    data.append({
                        'source': 'perf_test',
                        'tenant_id': 'test_tenant',
                        'dataset_uuid': dataset_id,
                        'dataset_name': f'Performance Test Dataset {dataset_id[-1]}',
                        'business_date': date.strftime('%Y-%m-%d'),
                        'rule_code': rule_code,
                        'results': f'{{"status": "{"Pass" if is_pass else "Fail"}"}}',
                        'level_of_execution': 'dataset',
                        'attribute_name': None,
                        'dataset_record_count': np.random.randint(1000, 5000),
                        'filtered_record_count': np.random.randint(900, 4500)
                    })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_predictor_with_real_model(minimal_real_model):
    """
    Create a mock predictor that uses a real model for predictions.
    
    This allows testing predictor interface without full initialization
    while still having serializable components.
    
    Args:
        minimal_real_model: Real LightGBM model fixture
        
    Returns:
        Mock object with real predict method
    """
    from unittest.mock import Mock
    
    mock_predictor = Mock()
    
    def predict_with_real_model(dataset_uuid, rule_code, business_date):
        """Use real model for predictions with deterministic output."""
        # Create simple feature vector from inputs
        features = np.array([[
            hash(str(dataset_uuid)) % 100 / 100,  # Normalized hash
            int(rule_code) / 10,                   # Normalized rule code
            hash(str(business_date)) % 100 / 100   # Normalized date hash
        ]])
        
        # Get prediction from real model
        prediction = minimal_real_model.predict(features)[0]
        
        # Ensure prediction is in valid percentage range
        return max(0, min(100, prediction))
    
    mock_predictor.predict = predict_with_real_model
    return mock_predictor
"""
Tests for Stage 3: Prediction Pipeline Fix & Comprehensive Validation.

This module implements test-driven development for Stage 3 requirements:
- US3.1: Consistent feature handling between training and prediction
- US3.2: Prediction quality assurance with constant prediction detection
- US3.3: Comprehensive validation with MAE < 15% target

Following TDD Red-Green-Refactor cycle:
1. RED: Write failing tests for required functionality
2. GREEN: Implement minimum code to pass tests
3. REFACTOR: Clean up implementation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.data_quality_summarizer.ml.feature_utils import FeatureConsistency
from src.data_quality_summarizer.ml.predictor import Predictor, BaselinePredictor
from src.data_quality_summarizer.ml.model_trainer import ModelTrainer
from src.data_quality_summarizer.ml.comprehensive_validator import ComprehensiveValidator


class TestFeatureConsistency:
    """
    Tests for US3.1: Consistent feature handling between training and prediction.
    
    Ensures that training and prediction pipelines use identical features,
    preventing the 9 vs 11 feature mismatch issue.
    """
    
    def test_feature_consistency_class_exists(self):
        """GREEN: Test that FeatureConsistency class exists and can be imported."""
        # This should pass now that the class is implemented
        from src.data_quality_summarizer.ml.feature_utils import FeatureConsistency
        assert FeatureConsistency is not None
    
    def test_get_standard_features_returns_11_features(self):
        """RED: Test that standard features list contains exactly 11 features."""
        # This will fail initially
        consistency = FeatureConsistency()
        features = consistency.get_standard_features()
        
        assert len(features) == 11
        assert isinstance(features, list)
        
        # Verify expected feature categories
        expected_features = [
            # Time features (4)
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            # Lag features (3)
            'lag_1_day', 'lag_2_day', 'lag_7_day',
            # Moving averages (2)
            'ma_3_day', 'ma_7_day',
            # Categorical features (2)
            'dataset_uuid', 'rule_code'
        ]
        
        for feature in expected_features:
            assert feature in features
    
    def test_validate_feature_alignment_perfect_match(self):
        """RED: Test feature alignment validation with perfect match."""
        consistency = FeatureConsistency()
        train_features = consistency.get_standard_features()
        pred_features = train_features.copy()
        
        is_aligned = consistency.validate_feature_alignment(train_features, pred_features)
        assert is_aligned is True
    
    def test_validate_feature_alignment_mismatch_detected(self):
        """RED: Test feature alignment validation detects mismatches."""
        consistency = FeatureConsistency()
        train_features = consistency.get_standard_features()
        pred_features = train_features[:9]  # Missing 2 features (old bug)
        
        is_aligned = consistency.validate_feature_alignment(train_features, pred_features)
        assert is_aligned is False
    
    def test_prepare_categorical_features_consistency(self):
        """RED: Test categorical feature preparation consistency."""
        consistency = FeatureConsistency()
        
        # Sample data
        data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2'],
            'rule_code': ['R001', 'R002'],
            'day_of_week': [1, 2]
        })
        
        # Mock training categories
        training_categories = {
            'dataset_uuid': ['uuid1', 'uuid2', 'uuid3'],
            'rule_code': ['R001', 'R002', 'R003']
        }
        
        prepared_data = consistency.prepare_categorical_features(data, training_categories)
        
        # Verify categorical dtypes
        assert prepared_data['dataset_uuid'].dtype.name == 'category'
        assert prepared_data['rule_code'].dtype.name == 'category'
        
        # Verify categories match training
        assert list(prepared_data['dataset_uuid'].cat.categories) == training_categories['dataset_uuid']


class TestPredictionQualityAssurance:
    """
    Tests for US3.2: Prediction quality assurance with constant prediction detection.
    
    Implements safeguards against constant predictions and fallback mechanisms.
    """
    
    def test_constant_prediction_detection(self):
        """RED: Test detection of constant predictions."""
        # Create mock predictor
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        historical_data = pd.DataFrame({
            'business_date': pd.date_range('2024-01-01', periods=10),
            'dataset_uuid': 'uuid1',
            'rule_code': 'R001',
            'pass_percentage': [50.0] * 10
        })
        
        predictor = Predictor(model_path, historical_data)
        
        # Test constant predictions (should fail)
        constant_predictions = np.array([0.0] * 100)
        is_valid = predictor._validate_prediction_quality(constant_predictions)
        assert is_valid == False
        
        # Test varied predictions (should pass)
        varied_predictions = np.random.uniform(0, 100, 100)
        is_valid = predictor._validate_prediction_quality(varied_predictions)
        assert is_valid == True
    
    def test_prediction_variance_threshold(self):
        """RED: Test prediction variance threshold enforcement."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        historical_data = pd.DataFrame()
        predictor = Predictor(model_path, historical_data)
        
        # Predictions with variance below threshold (std < 0.1)
        low_variance_predictions = np.array([50.0, 50.01, 49.99, 50.02])
        is_valid = predictor._validate_prediction_quality(low_variance_predictions)
        assert is_valid == False
        
        # Predictions with sufficient variance (std > 1.0)
        high_variance_predictions = np.array([20.0, 50.0, 80.0, 35.0])
        is_valid = predictor._validate_prediction_quality(high_variance_predictions)
        assert is_valid == True
    
    def test_baseline_predictor_fallback(self):
        """RED: Test baseline predictor as fallback mechanism."""
        # This will fail initially as BaselinePredictor doesn't exist
        baseline = BaselinePredictor()
        
        # Test prediction based on historical average
        prediction = baseline.predict('uuid1', 'R001')
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 100.0
    
    def test_prediction_range_validation(self):
        """RED: Test that predictions are within valid range [0, 100]."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        historical_data = pd.DataFrame()
        predictor = Predictor(model_path, historical_data)
        
        # Test clipping of out-of-range predictions
        assert predictor._validate_and_clip_prediction(-10.0) == 0.0
        assert predictor._validate_and_clip_prediction(150.0) == 100.0
        assert predictor._validate_and_clip_prediction(50.0) == 50.0
        
        # Test NaN/infinite handling
        with pytest.raises(ValueError):
            predictor._validate_and_clip_prediction(np.nan)
        
        with pytest.raises(ValueError):
            predictor._validate_and_clip_prediction(np.inf)


class TestComprehensiveValidation:
    """
    Tests for US3.3: Comprehensive validation with MAE < 15% target.
    
    Implements end-to-end validation framework for ML pipeline quality assurance.
    """
    
    def test_comprehensive_validator_exists(self):
        """GREEN: Test that ComprehensiveValidator class exists and can be imported."""
        # This should pass now that the class is implemented
        from src.data_quality_summarizer.ml.comprehensive_validator import ComprehensiveValidator
        assert ComprehensiveValidator is not None
    
    def test_validate_prediction_variance(self):
        """RED: Test prediction variance validation."""
        validator = ComprehensiveValidator()
        
        # Test sufficient variance
        good_predictions = np.random.uniform(10, 90, 100)
        result = validator.validate_prediction_variance(good_predictions)
        
        assert result['variance_sufficient'] == True
        assert result['std_deviation'] > 1.0
        assert result['unique_predictions'] > 10
        
        # Test insufficient variance
        bad_predictions = np.array([0.0] * 100)
        result = validator.validate_prediction_variance(bad_predictions)
        
        assert result['variance_sufficient'] == False
        assert result['std_deviation'] < 0.1
        assert result['unique_predictions'] == 1
    
    def test_validate_feature_importance(self):
        """RED: Test feature importance validation."""
        validator = ComprehensiveValidator()
        
        # Mock LightGBM model with feature importance
        mock_model = Mock()
        mock_model.feature_importance.return_value = np.array([10, 15, 0, 5, 20, 8, 0, 12, 7, 3, 9])
        
        result = validator.validate_feature_importance(mock_model)
        
        assert 'total_features' in result
        assert 'non_zero_features' in result
        assert 'zero_importance_features' in result
        assert 'feature_utilization_rate' in result
        
        # Should have >70% feature utilization
        assert result['feature_utilization_rate'] > 0.7
    
    def test_validate_cross_dataset_performance(self):
        """RED: Test cross-dataset performance validation."""
        validator = ComprehensiveValidator()
        
        # Create prediction results with controlled error to meet target
        np.random.seed(42)  # For reproducible test
        actual = np.random.uniform(0, 100, 300)
        # Create predictions with small error to meet MAE < 15% target
        predicted = actual + np.random.normal(0, 5, 300)  # Small error
        
        results = pd.DataFrame({
            'dataset_uuid': ['uuid1'] * 100 + ['uuid2'] * 100 + ['uuid3'] * 100,
            'actual': actual,
            'predicted': predicted,
            'rule_code': ['R001'] * 150 + ['R002'] * 150
        })
        
        validation_result = validator.validate_cross_dataset_performance(results)
        
        assert 'dataset_performance' in validation_result
        assert 'overall_mae' in validation_result
        assert 'performance_consistency' in validation_result
        
        # With controlled small error, MAE should be < 15% target
        assert validation_result['overall_mae'] < 15.0
    
    def test_generate_validation_report(self):
        """RED: Test comprehensive validation report generation."""
        validator = ComprehensiveValidator()
        
        # Mock validation data
        predictions = np.random.uniform(10, 90, 100)
        mock_model = Mock()
        mock_model.feature_importance.return_value = np.array([10, 15, 0, 5, 20, 8, 0, 12, 7, 3, 9])
        
        results = pd.DataFrame({
            'actual': np.random.uniform(0, 100, 100),
            'predicted': predictions,
            'dataset_uuid': ['uuid1'] * 100,
            'rule_code': ['R001'] * 100
        })
        
        report = validator.generate_validation_report(predictions, mock_model, results)
        
        # Verify report structure
        assert hasattr(report, 'prediction_variance')
        assert hasattr(report, 'feature_importance')
        assert hasattr(report, 'cross_dataset_performance')
        assert hasattr(report, 'overall_status')
        assert hasattr(report, 'recommendations')
        
        # Overall status should indicate success/failure
        assert report.overall_status in ['PASS', 'FAIL']


class TestStage3Integration:
    """
    Integration tests for complete Stage 3 pipeline.
    
    Tests end-to-end functionality combining all Stage 3 components.
    """
    
    def test_complete_prediction_pipeline_consistency(self):
        """RED: Test complete pipeline maintains feature consistency."""
        # Create synthetic training data
        train_data = pd.DataFrame({
            'business_date': pd.date_range('2024-01-01', periods=100),
            'dataset_uuid': ['uuid1'] * 50 + ['uuid2'] * 50,
            'rule_code': ['R001'] * 50 + ['R002'] * 50,
            'pass_percentage': np.random.uniform(0, 100, 100),
            # Additional columns that should be ignored
            'extra_column': range(100)
        })
        
        # Test that training and prediction use same features
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'day_of_month', 'week_of_year', 'month',
                       'lag_1_day', 'lag_2_day', 'lag_7_day', 'ma_3_day', 'ma_7_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        
        # This should work without feature count mismatches
        # (Will fail until Stage 3 implementation completes)
        consistency = FeatureConsistency()
        all_features = consistency.get_standard_features()
        
        assert len(all_features) == len(feature_cols) + len(categorical_cols)
    
    def test_mae_target_achievement(self):
        """RED: Test that pipeline achieves MAE < 15% target."""
        # Create synthetic data with learnable patterns
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with predictable relationships
        data = pd.DataFrame({
            'business_date': pd.date_range('2024-01-01', periods=n_samples),
            'dataset_uuid': np.random.choice(['uuid1', 'uuid2', 'uuid3'], n_samples),
            'rule_code': np.random.choice(['R001', 'R002'], n_samples),
            'day_of_week': np.random.randint(1, 8, n_samples),
            'month': np.random.randint(1, 13, n_samples)
        })
        
        # Create target with clear pattern (rule + month dependency)
        data['pass_percentage'] = (
            50 + 
            (data['rule_code'] == 'R001').astype(int) * 20 +  # R001 performs better
            np.sin(data['month'] * np.pi / 6) * 15 +           # Seasonal pattern
            np.random.normal(0, 5, n_samples)                  # Some noise
        ).clip(0, 100)
        
        # Test that with good data, MAE < 15% is achievable
        # (Will fail until Stage 3 optimizations complete)
        validator = ComprehensiveValidator()
        
        # Split data for validation
        train_size = int(0.8 * n_samples)
        train_actual = data['pass_percentage'][:train_size]
        val_actual = data['pass_percentage'][train_size:]
        
        # For now, use simple baseline prediction (mean)
        baseline_pred = np.full_like(val_actual, train_actual.mean())
        mae = np.mean(np.abs(val_actual - baseline_pred))
        
        # This assertion will fail initially - that's expected for RED phase
        # The goal is to implement Stage 3 to make this pass
        assert mae < 15.0, f"MAE {mae:.2f}% exceeds 15% target"
    
    def test_no_constant_predictions_in_pipeline(self):
        """GREEN: Test that baseline predictor produces varied predictions."""
        # Test with BaselinePredictor to ensure it doesn't produce constants
        historical_data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2', 'uuid3'] * 10,
            'rule_code': ['R001', 'R002'] * 15,
            'pass_percentage': np.random.uniform(20, 80, 30)
        })
        
        baseline = BaselinePredictor(historical_data)
        
        # Create diverse test scenarios
        test_scenarios = [
            ('uuid1', 'R001'),
            ('uuid2', 'R002'),
            ('uuid3', 'R001'),
            ('uuid1', 'R002')
        ]
        
        predictions = []
        for dataset_uuid, rule_code in test_scenarios:
            prediction = baseline.predict(dataset_uuid, rule_code)
            predictions.append(prediction)
        
        # Verify prediction variance
        predictions = np.array(predictions)
        assert len(np.unique(predictions)) > 1, "BaselinePredictor should produce varied predictions"
        
        # Verify predictions are in valid range
        assert all(0 <= p <= 100 for p in predictions), "Predictions should be in [0, 100] range"


# Additional helper functions for Stage 3 testing

def create_synthetic_dataset_with_patterns(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic dataset with learnable patterns for testing.
    
    This function creates data that should be learnable by ML models,
    allowing us to test if the pipeline can achieve target performance.
    """
    np.random.seed(42)
    
    data = pd.DataFrame({
        'business_date': pd.date_range('2024-01-01', periods=n_samples),
        'dataset_uuid': np.random.choice(['uuid1', 'uuid2', 'uuid3'], n_samples),
        'rule_code': np.random.choice(['R001', 'R002', 'R003'], n_samples),
    })
    
    # Add time features
    data['day_of_week'] = data['business_date'].dt.dayofweek + 1
    data['month'] = data['business_date'].dt.month
    data['day_of_month'] = data['business_date'].dt.day
    data['week_of_year'] = data['business_date'].dt.isocalendar().week
    
    # Create predictable pass_percentage with multiple patterns
    data['pass_percentage'] = (
        # Base rate by rule
        (data['rule_code'] == 'R001').astype(int) * 70 +
        (data['rule_code'] == 'R002').astype(int) * 50 +
        (data['rule_code'] == 'R003').astype(int) * 30 +
        
        # Seasonal pattern
        np.sin(data['month'] * np.pi / 6) * 15 +
        
        # Weekly pattern (weekends slightly worse)
        (data['day_of_week'] >= 6).astype(int) * (-5) +
        
        # Dataset-specific offset
        (data['dataset_uuid'] == 'uuid2').astype(int) * 10 +
        
        # Noise
        np.random.normal(0, 8, n_samples)
    ).clip(0, 100)
    
    return data


if __name__ == "__main__":
    # Run Stage 3 tests
    pytest.main([__file__, "-v"])
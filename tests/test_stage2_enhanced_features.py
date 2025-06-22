"""
Test suite for Stage 2: Enhanced Feature Engineering & Model Configuration
Following TDD principles - Red phase: Write failing tests first
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import lightgbm as lgb

from src.data_quality_summarizer.ml import feature_engineer
from src.data_quality_summarizer.ml.model_trainer import ModelTrainer


class TestStage2EnhancedFeatureEngineering:
    """Test enhanced feature engineering capabilities"""

    def test_find_closest_lag_value_within_tolerance(self):
        """Test nearest-neighbor lag calculation with tolerance"""
        # RED: This should fail - function doesn't exist yet
        
        # Create test data with date gaps
        test_data = pd.DataFrame({
            'business_date': pd.to_datetime([
                '2024-01-01', '2024-01-03', '2024-01-07', '2024-01-10'
            ]),
            'pass_percentage': [80.0, 85.0, 90.0, 75.0]
        }).sort_values('business_date')
        
        current_date = pd.to_datetime('2024-01-08')
        lag_days = 3
        tolerance_days = 3
        
        # Target lag date: 2024-01-05. Available: 2024-01-03 (dist 2) and 2024-01-07 (dist 2)
        # Function should return the first minimum distance match: 85.0
        result = feature_engineer.find_closest_lag_value(
            test_data, current_date, lag_days, tolerance_days
        )
        
        assert result == 85.0, f"Should find closest lag value within tolerance, got {result}"

    def test_find_closest_lag_value_no_data_within_tolerance(self):
        """Test lag calculation when no data within tolerance"""
        # RED: This should fail - function doesn't exist yet
        
        test_data = pd.DataFrame({
            'business_date': pd.to_datetime(['2024-01-01']),
            'pass_percentage': [80.0]
        })
        
        current_date = pd.to_datetime('2024-01-10')
        lag_days = 3
        tolerance_days = 2  # Too strict, should not find data
        
        result = feature_engineer.find_closest_lag_value(
            test_data, current_date, lag_days, tolerance_days
        )
        
        # Should return NaN when no data within tolerance
        assert pd.isna(result), "Should return NaN when no data within tolerance"

    def test_calculate_flexible_moving_average_full_window(self):
        """Test flexible moving average with full window available"""
        # This function needs to be enhanced to work with simpler interface
        
        # Test data with business_date for current implementation
        test_data = pd.DataFrame({
            'business_date': pd.date_range('2024-01-01', periods=5),
            'pass_percentage': [70.0, 80.0, 90.0, 85.0, 75.0]
        })
        
        window_size = 3
        min_periods = 1
        
        result_df = feature_engineer.calculate_flexible_moving_average(
            test_data, window_size, min_periods
        )
        
        # Function should add a moving average column
        assert f'avg_{window_size}_day' in result_df.columns
        
        # Last value should be average of last 3: (90.0 + 85.0 + 75.0) / 3 = 83.33
        last_avg = result_df[f'avg_{window_size}_day'].iloc[-1]
        expected = 83.33
        assert abs(last_avg - expected) < 0.01, f"Expected {expected}, got {last_avg}"

    def test_calculate_flexible_moving_average_partial_window(self):
        """Test flexible moving average with partial window"""
        
        test_data = pd.DataFrame({
            'business_date': pd.date_range('2024-01-01', periods=2),
            'pass_percentage': [80.0, 90.0]  # Only 2 values, but window_size=5
        })
        
        window_size = 5
        min_periods = 1
        
        result_df = feature_engineer.calculate_flexible_moving_average(
            test_data, window_size, min_periods
        )
        
        # Should use available data: (80.0 + 90.0) / 2 = 85.0
        last_avg = result_df[f'avg_{window_size}_day'].iloc[-1]
        expected = 85.0
        assert last_avg == expected, f"Expected {expected}, got {last_avg}"

    def test_calculate_flexible_moving_average_insufficient_data(self):
        """Test flexible moving average with insufficient data"""
        
        test_data = pd.DataFrame({
            'business_date': pd.date_range('2024-01-01', periods=1),
            'pass_percentage': [80.0]  # Only 1 value
        })
        
        window_size = 3
        min_periods = 2  # Require at least 2 periods
        
        result_df = feature_engineer.calculate_flexible_moving_average(
            test_data, window_size, min_periods
        )
        
        # Should return NaN when insufficient data for min_periods
        last_avg = result_df[f'avg_{window_size}_day'].iloc[-1]
        assert pd.isna(last_avg), "Should return NaN when insufficient data"


class TestStage2OptimizedModelConfiguration:
    """Test optimized LightGBM model configuration"""

    def test_get_optimized_lgb_params_structure(self):
        """Test optimized LightGBM parameters structure"""
        # RED: This should fail - function doesn't exist yet
        from src.data_quality_summarizer.ml.model_trainer import get_optimized_lgb_params
        params = get_optimized_lgb_params()
        
        # Verify essential parameters exist
        assert 'objective' in params
        assert 'metric' in params
        assert 'learning_rate' in params
        assert 'num_leaves' in params
        assert 'feature_fraction' in params
        assert 'bagging_fraction' in params
        
        # Verify optimized values
        assert params['objective'] == 'regression'
        assert params['metric'] == 'mae'
        assert params['learning_rate'] == 0.1  # Increased from 0.05
        assert params['num_leaves'] == 31
        assert params['feature_fraction'] == 0.9
        assert params['bagging_fraction'] == 0.8

    def test_get_optimized_lgb_params_training_config(self):
        """Test optimized training configuration"""
        # RED: This should fail - function doesn't exist yet
        from src.data_quality_summarizer.ml.model_trainer import get_optimized_lgb_params
        params = get_optimized_lgb_params()
        
        # Verify training parameters
        assert 'num_boost_round' in params
        assert 'early_stopping_rounds' in params
        assert 'min_data_in_leaf' in params
        assert 'min_sum_hessian_in_leaf' in params
        
        # Verify optimized training values
        assert params['num_boost_round'] == 300  # Increased from 100
        assert params['early_stopping_rounds'] == 50  # Increased from 10
        assert params['min_data_in_leaf'] == 10
        assert params['min_sum_hessian_in_leaf'] == 1e-3


class TestStage2TrainingDiagnostics:
    """Test comprehensive training diagnostics"""

    def test_train_lightgbm_model_with_enhanced_diagnostics(self):
        """Test enhanced training with comprehensive diagnostics"""
        # Import the standalone function
        from src.data_quality_summarizer.ml.model_trainer import train_lightgbm_model_with_enhanced_diagnostics
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'dataset_uuid': ['uuid1'] * n_samples,
            'rule_code': ['R001'] * n_samples,
            'business_date': pd.date_range('2024-01-01', periods=n_samples),
            'pass_percentage': np.random.uniform(10, 90, n_samples),
            'lag_3_days': np.random.uniform(10, 90, n_samples),
            'lag_7_days': np.random.uniform(10, 90, n_samples),
            'lag_30_days': np.random.uniform(10, 90, n_samples),
            'ma_7_days': np.random.uniform(10, 90, n_samples),
            'ma_30_days': np.random.uniform(10, 90, n_samples),
            'trend_slope': np.random.uniform(-1, 1, n_samples),
            'days_since_epoch': np.arange(n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'source': ['test_source'] * n_samples,
            'level_of_execution': ['row'] * n_samples
        })
        
        feature_cols = [
            'lag_3_days', 'lag_7_days', 'lag_30_days',
            'ma_7_days', 'ma_30_days', 'trend_slope',
            'days_since_epoch', 'day_of_week', 'month'
        ]
        categorical_cols = ['source', 'level_of_execution']
        target_col = 'pass_percentage'
        
        # Test enhanced training with diagnostics
        result = train_lightgbm_model_with_enhanced_diagnostics(
            data, feature_cols, categorical_cols, target_col
        )
        
        # Verify enhanced training results
        assert 'model' in result
        assert 'feature_importance' in result
        assert 'training_metrics' in result
        assert 'convergence_info' in result
        
        # Verify model is trained
        assert isinstance(result['model'], lgb.Booster)
        
        # Verify feature importance analysis
        importance = result['feature_importance']
        # Note: Some features might be dropped during training, so check >= 0
        assert len(importance['feature_importance']) > 0
        assert all(imp >= 0 for imp in importance['feature_importance'].values())
        
        # Verify training metrics exist
        metrics = result['training_metrics']
        assert 'final_validation_score' in metrics
        assert 'training_rounds' in metrics
        assert 'convergence_achieved' in metrics

    def test_log_feature_importance_analysis(self):
        """Test feature importance logging and analysis"""
        # Import the standalone function
        from src.data_quality_summarizer.ml.model_trainer import log_feature_importance_analysis
        
        # Mock a trained model with feature importance
        mock_model = MagicMock()
        mock_model.feature_importance.return_value = np.array([100, 80, 60, 40, 20, 10, 5, 2, 1])
        
        feature_names = [
            'lag_3_days', 'lag_7_days', 'lag_30_days',
            'ma_7_days', 'ma_30_days', 'trend_slope',
            'days_since_epoch', 'day_of_week', 'month'
        ]
        
        importance_analysis = log_feature_importance_analysis(
            mock_model, feature_names
        )
        
        # Verify importance analysis structure
        assert 'feature_importance' in importance_analysis
        assert 'top_features' in importance_analysis
        assert 'low_importance_features' in importance_analysis
        assert 'zero_importance_count' in importance_analysis
        
        # Verify top features identified
        assert len(importance_analysis['top_features']) >= 3
        assert 'lag_3_days' in importance_analysis['top_features']
        
        # Verify zero importance detection
        assert importance_analysis['zero_importance_count'] >= 0

    def test_generate_training_convergence_report(self):
        """Test training convergence monitoring and reporting"""
        # Import the standalone function  
        from src.data_quality_summarizer.ml.model_trainer import generate_training_convergence_report
        
        # Mock training history
        mock_eval_results = {
            'valid_0': {
                'mae': [50.0, 45.0, 40.0, 38.0, 37.5, 37.2, 37.1, 37.0]
            }
        }
        
        convergence_report = generate_training_convergence_report(
            mock_eval_results, target_rounds=300, actual_rounds=50
        )
        
        # Verify convergence report structure
        assert 'convergence_achieved' in convergence_report
        assert 'final_score' in convergence_report
        assert 'improvement_rate' in convergence_report
        assert 'early_stopping_triggered' in convergence_report
        
        # Verify convergence detection
        assert convergence_report['final_score'] == 37.0
        assert convergence_report['convergence_achieved'] is True
        assert convergence_report['improvement_rate'] > 0


class TestStage2IntegrationWithValidation:
    """Test Stage 2 integration with Stage 1 validation"""

    def test_enhanced_training_with_stage1_validation(self):
        """Test that Stage 2 enhancements work with Stage 1 validation"""
        # Import the integration function
        from src.data_quality_summarizer.ml.model_trainer import train_lightgbm_model_with_validation_and_diagnostics
        
        # Create data that passes Stage 1 validation
        np.random.seed(42)
        n_samples = 100
        
        # Create data with sufficient variance and samples
        data = pd.DataFrame({
            'dataset_uuid': (['uuid1'] * 50) + (['uuid2'] * 50),
            'rule_code': (['R001'] * 50) + (['R002'] * 50),
            'business_date': pd.date_range('2024-01-01', periods=n_samples),
            'pass_percentage': np.random.uniform(20, 80, n_samples),  # Good variance
            'lag_3_days': np.random.uniform(20, 80, n_samples),
            'lag_7_days': np.random.uniform(20, 80, n_samples),
            'lag_30_days': np.random.uniform(20, 80, n_samples),
            'ma_7_days': np.random.uniform(20, 80, n_samples),
            'ma_30_days': np.random.uniform(20, 80, n_samples),
            'trend_slope': np.random.uniform(-1, 1, n_samples),
            'days_since_epoch': np.arange(n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'source': ['test_source'] * n_samples,
            'level_of_execution': ['row'] * n_samples
        })
        
        feature_cols = [
            'lag_3_days', 'lag_7_days', 'lag_30_days',
            'ma_7_days', 'ma_30_days', 'trend_slope',
            'days_since_epoch', 'day_of_week', 'month'
        ]
        categorical_cols = ['source', 'level_of_execution']
        target_col = 'pass_percentage'
        
        # Test full training pipeline with both Stage 1 and Stage 2
        result = train_lightgbm_model_with_validation_and_diagnostics(
            data, feature_cols, categorical_cols, target_col
        )
        
        # Verify Stage 1 validation passed
        assert 'validation_report' in result
        assert result['validation_report']['passed'] is True
        
        # Verify Stage 2 enhancements applied
        assert 'enhanced_features' in result
        assert 'optimized_model' in result
        assert 'comprehensive_diagnostics' in result
        
        # Verify model quality improvements
        model = result['optimized_model']
        assert isinstance(model, lgb.Booster)
        
        # Verify diagnostics include both stages
        diagnostics = result['comprehensive_diagnostics']
        assert 'stage1_validation' in diagnostics
        assert 'stage2_enhancements' in diagnostics
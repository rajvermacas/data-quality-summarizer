"""
Test suite for ML Evaluator module.

This module tests the model evaluation functionality
for the predictive model pipeline.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_quality_summarizer.ml.evaluator import (
    calculate_mae,
    calculate_mse,
    calculate_rmse,
    calculate_mape,
    ModelEvaluator,
    generate_evaluation_report,
    plot_predictions_vs_actual,
    calculate_residuals
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test individual evaluation metric calculations."""

    def setUp(self):
        """Create test data for evaluation tests."""
        self.actual = np.array([85.0, 90.0, 75.0, 95.0, 80.0])
        self.predicted = np.array([83.0, 92.0, 78.0, 93.0, 82.0])
        
        # Perfect predictions for edge case testing
        self.perfect_predicted = self.actual.copy()
        
        # Extreme case predictions
        self.extreme_predicted = np.array([100.0, 0.0, 50.0, 25.0, 75.0])

    def test_calculate_mae_basic(self):
        """Test basic MAE calculation."""
        mae = calculate_mae(self.actual, self.predicted)
        
        # Expected: |85-83| + |90-92| + |75-78| + |95-93| + |80-82| / 5 = 2.2
        expected_mae = 2.2
        self.assertAlmostEqual(mae, expected_mae, places=1)

    def test_calculate_mae_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        mae = calculate_mae(self.actual, self.perfect_predicted)
        self.assertEqual(mae, 0.0)

    def test_calculate_mse_basic(self):
        """Test basic MSE calculation."""
        mse = calculate_mse(self.actual, self.predicted)
        
        # Expected: (2^2 + 2^2 + 3^2 + 2^2 + 2^2) / 5 = 5.0
        expected_mse = 5.0
        self.assertAlmostEqual(mse, expected_mse, places=1)

    def test_calculate_rmse_basic(self):
        """Test basic RMSE calculation."""
        rmse = calculate_rmse(self.actual, self.predicted)
        
        # Expected: sqrt(5.0) ≈ 2.236
        expected_rmse = np.sqrt(5.0)
        self.assertAlmostEqual(rmse, expected_rmse, places=2)

    def test_calculate_mape_basic(self):
        """Test basic MAPE calculation."""
        mape = calculate_mape(self.actual, self.predicted)
        
        # Calculate expected MAPE manually
        expected_mape = np.mean(np.abs((self.actual - self.predicted) / self.actual)) * 100
        self.assertAlmostEqual(mape, expected_mape, places=2)

    def test_metric_calculations_with_zeros(self):
        """Test metric calculations when actual values contain zeros."""
        actual_with_zero = np.array([0.0, 10.0, 20.0])
        predicted_with_zero = np.array([1.0, 11.0, 19.0])
        
        # MAE and MSE should work fine
        mae = calculate_mae(actual_with_zero, predicted_with_zero)
        mse = calculate_mse(actual_with_zero, predicted_with_zero)
        
        self.assertIsInstance(mae, float)
        self.assertIsInstance(mse, float)
        
        # MAPE should handle division by zero gracefully
        mape = calculate_mape(actual_with_zero, predicted_with_zero)
        self.assertTrue(np.isfinite(mape))

    def test_empty_arrays(self):
        """Test metric calculations with empty arrays."""
        empty = np.array([])
        
        with self.assertRaises(ValueError):
            calculate_mae(empty, empty)
        
        with self.assertRaises(ValueError):
            calculate_mse(empty, empty)
        
        with self.assertRaises(ValueError):
            calculate_rmse(empty, empty)

    def test_mismatched_array_lengths(self):
        """Test metric calculations with mismatched array lengths."""
        short_array = np.array([1.0, 2.0])
        long_array = np.array([1.0, 2.0, 3.0])
        
        with self.assertRaises(ValueError):
            calculate_mae(short_array, long_array)
        
        with self.assertRaises(ValueError):
            calculate_mse(short_array, long_array)


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class functionality."""

    def setUp(self):
        """Create test data and evaluator instance."""
        # Create test data with realistic pass percentages
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        
        self.test_data = pd.DataFrame({
            'dataset_uuid': ['dataset1'] * 25 + ['dataset2'] * 25,
            'rule_code': ['R001'] * 12 + ['R002'] * 13 + ['R001'] * 12 + ['R002'] * 13,
            'business_date': dates,
            'actual_pass_percentage': np.random.uniform(70, 95, 50),
            'predicted_pass_percentage': None
        })
        
        # Add realistic predictions with some noise
        self.test_data['predicted_pass_percentage'] = (
            self.test_data['actual_pass_percentage'] + 
            np.random.normal(0, 3, 50)
        )
        
        self.evaluator = ModelEvaluator()

    def test_model_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        
        self.assertIsInstance(evaluator.metrics, dict)
        self.assertEqual(len(evaluator.metrics), 0)

    def test_evaluate_predictions_basic(self):
        """Test basic prediction evaluation."""
        metrics = self.evaluator.evaluate_predictions(
            self.test_data['actual_pass_percentage'],
            self.test_data['predicted_pass_percentage']
        )
        
        required_metrics = ['mae', 'mse', 'rmse', 'mape']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertTrue(metrics[metric] >= 0)

    def test_evaluate_dataframe_predictions(self):
        """Test evaluation using DataFrame with column names."""
        metrics = self.evaluator.evaluate_dataframe(
            self.test_data,
            actual_col='actual_pass_percentage',
            predicted_col='predicted_pass_percentage'
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertIn('count', metrics)
        self.assertEqual(metrics['count'], len(self.test_data))

    def test_evaluate_by_groups(self):
        """Test evaluation grouped by dataset and rule."""
        group_metrics = self.evaluator.evaluate_by_groups(
            self.test_data,
            group_cols=['dataset_uuid', 'rule_code'],
            actual_col='actual_pass_percentage',
            predicted_col='predicted_pass_percentage'
        )
        
        self.assertIsInstance(group_metrics, pd.DataFrame)
        self.assertIn('mae', group_metrics.columns)
        self.assertIn('count', group_metrics.columns)
        
        # Should have metrics for each unique group combination
        expected_groups = self.test_data.groupby(['dataset_uuid', 'rule_code']).ngroups
        self.assertEqual(len(group_metrics), expected_groups)

    def test_calculate_residuals_method(self):
        """Test residual calculation method."""
        residuals = self.evaluator.calculate_residuals(
            self.test_data['actual_pass_percentage'],
            self.test_data['predicted_pass_percentage']
        )
        
        self.assertEqual(len(residuals), len(self.test_data))
        
        # Residuals should be actual - predicted
        expected_residuals = (
            self.test_data['actual_pass_percentage'] - 
            self.test_data['predicted_pass_percentage']
        )
        np.testing.assert_array_almost_equal(residuals, expected_residuals, decimal=5)

    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        # First evaluate some predictions
        self.evaluator.evaluate_predictions(
            self.test_data['actual_pass_percentage'],
            self.test_data['predicted_pass_percentage']
        )
        
        summary = self.evaluator.get_metrics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('mae', summary)
        self.assertIn('evaluation_date', summary)


class TestEvaluationReporting(unittest.TestCase):
    """Test evaluation reporting and visualization functions."""

    def setUp(self):
        """Create test data for reporting tests."""
        np.random.seed(42)  # For reproducible results
        
        self.actual = np.random.uniform(70, 95, 100)
        self.predicted = self.actual + np.random.normal(0, 5, 100)
        
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        self.evaluation_data = pd.DataFrame({
            'actual': self.actual,
            'predicted': self.predicted,
            'business_date': dates,
            'dataset_uuid': np.random.choice(['dataset1', 'dataset2'], 100),
            'rule_code': np.random.choice(['R001', 'R002', 'R003'], 100)
        })

    def test_generate_evaluation_report(self):
        """Test comprehensive evaluation report generation."""
        report = generate_evaluation_report(
            self.evaluation_data['actual'],
            self.evaluation_data['predicted']
        )
        
        self.assertIsInstance(report, dict)
        
        # Check required sections
        self.assertIn('metrics', report)
        self.assertIn('summary', report)
        self.assertIn('distribution_stats', report)
        
        # Check metrics
        metrics = report['metrics']
        required_metrics = ['mae', 'mse', 'rmse', 'mape']
        for metric in required_metrics:
            self.assertIn(metric, metrics)

    def test_calculate_residuals_function(self):
        """Test standalone residuals calculation function."""
        residuals = calculate_residuals(self.actual, self.predicted)
        
        self.assertEqual(len(residuals), len(self.actual))
        
        # Check that residuals are correct
        expected_residuals = self.actual - self.predicted
        np.testing.assert_array_almost_equal(residuals, expected_residuals)

    def test_plot_predictions_vs_actual_returns_data(self):
        """Test that plot function returns plot data even without display."""
        plot_data = plot_predictions_vs_actual(
            self.actual, self.predicted, show_plot=False
        )
        
        self.assertIsInstance(plot_data, dict)
        self.assertIn('actual', plot_data)
        self.assertIn('predicted', plot_data)
        self.assertEqual(len(plot_data['actual']), len(self.actual))
        self.assertEqual(len(plot_data['predicted']), len(self.predicted))


class TestEvaluatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for evaluator."""

    def test_evaluate_single_prediction(self):
        """Test evaluation with single prediction."""
        evaluator = ModelEvaluator()
        
        actual = np.array([85.0])
        predicted = np.array([83.0])
        
        metrics = evaluator.evaluate_predictions(actual, predicted)
        
        self.assertIn('mae', metrics)
        self.assertEqual(metrics['mae'], 2.0)

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        evaluator = ModelEvaluator()
        
        actual = np.array([85.0, 90.0, 75.0])
        predicted = actual.copy()
        
        metrics = evaluator.evaluate_predictions(actual, predicted)
        
        self.assertEqual(metrics['mae'], 0.0)
        self.assertEqual(metrics['mse'], 0.0)
        self.assertEqual(metrics['rmse'], 0.0)

    def test_evaluate_with_nan_values(self):
        """Test evaluation handling of NaN values."""
        evaluator = ModelEvaluator()
        
        actual = np.array([85.0, np.nan, 75.0])
        predicted = np.array([83.0, 88.0, 78.0])
        
        # Should handle NaN values gracefully
        metrics = evaluator.evaluate_predictions(actual, predicted)
        
        # Should only evaluate non-NaN values
        self.assertIsInstance(metrics['mae'], (int, float))
        self.assertTrue(np.isfinite(metrics['mae']))

    def test_dataframe_evaluation_missing_columns(self):
        """Test DataFrame evaluation with missing columns."""
        evaluator = ModelEvaluator()
        
        data = pd.DataFrame({
            'actual': [85.0, 90.0],
            # Missing 'predicted' column
        })
        
        with self.assertRaises(KeyError):
            evaluator.evaluate_dataframe(
                data, actual_col='actual', predicted_col='predicted'
            )


if __name__ == '__main__':
    unittest.main()
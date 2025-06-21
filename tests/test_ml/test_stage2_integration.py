"""
Integration test suite for Stage 2: Model Training Infrastructure.

This module tests the complete integration of data splitting,
model training, and evaluation components.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta

from src.data_quality_summarizer.ml.data_splitter import (
    split_data_chronologically, 
    determine_optimal_cutoff_date
)
from src.data_quality_summarizer.ml.model_trainer import ModelTrainer
from src.data_quality_summarizer.ml.evaluator import ModelEvaluator
from src.data_quality_summarizer.ml.data_loader import (
    load_and_validate_csv, parse_results_column, create_binary_pass_column
)
from src.data_quality_summarizer.ml.aggregator import aggregate_pass_percentages
from src.data_quality_summarizer.ml.feature_engineer import engineer_all_features


class TestStage2Integration(unittest.TestCase):
    """Test complete Stage 2 pipeline integration."""

    def setUp(self):
        """Create comprehensive test dataset for Stage 2 integration."""
        np.random.seed(42)  # For reproducible results
        
        # Create 6 months of data with realistic patterns
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(180)]
        
        # Create dataset with multiple groups for realistic testing
        datasets = ['dataset1', 'dataset2', 'dataset3']
        rules = ['R001', 'R002', 'R003']
        
        data_rows = []
        for date in dates:
            for dataset in datasets:
                for rule in rules:
                    # Create multiple executions per day
                    num_executions = np.random.randint(5, 15)
                    for _ in range(num_executions):
                        # Create realistic pass patterns with some seasonality
                        base_pass_rate = 0.85
                        seasonal_effect = 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                        dataset_effect = {'dataset1': 0.05, 'dataset2': 0, 'dataset3': -0.03}[dataset]
                        rule_effect = {'R001': 0.02, 'R002': 0, 'R003': -0.02}[rule]
                        
                        pass_probability = base_pass_rate + seasonal_effect + dataset_effect + rule_effect
                        pass_probability = np.clip(pass_probability, 0.1, 0.95)
                        
                        is_pass = np.random.random() < pass_probability
                        status = "Pass" if is_pass else "Fail"
                        
                        data_rows.append({
                            'source': 'test_source',
                            'tenant_id': 'test_tenant',
                            'dataset_uuid': dataset,
                            'dataset_name': f'{dataset}_name',
                            'rule_code': rule,
                            'business_date': date.strftime('%Y-%m-%d'),
                            'results': f'{{"status": "{status}"}}',
                            'level_of_execution': 'dataset',
                            'attribute_name': 'test_attr'
                        })
        
        self.raw_data = pd.DataFrame(data_rows)
        logger_name = f"Created {len(self.raw_data)} rows of test data"

    def test_complete_stage2_pipeline(self):
        """Test complete Stage 2 pipeline from raw data to evaluation."""
        
        # Step 1: Data preparation (using Stage 1 components)
        raw_data = self.raw_data.copy()
        
        # Parse results column and create binary pass column
        parsed_results = parse_results_column(raw_data['results'].tolist())
        raw_data['is_pass'] = create_binary_pass_column(parsed_results)
        
        # Aggregate pass percentages
        aggregated_data = aggregate_pass_percentages(raw_data)
        featured_data = engineer_all_features(aggregated_data)
        
        # Verify we have sufficient data
        self.assertGreater(len(featured_data), 100, "Need sufficient data for training")
        
        # Step 2: Chronological data splitting
        cutoff_date = determine_optimal_cutoff_date(featured_data, train_ratio=0.7)
        train_data, test_data = split_data_chronologically(featured_data, cutoff_date)
        
        # Verify split worked correctly
        self.assertGreater(len(train_data), 0, "Training data should not be empty")
        self.assertGreater(len(test_data), 0, "Test data should not be empty")
        
        train_ratio = len(train_data) / len(featured_data)
        self.assertAlmostEqual(train_ratio, 0.7, delta=0.15, 
                              msg="Train ratio should be approximately 70%")
        
        # Step 3: Model training
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'month', 'lag_1_day', 'lag_7_day', 'ma_3_day', 'ma_7_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        # Filter features that exist in the data
        available_features = [col for col in feature_cols if col in train_data.columns]
        self.assertGreater(len(available_features), 0, "Should have at least some features")
        
        trained_model = trainer.fit(
            train_data, available_features, categorical_cols, target_col
        )
        
        self.assertIsNotNone(trained_model, "Model should be trained successfully")
        
        # Step 4: Model prediction
        train_predictions = trainer.predict(train_data)
        test_predictions = trainer.predict(test_data)
        
        self.assertEqual(len(train_predictions), len(train_data))
        self.assertEqual(len(test_predictions), len(test_data))
        
        # All predictions should be in valid range (0-100)
        self.assertTrue(all(0 <= p <= 100 for p in train_predictions))
        self.assertTrue(all(0 <= p <= 100 for p in test_predictions))
        
        # Step 5: Model evaluation
        evaluator = ModelEvaluator()
        
        # Evaluate training performance
        train_metrics = evaluator.evaluate_predictions(
            train_data[target_col].values, train_predictions
        )
        
        # Evaluate test performance
        test_metrics = evaluator.evaluate_predictions(
            test_data[target_col].values, test_predictions
        )
        
        # Verify metrics are reasonable
        required_metrics = ['mae', 'mse', 'rmse', 'mape']
        for metric in required_metrics:
            self.assertIn(metric, train_metrics)
            self.assertIn(metric, test_metrics)
            self.assertGreater(train_metrics[metric], 0)
            self.assertGreater(test_metrics[metric], 0)
        
        # Training error should generally be lower than test error
        # But we won't enforce this strictly due to randomness
        self.assertLess(train_metrics['mae'], 50, "Training MAE should be reasonable")
        self.assertLess(test_metrics['mae'], 50, "Test MAE should be reasonable")
        
        # Step 6: Group-based evaluation
        test_data_with_predictions = test_data.copy()
        test_data_with_predictions['predicted_pass_percentage'] = test_predictions
        
        group_metrics = evaluator.evaluate_by_groups(
            test_data_with_predictions,
            group_cols=['dataset_uuid', 'rule_code'],
            actual_col=target_col,
            predicted_col='predicted_pass_percentage'
        )
        
        self.assertGreater(len(group_metrics), 0, "Should have group-level metrics")
        self.assertIn('mae', group_metrics.columns)
        
        # Log final results
        print(f"\n=== Stage 2 Integration Test Results ===")
        print(f"Training data: {len(train_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        print(f"Training MAE: {train_metrics['mae']:.2f}")
        print(f"Test MAE: {test_metrics['mae']:.2f}")
        print(f"Group evaluations: {len(group_metrics)}")

    def test_model_serialization_in_pipeline(self):
        """Test model save/load functionality within complete pipeline."""
        
        # Prepare minimal dataset
        raw_data = self.raw_data.copy()
        parsed_results = parse_results_column(raw_data['results'].tolist())
        raw_data['is_pass'] = create_binary_pass_column(parsed_results)
        aggregated_data = aggregate_pass_percentages(raw_data)
        featured_data = engineer_all_features(aggregated_data)
        
        # Train model
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'month']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        available_features = [col for col in feature_cols if col in featured_data.columns]
        trainer.fit(featured_data, available_features, categorical_cols, target_col)
        
        original_predictions = trainer.predict(featured_data)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            from src.data_quality_summarizer.ml.model_trainer import save_model, load_model
            
            save_model(trainer.model, model_path)
            loaded_model = load_model(model_path)
            
            # Create new trainer with loaded model
            new_trainer = ModelTrainer()
            new_trainer.model = loaded_model
            new_trainer.feature_cols = trainer.feature_cols
            new_trainer.categorical_cols = trainer.categorical_cols
            
            loaded_predictions = new_trainer.predict(featured_data)
            
            # Predictions should be identical
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=5
            )
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_pipeline_performance_requirements(self):
        """Test that Stage 2 pipeline meets performance requirements."""
        import time
        
        # Use smaller dataset for performance testing
        small_data = self.raw_data.head(1000).copy()
        
        start_time = time.time()
        
        # Run complete pipeline
        parsed_results = parse_results_column(small_data['results'].tolist())
        small_data['is_pass'] = create_binary_pass_column(parsed_results)
        aggregated_data = aggregate_pass_percentages(small_data)
        featured_data = engineer_all_features(aggregated_data)
        
        cutoff_date = determine_optimal_cutoff_date(featured_data, train_ratio=0.8)
        train_data, test_data = split_data_chronologically(featured_data, cutoff_date)
        
        # Train model if we have sufficient data
        if len(train_data) > 10:
            trainer = ModelTrainer()
            feature_cols = ['day_of_week', 'month']
            categorical_cols = ['dataset_uuid', 'rule_code']
            target_col = 'pass_percentage'
            
            available_features = [col for col in feature_cols if col in train_data.columns]
            if available_features:
                trainer.fit(train_data, available_features, categorical_cols, target_col)
                predictions = trainer.predict(test_data)
                
                evaluator = ModelEvaluator()
                metrics = evaluator.evaluate_predictions(
                    test_data[target_col].values, predictions
                )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance requirements: Should complete in reasonable time
        self.assertLess(processing_time, 30, 
                       f"Pipeline took {processing_time:.2f}s, should be <30s for 1k rows")
        
        print(f"\nStage 2 Performance: {processing_time:.2f}s for {len(small_data)} rows")

    def test_pipeline_with_edge_cases(self):
        """Test Stage 2 pipeline with edge cases and challenging data."""
        
        # Create edge case data
        edge_case_data = pd.DataFrame([
            {
                'source': 'test', 'tenant_id': 'test', 'dataset_uuid': 'single_dataset',
                'dataset_name': 'test', 'rule_code': 'single_rule', 
                'business_date': '2024-01-01', 'results': '{"status": "Pass"}',
                'level_of_execution': 'dataset', 'attribute_name': 'test'
            },
            {
                'source': 'test', 'tenant_id': 'test', 'dataset_uuid': 'single_dataset',
                'dataset_name': 'test', 'rule_code': 'single_rule', 
                'business_date': '2024-01-02', 'results': '{"status": "Fail"}',
                'level_of_execution': 'dataset', 'attribute_name': 'test'
            }
        ])
        
        # Should handle minimal data gracefully
        parsed_results = parse_results_column(edge_case_data['results'].tolist())
        edge_case_data['is_pass'] = create_binary_pass_column(parsed_results)
        aggregated_data = aggregate_pass_percentages(edge_case_data)
        featured_data = engineer_all_features(aggregated_data)
        
        # Should have some features even with minimal data
        self.assertGreater(len(featured_data.columns), 5)
        
        # Data splitting should work even with minimal data
        cutoff_date = determine_optimal_cutoff_date(featured_data, train_ratio=0.5)
        train_data, test_data = split_data_chronologically(featured_data, cutoff_date)
        
        # At least one of train/test should have data
        total_data = len(train_data) + len(test_data)
        self.assertEqual(total_data, len(featured_data))


class TestStage2ComponentInteraction(unittest.TestCase):
    """Test specific interactions between Stage 2 components."""

    def test_data_splitter_preserves_feature_columns(self):
        """Test that data splitting preserves all feature columns."""
        # Create sample featured data
        data = pd.DataFrame({
            'dataset_uuid': ['d1', 'd1', 'd2', 'd2'],
            'rule_code': ['R1', 'R1', 'R2', 'R2'],
            'business_date': [
                datetime(2024, 1, 1), datetime(2024, 1, 2),
                datetime(2024, 1, 3), datetime(2024, 1, 4)
            ],
            'pass_percentage': [85.0, 90.0, 75.0, 80.0],
            'day_of_week': [0, 1, 2, 3],
            'month': [1, 1, 1, 1],
            'lag_1_day': [82.0, 85.0, 78.0, 75.0],
            'ma_3_day': [83.0, 87.0, 76.5, 77.5]
        })
        
        cutoff_date = datetime(2024, 1, 3)
        train_data, test_data = split_data_chronologically(data, cutoff_date)
        
        # All feature columns should be preserved
        self.assertEqual(list(train_data.columns), list(data.columns))
        self.assertEqual(list(test_data.columns), list(data.columns))

    def test_model_trainer_evaluator_consistency(self):
        """Test consistency between model trainer and evaluator."""
        # Create simple test data
        train_data = pd.DataFrame({
            'dataset_uuid': ['d1'] * 10,
            'rule_code': ['R1'] * 10,
            'pass_percentage': np.random.uniform(70, 95, 10),
            'feature1': np.random.uniform(0, 1, 10)
        })
        
        # Train model
        trainer = ModelTrainer()
        trainer.fit(train_data, ['feature1'], ['dataset_uuid', 'rule_code'], 'pass_percentage')
        
        # Make predictions
        predictions = trainer.predict(train_data)
        
        # Evaluate predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_predictions(
            train_data['pass_percentage'].values, predictions
        )
        
        # Metrics should be reasonable
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['mae'], 0)
        self.assertEqual(metrics['count'], len(train_data))


if __name__ == '__main__':
    unittest.main()
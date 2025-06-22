"""
Stage 3 Advanced ML Features Tests - TDD Implementation.

Tests for the Stage 3 advanced features:
- US3.1: Model Versioning System with comparison capabilities
- US3.2: Hyperparameter Optimization framework  
- US3.3: A/B Testing framework

Following strict TDD Red-Green-Refactor methodology.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.data_quality_summarizer.ml.production import ProductionUtils


class TestAdvancedModelRegistry:
    """
    Test enhanced model registry with comparison capabilities.
    
    US3.1: Model Versioning System
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.production = ProductionUtils(base_path=self.temp_dir)
        
        # Create simple serializable models for testing
        import lightgbm as lgb
        
        # Create minimal datasets for training simple models
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [85.0, 87.0, 89.0, 91.0, 93.0]
        })
        
        # Train basic LightGBM models
        train_x = train_data[['feature1', 'feature2']]
        train_y = train_data['target']
        
        lgb_train1 = lgb.Dataset(train_x, train_y)
        self.model_v1 = lgb.train({'objective': 'regression', 'verbose': -1}, lgb_train1, num_boost_round=1)
        
        lgb_train2 = lgb.Dataset(train_x, train_y)  
        self.model_v2 = lgb.train({'objective': 'regression', 'verbose': -1}, lgb_train2, num_boost_round=2)
        
        # Sample metadata for models
        self.metadata_v1 = {
            'performance_metrics': {
                'mae': 15.2,
                'rmse': 20.1,
                'r2': 0.75
            },
            'features': ['feature1', 'feature2', 'feature3'],
            'training_data_size': 1000,
            'created_by': 'test_user'
        }
        
        self.metadata_v2 = {
            'performance_metrics': {
                'mae': 12.8,
                'rmse': 18.3,
                'r2': 0.82
            },
            'features': ['feature1', 'feature2', 'feature3', 'feature4'],
            'training_data_size': 1500,
            'created_by': 'test_user'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compare_models_performance_v1_vs_v2(self):
        """
        Test model comparison between versions.
        
        RED: This should fail initially as compare_models method doesn't exist.
        """
        # Save two model versions
        version_id_v1 = self.production.save_model_version(
            self.model_v1, self.metadata_v1, version="1.0.0"
        )
        version_id_v2 = self.production.save_model_version(
            self.model_v2, self.metadata_v2, version="1.1.0"
        )
        
        # Compare models - this should work after implementation
        comparison = self.production.compare_models(version_id_v1, version_id_v2)
        
        # Expected comparison structure
        assert 'model_a' in comparison
        assert 'model_b' in comparison
        assert 'performance_comparison' in comparison
        assert 'recommendation' in comparison
        
        # Verify performance comparison
        perf_comp = comparison['performance_comparison']
        assert perf_comp['mae']['model_a'] == 15.2
        assert perf_comp['mae']['model_b'] == 12.8
        assert perf_comp['mae']['winner'] == 'model_b'
        
        # Should recommend the better performing model
        assert comparison['recommendation'] == 'model_b'
    
    def test_promote_model_to_production(self):
        """
        Test model promotion to production environment.
        
        RED: This should fail as promote_model method doesn't exist.
        """
        # Save a model version
        version_id = self.production.save_model_version(
            self.model_v1, self.metadata_v1, version="1.2.0"
        )
        
        # Promote to production
        success = self.production.promote_model(version_id, "production")
        
        assert success is True
        
        # Verify promoted model is marked in registry
        registry = self.production.get_model_registry()
        assert registry[version_id]['environment'] == 'production'
        assert 'promoted_at' in registry[version_id]
    
    def test_get_models_by_semantic_version(self):
        """
        Test retrieving models by semantic version pattern.
        
        RED: This should fail as get_models_by_version method doesn't exist.
        """
        # Save multiple versions
        v1_id = self.production.save_model_version(
            self.model_v1, self.metadata_v1, version="1.0.0"
        )
        v2_id = self.production.save_model_version(
            self.model_v2, self.metadata_v2, version="1.1.0"
        )
        v3_id = self.production.save_model_version(
            self.model_v1, self.metadata_v1, version="2.0.0"
        )
        
        # Get all v1.x versions
        v1_models = self.production.get_models_by_version("1.*")
        assert len(v1_models) == 2
        assert v1_id in [m['version_id'] for m in v1_models]
        assert v2_id in [m['version_id'] for m in v1_models]
        assert v3_id not in [m['version_id'] for m in v1_models]


class TestHyperparameterOptimization:
    """
    Test hyperparameter optimization framework.
    
    US3.2: Hyperparameter Optimization
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample training data
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'dataset_uuid': np.random.choice(['ds1', 'ds2'], 100),
            'rule_code': np.random.choice(['R1', 'R2'], 100),  
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.uniform(0, 1, 100),
            'pass_percentage': np.random.uniform(60, 95, 100)
        })
        
        # Define parameter space
        self.param_space = {
            'num_leaves': [10, 20, 31, 50],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'feature_fraction': [0.8, 0.9, 1.0],
            'n_estimators': [50, 100, 200]
        }
    
    def test_hyperparameter_optimization_basic(self):
        """
        Test basic hyperparameter optimization.
        
        RED: This should fail as HyperparameterOptimizer doesn't exist yet.
        """
        from src.data_quality_summarizer.ml.hyperparameter_optimization import HyperparameterOptimizer
        
        optimizer = HyperparameterOptimizer()
        
        # Run optimization
        result = optimizer.optimize(
            data=self.train_data,
            param_space=self.param_space,
            cv_folds=3
        )
        
        # Verify optimization results
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'cv_results' in result
        assert 'optimization_time' in result
        
        # Best params should be from param space
        for param, value in result['best_params'].items():
            assert value in self.param_space[param]
        
        # Score should be reasonable
        assert 0 <= result['best_score'] <= 100
    
    def test_bayesian_optimization(self):
        """
        Test Bayesian optimization for efficient parameter search.
        
        RED: This should fail as bayesian_search method doesn't exist.
        """
        from src.data_quality_summarizer.ml.hyperparameter_optimization import HyperparameterOptimizer
        
        optimizer = HyperparameterOptimizer()
        
        # Run Bayesian optimization
        result = optimizer.bayesian_search(
            data=self.train_data,
            param_space=self.param_space,
            n_trials=10
        )
        
        # Verify results structure
        assert 'best_params' in result
        assert 'best_score' in result  
        assert 'trials_history' in result
        assert len(result['trials_history']) == 10
        
        # Each trial should have params and score
        for trial in result['trials_history']:
            assert 'params' in trial
            assert 'score' in trial
    
    def test_optimization_with_time_constraints(self):
        """
        Test optimization with time limits.
        
        RED: This should fail as optimize method doesn't support time_limit.
        """
        from src.data_quality_summarizer.ml.hyperparameter_optimization import HyperparameterOptimizer
        
        optimizer = HyperparameterOptimizer()
        
        # Create a larger parameter space to ensure time limit is hit
        large_param_space = {
            'num_leaves': [5, 10, 15, 20, 25, 31, 40, 50, 60, 70],
            'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
            'n_estimators': [25, 50, 75, 100, 150, 200]
        }
        
        # Run with very short time constraint
        result = optimizer.optimize(
            data=self.train_data,
            param_space=large_param_space,
            cv_folds=3,
            time_limit_minutes=0.1  # 6 seconds - very short
        )
        
        # Should complete within time limit
        assert result['optimization_time'] <= 20  # 6 seconds + buffer
        assert 'best_params' in result
        
        # Verify that not all combinations were tested (indicating early stop)
        total_combinations = 10 * 8 * 5 * 6  # 2400 combinations
        assert result['total_evaluations'] < total_combinations


class TestABTestingFramework:
    """
    Test A/B testing framework for model comparison.
    
    US3.3: A/B Testing Framework  
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        import shutil
        
        # Create temporary directory for A/B testing storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create control and treatment models
        self.control_model = Mock()
        self.control_model.predict = Mock(return_value=np.array([85.0]))
        
        self.treatment_model = Mock()  
        self.treatment_model.predict = Mock(return_value=np.array([87.0]))
        
        # Sample prediction requests
        self.prediction_requests = [
            {'dataset_uuid': 'ds1', 'rule_code': 'R1', 'date': '2024-01-01'},
            {'dataset_uuid': 'ds1', 'rule_code': 'R2', 'date': '2024-01-01'},
            {'dataset_uuid': 'ds2', 'rule_code': 'R1', 'date': '2024-01-01'},
        ]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_ab_experiment(self):
        """
        Test creating A/B test experiment.
        
        RED: This should fail as ABTestingService doesn't exist.
        """
        from src.data_quality_summarizer.ml.ab_testing import ABTestingService
        
        ab_service = ABTestingService(storage_path=self.temp_dir)
        
        # Create experiment
        experiment_id = ab_service.create_experiment(
            control_model_version="1.0.0",
            treatment_model_version="1.1.0", 
            traffic_split=0.5,
            experiment_name="Model v1.1 Performance Test"
        )
        
        # Verify experiment creation
        assert experiment_id is not None
        assert isinstance(experiment_id, str)
        
        # Get experiment details
        experiment = ab_service.get_experiment(experiment_id)
        assert experiment['control_model'] == "1.0.0"
        assert experiment['treatment_model'] == "1.1.0"
        assert experiment['traffic_split'] == 0.5
        assert experiment['status'] == 'created'
    
    def test_traffic_splitting(self):
        """
        Test traffic splitting between control and treatment.
        
        RED: This should fail as route_prediction method doesn't exist.
        """
        from src.data_quality_summarizer.ml.ab_testing import ABTestingService
        
        ab_service = ABTestingService(storage_path=self.temp_dir)
        
        # Create experiment with 30% treatment traffic
        experiment_id = ab_service.create_experiment(
            control_model_version="1.0.0",
            treatment_model_version="1.1.0",
            traffic_split=0.3
        )
        
        # Start the experiment to enable routing
        ab_service.start_experiment(experiment_id)
        
        # Route many predictions and check distribution
        routes = []
        for i in range(1000):
            route = ab_service.route_prediction(
                experiment_id=experiment_id,
                user_id=f"user_{i}"
            )
            routes.append(route)
        
        # Check traffic split is roughly correct (within 5%)
        treatment_ratio = sum(1 for r in routes if r == 'treatment') / len(routes)
        assert 0.25 <= treatment_ratio <= 0.35
    
    def test_statistical_significance_evaluation(self):
        """
        Test statistical significance evaluation of A/B test results.
        
        RED: This should fail as evaluate_experiment method doesn't exist.
        """
        from src.data_quality_summarizer.ml.ab_testing import ABTestingService
        
        ab_service = ABTestingService(storage_path=self.temp_dir)
        
        # Create experiment
        experiment_id = ab_service.create_experiment(
            control_model_version="1.0.0",
            treatment_model_version="1.1.0",
            traffic_split=0.5
        )
        
        # Simulate experiment results
        control_results = np.random.normal(85, 5, 1000)  # Control mean: 85
        treatment_results = np.random.normal(87, 5, 1000)  # Treatment mean: 87 (better)
        
        # Record results
        ab_service.record_results(experiment_id, 'control', control_results)
        ab_service.record_results(experiment_id, 'treatment', treatment_results)
        
        # Evaluate experiment
        evaluation = ab_service.evaluate_experiment(experiment_id)
        
        # Check evaluation structure
        assert 'control_mean' in evaluation
        assert 'treatment_mean' in evaluation
        assert 'p_value' in evaluation
        assert 'effect_size' in evaluation
        assert 'significant' in evaluation
        assert 'confidence_interval' in evaluation
        
        # Should detect significant improvement
        assert evaluation['significant'] == True
        assert evaluation['treatment_mean'] > evaluation['control_mean']
        assert evaluation['p_value'] < 0.05
    
    def test_experiment_lifecycle_management(self):
        """
        Test complete experiment lifecycle management.
        
        RED: This should fail as experiment management methods don't exist.
        """
        from src.data_quality_summarizer.ml.ab_testing import ABTestingService
        
        ab_service = ABTestingService(storage_path=self.temp_dir)
        
        # Create experiment
        experiment_id = ab_service.create_experiment(
            control_model_version="1.0.0",
            treatment_model_version="1.1.0",
            traffic_split=0.5,
            duration_days=7
        )
        
        # Start experiment
        ab_service.start_experiment(experiment_id)
        experiment = ab_service.get_experiment(experiment_id)
        assert experiment['status'] == 'running'
        assert 'started_at' in experiment
        
        # Stop experiment
        ab_service.stop_experiment(experiment_id)
        experiment = ab_service.get_experiment(experiment_id)
        assert experiment['status'] == 'stopped'
        assert 'stopped_at' in experiment
        
        # Archive experiment
        ab_service.archive_experiment(experiment_id)
        experiment = ab_service.get_experiment(experiment_id)
        assert experiment['status'] == 'archived'
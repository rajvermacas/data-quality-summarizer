"""
Tests for Production Utilities module.

This module tests model versioning and management, health checks and monitoring,
and configuration management capabilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.data_quality_summarizer.ml.production import ProductionUtils


class TestProductionUtils:
    """Test cases for ProductionUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.production_utils = ProductionUtils(base_path=self.temp_dir)
        
        # Create a simple picklable model for testing
        self.mock_model = {'model_type': 'test', 'predict_values': [85.5, 72.3, 91.2]}
        
        # Sample model metadata
        self.model_metadata = {
            'version': '1.0.0',
            'created_at': '2024-01-15T10:00:00Z',
            'training_data_size': 10000,
            'performance_metrics': {'mae': 3.5, 'rmse': 4.2, 'r2': 0.85},
            'feature_columns': ['pass_percentage', 'lag_1_day', 'ma_3_day'],
            'model_type': 'LightGBM'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_production_utils_initialization(self):
        """Test that production utils initializes correctly."""
        assert self.production_utils is not None
        assert hasattr(self.production_utils, 'save_model_version')
        assert hasattr(self.production_utils, 'load_model_version')
        assert hasattr(self.production_utils, 'health_check')
        assert hasattr(self.production_utils, 'get_model_registry')
    
    def test_model_versioning_save_and_load(self):
        """Test model versioning save and load functionality."""
        version_id = self.production_utils.save_model_version(
            self.mock_model,
            self.model_metadata,
            version='1.0.0'
        )
        
        assert version_id is not None
        assert isinstance(version_id, str)
        
        # Load the model back
        loaded_model, loaded_metadata = self.production_utils.load_model_version(version_id)
        
        assert loaded_model is not None
        assert loaded_metadata['version'] == '1.0.0'
        assert loaded_metadata['model_type'] == 'LightGBM'
    
    def test_model_registry_tracks_versions(self):
        """Test that model registry tracks all versions."""
        # Save multiple model versions
        version_ids = []
        for i, version in enumerate(['1.0.0', '1.1.0', '2.0.0']):
            metadata = self.model_metadata.copy()
            metadata['version'] = version
            metadata['performance_metrics']['mae'] = 3.5 - (i * 0.5)  # Improving performance
            
            version_id = self.production_utils.save_model_version(
                self.mock_model,
                metadata,
                version=version
            )
            version_ids.append(version_id)
        
        registry = self.production_utils.get_model_registry()
        
        assert len(registry) == 3
        assert all(vid in registry for vid in version_ids)
        
        # Check latest version tracking
        latest_version = self.production_utils.get_latest_model_version()
        assert latest_version['metadata']['version'] == '2.0.0'
    
    def test_health_check_reports_system_status(self):
        """Test that health check reports comprehensive system status."""
        # Save a model first
        version_id = self.production_utils.save_model_version(
            self.mock_model,
            self.model_metadata
        )
        
        health_status = self.production_utils.health_check()
        
        assert 'status' in health_status
        assert 'timestamp' in health_status
        assert 'model_status' in health_status
        assert 'system_metrics' in health_status
        assert 'last_prediction_time' in health_status
        
        assert health_status['status'] in ['HEALTHY', 'WARNING', 'CRITICAL']
        assert health_status['model_status']['models_available'] == 1
    
    def test_health_check_detects_issues(self):
        """Test that health check detects system issues."""
        # Create scenario with no models available
        health_status = self.production_utils.health_check()
        
        assert health_status['model_status']['models_available'] == 0
        assert health_status['status'] in ['WARNING', 'CRITICAL']
    
    def test_configuration_management(self):
        """Test configuration management capabilities."""
        config = {
            'model_refresh_interval': 3600,  # 1 hour
            'performance_threshold': 5.0,
            'drift_detection_enabled': True,
            'monitoring_interval': 300,  # 5 minutes
            'max_model_versions': 10
        }
        
        self.production_utils.save_configuration(config)
        loaded_config = self.production_utils.load_configuration()
        
        assert loaded_config == config
        assert loaded_config['model_refresh_interval'] == 3600
        assert loaded_config['drift_detection_enabled'] is True
    
    def test_model_cleanup_removes_old_versions(self):
        """Test that model cleanup removes old versions when limit exceeded."""
        # Configure max versions
        config = {'max_model_versions': 3}
        self.production_utils.save_configuration(config)
        
        # Save more models than the limit
        for i in range(5):
            metadata = self.model_metadata.copy()
            metadata['version'] = f'1.{i}.0'
            metadata['created_at'] = (datetime.now() - timedelta(days=i)).isoformat()
            
            self.production_utils.save_model_version(
                self.mock_model,
                metadata,
                version=f'1.{i}.0'
            )
        
        # Trigger cleanup
        self.production_utils.cleanup_old_models()
        
        registry = self.production_utils.get_model_registry()
        assert len(registry) <= 3  # Should keep only latest 3
    
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        # Save a model
        version_id = self.production_utils.save_model_version(
            self.mock_model,
            self.model_metadata
        )
        
        # Record some predictions
        predictions = [85.5, 72.3, 91.2]
        actuals = [83.0, 74.0, 89.0]
        
        self.production_utils.record_prediction_performance(
            version_id,
            predictions,
            actuals,
            timestamp='2024-01-15T10:30:00Z'
        )
        
        performance_history = self.production_utils.get_performance_history(version_id)
        
        assert len(performance_history) == 1
        assert '2024-01-15T10:30:00Z' in performance_history
        assert 'mae' in performance_history['2024-01-15T10:30:00Z']
    
    def test_model_comparison_capabilities(self):
        """Test model comparison capabilities."""
        # Save two model versions with different performance
        version_1_metadata = self.model_metadata.copy()
        version_1_metadata['version'] = '1.0.0'
        version_1_metadata['performance_metrics']['mae'] = 4.5
        
        version_2_metadata = self.model_metadata.copy()
        version_2_metadata['version'] = '2.0.0'
        version_2_metadata['performance_metrics']['mae'] = 3.2
        
        v1_id = self.production_utils.save_model_version(
            self.mock_model, version_1_metadata, version='1.0.0'
        )
        v2_id = self.production_utils.save_model_version(
            self.mock_model, version_2_metadata, version='2.0.0'
        )
        
        comparison = self.production_utils.compare_model_versions(v1_id, v2_id)
        
        assert 'performance_improvement' in comparison
        assert 'version_1' in comparison
        assert 'version_2' in comparison
        assert comparison['performance_improvement']['mae'] < 0  # Improvement (lower MAE)
    
    def test_rollback_capabilities(self):
        """Test model rollback capabilities."""
        # Save multiple versions
        v1_id = self.production_utils.save_model_version(
            self.mock_model, self.model_metadata, version='1.0.0'
        )
        
        v2_metadata = self.model_metadata.copy()
        v2_metadata['version'] = '2.0.0'
        v2_id = self.production_utils.save_model_version(
            self.mock_model, v2_metadata, version='2.0.0'
        )
        
        # Set v2 as active
        self.production_utils.set_active_model(v2_id)
        assert self.production_utils.get_active_model_id() == v2_id
        
        # Rollback to v1
        self.production_utils.rollback_to_version(v1_id)
        assert self.production_utils.get_active_model_id() == v1_id
    
    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        # Save some models and configuration
        version_id = self.production_utils.save_model_version(
            self.mock_model, self.model_metadata
        )
        
        config = {'test_setting': 'test_value'}
        self.production_utils.save_configuration(config)
        
        # Create backup
        backup_path = os.path.join(self.temp_dir, 'backup.zip')
        self.production_utils.create_backup(backup_path)
        
        assert os.path.exists(backup_path)
        assert os.path.getsize(backup_path) > 0
        
        # Test restore (would need separate temp directory in real implementation)
        # For test, just verify backup file structure
        import zipfile
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            files_in_backup = backup_zip.namelist()
            assert any('models' in f for f in files_in_backup)
            assert any('config' in f for f in files_in_backup)
    
    def test_monitoring_alerts_generation(self):
        """Test monitoring alerts generation."""
        # Configure alert thresholds
        config = {
            'performance_threshold': 5.0,
            'drift_threshold': 0.5,
            'health_check_interval': 300
        }
        self.production_utils.save_configuration(config)
        
        # Simulate poor performance that should trigger alert
        version_id = self.production_utils.save_model_version(
            self.mock_model, self.model_metadata
        )
        
        poor_predictions = [50.0, 45.0, 40.0]  # Very different from expected
        actuals = [85.0, 80.0, 90.0]
        
        alerts = self.production_utils.check_for_alerts(
            version_id, poor_predictions, actuals
        )
        
        assert len(alerts) > 0
        assert any(alert['type'] == 'PERFORMANCE_DEGRADATION' for alert in alerts)
    
    def test_deployment_readiness_check(self):
        """Test deployment readiness check."""
        # Save a model with good metrics
        good_metadata = self.model_metadata.copy()
        good_metadata['performance_metrics'] = {'mae': 2.5, 'rmse': 3.0, 'r2': 0.92}
        
        version_id = self.production_utils.save_model_version(
            self.mock_model, good_metadata
        )
        
        readiness_report = self.production_utils.check_deployment_readiness(version_id)
        
        assert 'ready_for_deployment' in readiness_report
        assert 'checks_passed' in readiness_report
        assert 'recommendations' in readiness_report
        
        # Should pass performance checks
        assert readiness_report['ready_for_deployment'] is True
        assert 'performance' in readiness_report['checks_passed']
"""
Production Utilities for ML pipeline.

This module provides model versioning and management, health checks and monitoring,
and configuration management capabilities.
"""

import json
import os
import pickle
import shutil
import zipfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import uuid
import logging
import psutil

logger = logging.getLogger(__name__)


class ProductionUtils:
    """
    Production utilities for model management and monitoring.
    
    This class provides comprehensive production-ready utilities including
    model versioning, health checks, configuration management, and monitoring.
    """
    
    def __init__(self, base_path: str = './ml_production'):
        """
        Initialize production utilities.
        
        Args:
            base_path: Base directory for production files
        """
        self.base_path = base_path
        self.models_path = os.path.join(base_path, 'models')
        self.config_path = os.path.join(base_path, 'config')
        self.monitoring_path = os.path.join(base_path, 'monitoring')
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.config_path, exist_ok=True)
        os.makedirs(self.monitoring_path, exist_ok=True)
        
        self.model_registry = self._load_model_registry()
        self.performance_history = {}
        
        logger.info(f"ProductionUtils initialized with base path: {base_path}")
    
    def save_model_version(self, model, metadata: Dict[str, Any], 
                          version: Optional[str] = None) -> str:
        """
        Save a model version with metadata.
        
        Args:
            model: Trained model to save
            metadata: Model metadata
            version: Optional version string
            
        Returns:
            Unique version ID
        """
        version_id = str(uuid.uuid4())
        
        # Create a copy of metadata to avoid modifying the original
        metadata_copy = metadata.copy()
        
        if version:
            metadata_copy['version'] = version
        
        metadata_copy['version_id'] = version_id
        metadata_copy['saved_at'] = datetime.now().isoformat()
        
        # Save model
        model_file = os.path.join(self.models_path, f'{version_id}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_file = os.path.join(self.models_path, f'{version_id}_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata_copy, f, indent=2)
        
        # Update registry
        self.model_registry[version_id] = metadata_copy
        self._save_model_registry()
        
        logger.info(f"Model version saved: {version_id} (version: {version})")
        
        return version_id
    
    def load_model_version(self, version_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a specific model version.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            Tuple of (model, metadata)
        """
        model_file = os.path.join(self.models_path, f'{version_id}.pkl')
        metadata_file = os.path.join(self.models_path, f'{version_id}_metadata.json')
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model version {version_id} not found")
        
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model version loaded: {version_id}")
        
        return model, metadata
    
    def get_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the complete model registry.
        
        Returns:
            Dictionary of all model versions and metadata
        """
        return self.model_registry.copy()
    
    def get_latest_model_version(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest model version.
        
        Returns:
            Dictionary with latest model info or None if no models
        """
        if not self.model_registry:
            return None
        
        # Sort by saved_at timestamp
        latest = max(
            self.model_registry.items(),
            key=lambda x: x[1].get('saved_at', '0')
        )
        
        return {
            'version_id': latest[0],
            'metadata': latest[1]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dictionary containing system health status
        """
        logger.info("Performing health check")
        
        # Check system metrics
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage(self.base_path).percent
        
        # Check model availability
        models_available = len(self.model_registry)
        
        # Determine overall status
        if models_available == 0:
            status = 'CRITICAL'
        elif memory_usage > 90 or disk_usage > 90:
            status = 'WARNING'
        else:
            status = 'HEALTHY'
        
        health_status = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'model_status': {
                'models_available': models_available,
                'latest_model': self.get_latest_model_version()
            },
            'system_metrics': {
                'memory_usage_percent': memory_usage,
                'disk_usage_percent': disk_usage
            },
            'last_prediction_time': self._get_last_prediction_time()
        }
        
        logger.info(f"Health check completed: {status}")
        
        return health_status
    
    def save_configuration(self, config: Dict[str, Any]) -> None:
        """
        Save production configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.config_path, 'production_config.json')
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Production configuration saved")
    
    def load_configuration(self) -> Dict[str, Any]:
        """
        Load production configuration.
        
        Returns:
            Configuration dictionary
        """
        config_file = os.path.join(self.config_path, 'production_config.json')
        
        if not os.path.exists(config_file):
            return {}
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info("Production configuration loaded")
        
        return config
    
    def cleanup_old_models(self) -> None:
        """Clean up old model versions based on configuration."""
        config = self.load_configuration()
        max_versions = config.get('max_model_versions', 10)
        
        if len(self.model_registry) <= max_versions:
            return
        
        # Sort by saved_at timestamp
        sorted_models = sorted(
            self.model_registry.items(),
            key=lambda x: x[1].get('saved_at', '0')
        )
        
        # Keep only the latest versions
        models_to_remove = sorted_models[:-max_versions]
        
        for version_id, _ in models_to_remove:
            self._remove_model_version(version_id)
        
        logger.info(f"Cleaned up {len(models_to_remove)} old model versions")
    
    def record_prediction_performance(self, version_id: str, predictions: List[float],
                                    actuals: List[float], timestamp: Optional[str] = None) -> None:
        """
        Record prediction performance for a model version.
        
        Args:
            version_id: Model version ID
            predictions: Prediction values
            actuals: Actual values
            timestamp: Timestamp for the record
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(actuals, predictions)
        
        if version_id not in self.performance_history:
            self.performance_history[version_id] = {}
        
        self.performance_history[version_id][timestamp] = {
            'mae': mae,
            'prediction_count': len(predictions),
            'timestamp': timestamp
        }
        
        logger.info(f"Performance recorded for {version_id}: MAE={mae:.3f}")
    
    def get_performance_history(self, version_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get performance history for a model version.
        
        Args:
            version_id: Model version ID
            
        Returns:
            Performance history dictionary
        """
        return self.performance_history.get(version_id, {})
    
    def compare_model_versions(self, version_1_id: str, version_2_id: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_1_id: First model version ID
            version_2_id: Second model version ID
            
        Returns:
            Comparison results
        """
        v1_metadata = self.model_registry.get(version_1_id, {})
        v2_metadata = self.model_registry.get(version_2_id, {})
        
        v1_metrics = v1_metadata.get('performance_metrics', {})
        v2_metrics = v2_metadata.get('performance_metrics', {})
        
        comparison = {
            'version_1': {
                'version_id': version_1_id,
                'metrics': v1_metrics
            },
            'version_2': {
                'version_id': version_2_id,
                'metrics': v2_metrics
            },
            'performance_improvement': {}
        }
        
        # Calculate improvements (negative = improvement for MAE)
        if 'mae' in v1_metrics and 'mae' in v2_metrics:
            # Lower is better for MAE, so v2 - v1 gives negative when v2 is better
            comparison['performance_improvement']['mae'] = v2_metrics['mae'] - v1_metrics['mae']
        
        logger.info(f"Model comparison completed: {version_1_id} vs {version_2_id}")
        
        return comparison
    
    def set_active_model(self, version_id: str) -> None:
        """Set a model version as active."""
        active_config = {'active_model_id': version_id, 'updated_at': datetime.now().isoformat()}
        
        with open(os.path.join(self.config_path, 'active_model.json'), 'w') as f:
            json.dump(active_config, f, indent=2)
        
        logger.info(f"Active model set to: {version_id}")
    
    def get_active_model_id(self) -> Optional[str]:
        """Get the active model version ID."""
        active_file = os.path.join(self.config_path, 'active_model.json')
        
        if not os.path.exists(active_file):
            return None
        
        with open(active_file, 'r') as f:
            active_config = json.load(f)
        
        return active_config.get('active_model_id')
    
    def rollback_to_version(self, version_id: str) -> None:
        """Rollback to a specific model version."""
        if version_id not in self.model_registry:
            raise ValueError(f"Model version {version_id} not found")
        
        self.set_active_model(version_id)
        logger.info(f"Rolled back to model version: {version_id}")
    
    def create_backup(self, backup_path: str) -> None:
        """Create a backup of all production data."""
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Backup models
            for root, dirs, files in os.walk(self.models_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, self.base_path)
                    backup_zip.write(file_path, archive_path)
            
            # Backup config
            for root, dirs, files in os.walk(self.config_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, self.base_path)
                    backup_zip.write(file_path, archive_path)
        
        logger.info(f"Backup created: {backup_path}")
    
    def check_for_alerts(self, version_id: str, predictions: List[float], 
                        actuals: List[float]) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        from sklearn.metrics import mean_absolute_error
        current_mae = mean_absolute_error(actuals, predictions)
        
        config = self.load_configuration()
        performance_threshold = config.get('performance_threshold', 5.0)
        
        alerts = []
        
        if current_mae > performance_threshold:
            alerts.append({
                'type': 'PERFORMANCE_DEGRADATION',
                'message': f'MAE ({current_mae:.3f}) exceeds threshold ({performance_threshold})',
                'severity': 'HIGH',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def check_deployment_readiness(self, version_id: str) -> Dict[str, Any]:
        """Check if a model version is ready for deployment."""
        if version_id not in self.model_registry:
            return {
                'ready_for_deployment': False,
                'checks_passed': [],
                'checks_failed': ['Model version not found'],
                'recommendations': ['Ensure model version exists in registry']
            }
        
        metadata = self.model_registry[version_id]
        metrics = metadata.get('performance_metrics', {})
        
        checks_passed = []
        checks_failed = []
        recommendations = []
        
        # Check performance metrics
        if metrics.get('mae', float('inf')) < 5.0:
            checks_passed.append('performance')
        else:
            checks_failed.append('performance')
            recommendations.append('Improve model performance (MAE > 5.0)')
        
        # Check if model files exist
        model_file = os.path.join(self.models_path, f'{version_id}.pkl')
        if os.path.exists(model_file):
            checks_passed.append('model_files')
        else:
            checks_failed.append('model_files')
            recommendations.append('Ensure model files are properly saved')
        
        ready = len(checks_failed) == 0
        
        return {
            'ready_for_deployment': ready,
            'checks_passed': checks_passed,
            'checks_failed': checks_failed,
            'recommendations': recommendations
        }
    
    def _load_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load model registry from disk."""
        registry_file = os.path.join(self.config_path, 'model_registry.json')
        
        if not os.path.exists(registry_file):
            return {}
        
        with open(registry_file, 'r') as f:
            return json.load(f)
    
    def _save_model_registry(self) -> None:
        """Save model registry to disk."""
        registry_file = os.path.join(self.config_path, 'model_registry.json')
        
        with open(registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def _remove_model_version(self, version_id: str) -> None:
        """Remove a model version from disk and registry."""
        # Remove files
        for ext in ['.pkl', '_metadata.json']:
            file_path = os.path.join(self.models_path, f'{version_id}{ext}')
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove from registry
        if version_id in self.model_registry:
            del self.model_registry[version_id]
        
        self._save_model_registry()
    
    def _get_last_prediction_time(self) -> Optional[str]:
        """Get the timestamp of the last prediction."""
        all_timestamps = []
        
        for version_history in self.performance_history.values():
            all_timestamps.extend(version_history.keys())
        
        if not all_timestamps:
            return None
        
        return max(all_timestamps)
    
    def compare_models(self, version_id_a: str, version_id_b: str) -> Dict[str, Any]:
        """
        Compare performance between two model versions.
        
        Stage 3 enhancement: Advanced model comparison with performance metrics
        and automated recommendation generation.
        
        Args:
            version_id_a: First model version ID
            version_id_b: Second model version ID
            
        Returns:
            Dictionary containing detailed comparison results
        """
        if version_id_a not in self.model_registry:
            raise ValueError(f"Model version {version_id_a} not found in registry")
        if version_id_b not in self.model_registry:
            raise ValueError(f"Model version {version_id_b} not found in registry")
        
        metadata_a = self.model_registry[version_id_a]
        metadata_b = self.model_registry[version_id_b]
        
        # Extract performance metrics
        metrics_a = metadata_a.get('performance_metrics', {})
        metrics_b = metadata_b.get('performance_metrics', {})
        
        # Compare each metric
        performance_comparison = {}
        for metric in ['mae', 'rmse', 'r2']:
            if metric in metrics_a and metric in metrics_b:
                value_a = metrics_a[metric]
                value_b = metrics_b[metric]
                
                # Determine winner (lower is better for mae/rmse, higher for r2)
                if metric in ['mae', 'rmse']:
                    winner = 'model_a' if value_a < value_b else 'model_b'
                else:  # r2
                    winner = 'model_a' if value_a > value_b else 'model_b'
                
                performance_comparison[metric] = {
                    'model_a': value_a,
                    'model_b': value_b,
                    'winner': winner,
                    'difference': abs(value_a - value_b)
                }
        
        # Determine overall recommendation
        wins_a = sum(1 for comp in performance_comparison.values() if comp['winner'] == 'model_a')
        wins_b = sum(1 for comp in performance_comparison.values() if comp['winner'] == 'model_b')
        
        if wins_a > wins_b:
            recommendation = 'model_a'
        elif wins_b > wins_a:
            recommendation = 'model_b'
        else:
            recommendation = 'tie'
        
        logger.info(f"Model comparison completed: {version_id_a} vs {version_id_b}, recommendation: {recommendation}")
        
        return {
            'model_a': {
                'version_id': version_id_a,
                'version': metadata_a.get('version', 'unknown'),
                'metrics': metrics_a
            },
            'model_b': {
                'version_id': version_id_b,  
                'version': metadata_b.get('version', 'unknown'),
                'metrics': metrics_b
            },
            'performance_comparison': performance_comparison,
            'recommendation': recommendation,
            'comparison_timestamp': datetime.now().isoformat()
        }
    
    def promote_model(self, version_id: str, environment: str) -> bool:
        """
        Promote a model version to a specific environment.
        
        Stage 3 enhancement: Model promotion with environment tracking
        and deployment validation.
        
        Args:
            version_id: Model version ID to promote
            environment: Target environment (e.g., 'staging', 'production')
            
        Returns:
            True if promotion successful, False otherwise
        """
        if version_id not in self.model_registry:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        # Update model metadata with promotion info
        self.model_registry[version_id]['environment'] = environment
        self.model_registry[version_id]['promoted_at'] = datetime.now().isoformat()
        
        # Save updated registry
        self._save_model_registry()
        
        logger.info(f"Model {version_id} promoted to {environment}")
        
        return True
    
    def get_models_by_version(self, version_pattern: str) -> List[Dict[str, Any]]:
        """
        Get models matching a semantic version pattern.
        
        Stage 3 enhancement: Semantic version querying with pattern matching.
        
        Args:
            version_pattern: Version pattern (e.g., '1.*', '2.1.*')
            
        Returns:
            List of model dictionaries matching the pattern
        """
        import re
        
        # Convert pattern to regex (simple implementation)
        regex_pattern = version_pattern.replace('.', r'\.').replace('*', r'.*')
        regex_pattern = f'^{regex_pattern}$'
        
        matching_models = []
        
        for version_id, metadata in self.model_registry.items():
            version = metadata.get('version', '')
            if re.match(regex_pattern, version):
                matching_models.append({
                    'version_id': version_id,
                    'version': version,
                    'metadata': metadata
                })
        
        # Sort by version (basic string sort, could be enhanced)
        matching_models.sort(key=lambda x: x['version'])
        
        logger.info(f"Found {len(matching_models)} models matching pattern '{version_pattern}'")
        
        return matching_models
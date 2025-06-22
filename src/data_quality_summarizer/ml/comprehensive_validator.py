"""
Comprehensive validation framework for Stage 3 ML pipeline quality assurance.

This module implements US3.3: Comprehensive validation with MAE < 15% target.
It provides end-to-end validation capabilities for ML model performance,
prediction quality, and feature importance analysis.

Key functionality:
- Prediction variance validation
- Feature importance analysis
- Cross-dataset performance validation
- Comprehensive validation reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import structlog
from dataclasses import dataclass

logger = structlog.get_logger(__name__)


@dataclass
class ValidationReport:
    """
    Data class for comprehensive validation report results.
    """
    prediction_variance: Dict[str, Any]
    feature_importance: Dict[str, Any]
    cross_dataset_performance: Dict[str, Any]
    overall_status: str
    recommendations: List[str]
    summary_metrics: Dict[str, float]


class ComprehensiveValidator:
    """
    Comprehensive validation framework for ML pipeline quality assurance.
    
    This class implements the complete Stage 3 validation requirements,
    ensuring that the ML pipeline meets quality standards and performance
    targets before deployment.
    """
    
    def __init__(self, mae_target: float = 15.0, variance_threshold: float = 1.0):
        """
        Initialize the comprehensive validator.
        
        Args:
            mae_target: Target Mean Absolute Error threshold (default: 15%)
            variance_threshold: Minimum prediction variance threshold (default: 1%)
        """
        self.mae_target = mae_target
        self.variance_threshold = variance_threshold
        
        logger.info("ComprehensiveValidator initialized",
                   mae_target=mae_target,
                   variance_threshold=variance_threshold)
    
    def validate_prediction_variance(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Validate that predictions show sufficient variance.
        
        This addresses the critical issue where models predict constant values
        (e.g., all 0.0%), indicating failure to learn meaningful patterns.
        
        Args:
            predictions: Array of model predictions
            
        Returns:
            Dictionary with variance validation results
        """
        if len(predictions) == 0:
            return {
                'variance_sufficient': False,
                'error': 'Empty predictions array',
                'std_deviation': 0.0,
                'unique_predictions': 0,
                'prediction_range': 0.0
            }
        
        # Calculate variance metrics
        std_deviation = np.std(predictions)
        unique_predictions = len(np.unique(predictions))
        prediction_range = np.max(predictions) - np.min(predictions)
        
        # Check variance sufficiency
        variance_sufficient = std_deviation > self.variance_threshold
        
        # Additional quality checks
        min_unique_threshold = max(3, len(predictions) // 100)  # At least 3 or 1% unique
        uniqueness_sufficient = unique_predictions >= min_unique_threshold
        
        range_sufficient = prediction_range > (2 * self.variance_threshold)
        
        overall_sufficient = variance_sufficient and uniqueness_sufficient and range_sufficient
        
        result = {
            'variance_sufficient': overall_sufficient,
            'std_deviation': float(std_deviation),
            'unique_predictions': int(unique_predictions),
            'prediction_range': float(prediction_range),
            'variance_threshold': self.variance_threshold,
            'meets_std_threshold': variance_sufficient,
            'meets_uniqueness_threshold': uniqueness_sufficient,
            'meets_range_threshold': range_sufficient,
            'prediction_count': len(predictions),
            'prediction_summary': {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'mean': float(np.mean(predictions)),
                'median': float(np.median(predictions))
            }
        }
        
        logger.info("Prediction variance validation completed",
                   sufficient=overall_sufficient,
                   std=std_deviation,
                   unique_count=unique_predictions,
                   range=prediction_range)
        
        return result
    
    def validate_feature_importance(self, model) -> Dict[str, Any]:
        """
        Validate feature importance to ensure model is learning from features.
        
        Args:
            model: Trained LightGBM model with feature_importance method
            
        Returns:
            Dictionary with feature importance validation results
        """
        try:
            # Get feature importance from model
            feature_importance = model.feature_importance()
            
            if len(feature_importance) == 0:
                return {
                    'error': 'No feature importance available',
                    'feature_utilization_rate': 0.0
                }
            
            # Calculate importance metrics
            total_features = len(feature_importance)
            non_zero_features = np.sum(feature_importance > 0)
            zero_importance_features = np.sum(feature_importance == 0)
            
            feature_utilization_rate = non_zero_features / total_features if total_features > 0 else 0.0
            
            # Analyze importance distribution
            importance_stats = {
                'mean': float(np.mean(feature_importance)),
                'std': float(np.std(feature_importance)),
                'min': float(np.min(feature_importance)),
                'max': float(np.max(feature_importance)),
                'sum': float(np.sum(feature_importance))
            }
            
            # Check if feature utilization meets target (>70%)
            utilization_target = 0.7
            utilization_sufficient = feature_utilization_rate >= utilization_target
            
            result = {
                'total_features': int(total_features),
                'non_zero_features': int(non_zero_features),
                'zero_importance_features': int(zero_importance_features),
                'feature_utilization_rate': float(feature_utilization_rate),
                'utilization_target': utilization_target,
                'utilization_sufficient': utilization_sufficient,
                'importance_statistics': importance_stats,
                'top_features': self._get_top_features(feature_importance),
                'unused_features': self._get_unused_features(feature_importance)
            }
            
            logger.info("Feature importance validation completed",
                       utilization_rate=feature_utilization_rate,
                       sufficient=utilization_sufficient,
                       non_zero_count=non_zero_features)
            
            return result
            
        except Exception as e:
            logger.error("Feature importance validation failed", error=str(e))
            return {
                'error': f'Feature importance validation failed: {str(e)}',
                'feature_utilization_rate': 0.0
            }
    
    def _get_top_features(self, feature_importance: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top N features by importance."""
        if len(feature_importance) == 0:
            return []
        
        # Get indices of top features
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        
        top_features = []
        for i, idx in enumerate(top_indices):
            if idx < len(feature_importance):
                top_features.append({
                    'rank': i + 1,
                    'feature_index': int(idx),
                    'importance': float(feature_importance[idx])
                })
        
        return top_features
    
    def _get_unused_features(self, feature_importance: np.ndarray) -> List[int]:
        """Get indices of features with zero importance."""
        return [int(i) for i, importance in enumerate(feature_importance) if importance == 0]
    
    def validate_cross_dataset_performance(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate model performance across different datasets.
        
        Args:
            results: DataFrame with columns ['dataset_uuid', 'actual', 'predicted', 'rule_code']
            
        Returns:
            Dictionary with cross-dataset performance validation results
        """
        required_columns = ['dataset_uuid', 'actual', 'predicted']
        if not all(col in results.columns for col in required_columns):
            return {
                'error': f'Missing required columns. Need: {required_columns}',
                'overall_mae': float('inf')
            }
        
        if len(results) == 0:
            return {
                'error': 'Empty results DataFrame',
                'overall_mae': float('inf')
            }
        
        try:
            # Calculate overall metrics
            overall_mae = np.mean(np.abs(results['actual'] - results['predicted']))
            overall_rmse = np.sqrt(np.mean((results['actual'] - results['predicted']) ** 2))
            
            # Calculate per-dataset performance
            dataset_performance = {}
            for dataset_uuid in results['dataset_uuid'].unique():
                dataset_data = results[results['dataset_uuid'] == dataset_uuid]
                
                if len(dataset_data) > 0:
                    dataset_mae = np.mean(np.abs(dataset_data['actual'] - dataset_data['predicted']))
                    dataset_rmse = np.sqrt(np.mean((dataset_data['actual'] - dataset_data['predicted']) ** 2))
                    
                    dataset_performance[dataset_uuid] = {
                        'mae': float(dataset_mae),
                        'rmse': float(dataset_rmse),
                        'sample_count': len(dataset_data),
                        'mae_meets_target': dataset_mae < self.mae_target
                    }
            
            # Calculate performance consistency
            dataset_maes = [perf['mae'] for perf in dataset_performance.values()]
            mae_std = np.std(dataset_maes) if len(dataset_maes) > 1 else 0.0
            
            # Check if overall performance meets target
            mae_meets_target = overall_mae < self.mae_target
            
            # Check consistency (datasets shouldn't vary too much)
            consistency_threshold = 5.0  # MAE standard deviation should be < 5%
            performance_consistent = mae_std < consistency_threshold
            
            result = {
                'overall_mae': float(overall_mae),
                'overall_rmse': float(overall_rmse),
                'mae_target': self.mae_target,
                'mae_meets_target': mae_meets_target,
                'dataset_performance': dataset_performance,
                'performance_consistency': {
                    'mae_std_across_datasets': float(mae_std),
                    'consistency_threshold': consistency_threshold,
                    'is_consistent': performance_consistent
                },
                'dataset_count': len(dataset_performance),
                'total_predictions': len(results)
            }
            
            logger.info("Cross-dataset performance validation completed",
                       overall_mae=overall_mae,
                       meets_target=mae_meets_target,
                       dataset_count=len(dataset_performance),
                       consistent=performance_consistent)
            
            return result
            
        except Exception as e:
            logger.error("Cross-dataset performance validation failed", error=str(e))
            return {
                'error': f'Performance validation failed: {str(e)}',
                'overall_mae': float('inf')
            }
    
    def generate_validation_report(self, predictions: np.ndarray, model, results: pd.DataFrame) -> ValidationReport:
        """
        Generate comprehensive validation report combining all validation aspects.
        
        Args:
            predictions: Array of model predictions
            model: Trained LightGBM model
            results: DataFrame with actual vs predicted results
            
        Returns:
            ValidationReport with comprehensive validation results
        """
        logger.info("Generating comprehensive validation report")
        
        # Run all validation components
        variance_validation = self.validate_prediction_variance(predictions)
        importance_validation = self.validate_feature_importance(model)
        performance_validation = self.validate_cross_dataset_performance(results)
        
        # Determine overall status
        variance_pass = variance_validation.get('variance_sufficient', False)
        importance_pass = importance_validation.get('utilization_sufficient', False)
        performance_pass = performance_validation.get('mae_meets_target', False)
        
        all_pass = variance_pass and importance_pass and performance_pass
        overall_status = 'PASS' if all_pass else 'FAIL'
        
        # Generate recommendations
        recommendations = []
        
        if not variance_pass:
            recommendations.append("Improve model to generate varied predictions (current predictions too constant)")
        
        if not importance_pass:
            recommendations.append("Enhance feature engineering to improve feature utilization (target: >70%)")
        
        if not performance_pass:
            mae = performance_validation.get('overall_mae', float('inf'))
            recommendations.append(f"Reduce prediction error (current MAE: {mae:.2f}%, target: <{self.mae_target}%)")
        
        if all_pass:
            recommendations.append("All validation criteria met - model ready for production")
        
        # Summary metrics
        summary_metrics = {
            'overall_mae': performance_validation.get('overall_mae', float('inf')),
            'prediction_std': variance_validation.get('std_deviation', 0.0),
            'feature_utilization_rate': importance_validation.get('feature_utilization_rate', 0.0),
            'validation_score': sum([variance_pass, importance_pass, performance_pass]) / 3.0
        }
        
        report = ValidationReport(
            prediction_variance=variance_validation,
            feature_importance=importance_validation,
            cross_dataset_performance=performance_validation,
            overall_status=overall_status,
            recommendations=recommendations,
            summary_metrics=summary_metrics
        )
        
        logger.info("Comprehensive validation report generated",
                   overall_status=overall_status,
                   variance_pass=variance_pass,
                   importance_pass=importance_pass,
                   performance_pass=performance_pass)
        
        return report
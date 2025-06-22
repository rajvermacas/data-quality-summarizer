"""Data validation module for ML pipeline quality assurance."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DataQualityException(Exception):
    """Exception raised when data quality checks fail."""
    pass


@dataclass
class ValidationReport:
    """Data validation report structure."""
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Data quality validator for ML training pipeline."""
    
    def __init__(self, min_variance: float = 0.1, min_samples_per_group: int = 50):
        """
        Initialize data validator.
        
        Args:
            min_variance: Minimum required standard deviation for target variable
            min_samples_per_group: Minimum samples required per dataset-rule combination
        """
        self.min_variance = min_variance
        self.min_samples_per_group = min_samples_per_group
    
    def validate_target_distribution(self, data: pd.DataFrame, target_col: str) -> ValidationReport:
        """
        Validate target variable distribution for training suitability.
        
        Args:
            data: Training data DataFrame
            target_col: Name of target variable column
            
        Returns:
            ValidationReport with distribution analysis
            
        Raises:
            DataQualityException: If target variance is insufficient for training
        """
        if target_col not in data.columns:
            raise KeyError(f"Target column '{target_col}' not found in data")
        
        target_values = data[target_col].dropna()
        
        if len(target_values) == 0:
            raise DataQualityException("No valid target values found")
        
        # Calculate target statistics
        target_stats = {
            'count': len(target_values),
            'null_count': data[target_col].isnull().sum(),
            'zero_count': (target_values == 0).sum(),
            'mean': float(target_values.mean()),
            'std': float(target_values.std()),
            'min': float(target_values.min()),
            'max': float(target_values.max())
        }
        
        # Check for insufficient variance
        if target_stats['std'] < self.min_variance:
            raise DataQualityException(
                f"Insufficient target variable variance: {target_stats['std']:.4f} < {self.min_variance}"
            )
        
        # Check for high zero percentage (warning, not failure)
        zero_percentage = (target_stats['zero_count'] / target_stats['count']) * 100
        if zero_percentage > 90:
            logger.warning(
                "High zero percentage in target",
                percentage=zero_percentage,
                zero_count=target_stats['zero_count'],
                total_count=target_stats['count']
            )
        
        return ValidationReport(
            passed=True,
            message="Target distribution validation passed",
            details={'target_stats': target_stats}
        )
    
    def check_sample_sizes(self, data: pd.DataFrame, group_cols: List[str]) -> Dict[str, int]:
        """
        Check sample sizes per group combination.
        
        Args:
            data: Training data DataFrame
            group_cols: List of columns to group by
            
        Returns:
            Dictionary mapping group keys to sample counts
            
        Raises:
            DataQualityException: If any group has insufficient samples
        """
        if not group_cols:
            raise ValueError("Group columns list cannot be empty")
        
        # Check if all group columns exist
        missing_cols = [col for col in group_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Group columns not found: {missing_cols}")
        
        # Calculate group sizes
        group_sizes = data.groupby(group_cols).size()
        sample_counts = {str(key): count for key, count in group_sizes.items()}
        
        # Check for insufficient samples
        insufficient_groups = {
            key: count for key, count in sample_counts.items() 
            if count < self.min_samples_per_group
        }
        
        if insufficient_groups:
            raise DataQualityException(
                f"Insufficient sample size for groups (min {self.min_samples_per_group}): "
                f"{insufficient_groups}"
            )
        
        return sample_counts
    
    def validate_feature_matrix_rank(self, X: pd.DataFrame) -> float:
        """
        Validate feature matrix rank to detect multicollinearity.
        
        Args:
            X: Feature matrix DataFrame
            
        Returns:
            Rank ratio (rank / num_features)
            
        Raises:
            DataQualityException: If feature matrix has poor rank
        """
        if X.empty:
            raise ValueError("Feature matrix is empty")
        
        # Convert to numeric and handle any remaining non-numeric data
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            raise ValueError("No numeric features found in feature matrix")
        
        # Calculate matrix rank
        matrix_rank = np.linalg.matrix_rank(X_numeric.values)
        num_features = X_numeric.shape[1]
        rank_ratio = matrix_rank / num_features
        
        # Check for poor rank (indicates multicollinearity)
        if rank_ratio < 0.8:
            raise DataQualityException(
                f"Poor feature matrix rank: {rank_ratio:.3f} "
                f"(rank {matrix_rank} / {num_features} features). "
                "This indicates high multicollinearity."
            )
        
        logger.info(f"Feature matrix rank validation passed: {rank_ratio:.3f}")
        return rank_ratio
    
    def generate_quality_report(self, data: pd.DataFrame, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: Training data DataFrame
            output_path: Optional path to save report JSON
            
        Returns:
            Dictionary containing complete quality analysis
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        try:
            # Target distribution analysis (assuming 'pass_percentage' is target)
            if 'pass_percentage' in data.columns:
                target_report = self.validate_target_distribution(data, 'pass_percentage')
                report['target_distribution'] = target_report.details
            
            # Sample size analysis
            if all(col in data.columns for col in ['dataset_uuid', 'rule_code']):
                sample_sizes = self.check_sample_sizes(data, ['dataset_uuid', 'rule_code'])
                report['sample_sizes'] = {
                    'group_counts': sample_sizes,
                    'total_groups': len(sample_sizes),
                    'min_group_size': min(sample_sizes.values()) if sample_sizes else 0,
                    'max_group_size': max(sample_sizes.values()) if sample_sizes else 0
                }
            
            # Feature matrix rank analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                rank_ratio = self.validate_feature_matrix_rank(data[numeric_cols])
                report['feature_matrix_rank'] = {
                    'rank_ratio': rank_ratio,
                    'num_features': len(numeric_cols)
                }
            
            report['validation_status'] = 'passed'
            
        except Exception as e:
            report['validation_status'] = 'failed'
            report['error'] = str(e)
            logger.error(f"Quality report generation failed: {e}")
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_report = self._convert_numpy_types(report)
                json.dump(json_report, f, indent=2)
            logger.info(f"Quality report saved to {output_path}")
        
        return report
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
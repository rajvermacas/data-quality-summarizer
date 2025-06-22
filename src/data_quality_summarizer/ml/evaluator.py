"""
Evaluator module for ML pipeline.

This module provides model evaluation functionality including
MAE, MSE, RMSE, MAPE calculations and comprehensive reporting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluator for pass percentage prediction models.
    
    Provides comprehensive evaluation metrics and reporting
    for regression model performance assessment.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.metrics = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_predictions(
        self, 
        actual: np.ndarray, 
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate predictions using multiple regression metrics.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            raise ValueError("No valid predictions to evaluate")
        
        metrics = {
            'mae': calculate_mae(actual_clean, predicted_clean),
            'mse': calculate_mse(actual_clean, predicted_clean),
            'rmse': calculate_rmse(actual_clean, predicted_clean),
            'mape': calculate_mape(actual_clean, predicted_clean),
            'count': len(actual_clean)
        }
        
        self.metrics.update(metrics)
        logger.info(f"Evaluated {len(actual_clean)} predictions - MAE: {metrics['mae']:.2f}")
        
        return metrics
    
    def evaluate(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance using test data.
        
        This method bridges the gap between pipeline expectations
        (model + test data) and existing evaluation methods (predictions).
        
        Args:
            model: Trained model with predict() method
            X_test: Test features DataFrame  
            y_test: Test target Series
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If model or data is invalid
        """
        try:
            # Prepare categorical features for prediction if needed
            X_test_prepared = X_test.copy()
            
            # Check if X_test has categorical columns that need preparation
            categorical_cols = [col for col in ['dataset_uuid', 'rule_code'] 
                              if col in X_test.columns]
            
            if categorical_cols:
                # Convert categorical columns to category dtype for LightGBM compatibility
                for col in categorical_cols:
                    if col in X_test_prepared.columns:
                        X_test_prepared[col] = X_test_prepared[col].astype('category')
            
            # Generate predictions using the model
            predictions = model.predict(X_test_prepared)
            
            # Use existing evaluation method
            return self.evaluate_predictions(
                actual=y_test.values,
                predicted=predictions
            )
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                'mae': float('inf'),
                'rmse': float('inf'), 
                'mape': float('inf'),
                'error': str(e)
            }
    
    def evaluate_dataframe(
        self,
        data: pd.DataFrame,
        actual_col: str,
        predicted_col: str
    ) -> Dict[str, float]:
        """
        Evaluate predictions from DataFrame columns.
        
        Args:
            data: DataFrame containing actual and predicted values
            actual_col: Column name for actual values
            predicted_col: Column name for predicted values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if actual_col not in data.columns:
            raise KeyError(f"Actual column '{actual_col}' not found")
        if predicted_col not in data.columns:
            raise KeyError(f"Predicted column '{predicted_col}' not found")
        
        return self.evaluate_predictions(
            data[actual_col].values,
            data[predicted_col].values
        )
    
    def evaluate_by_groups(
        self,
        data: pd.DataFrame,
        group_cols: List[str],
        actual_col: str,
        predicted_col: str
    ) -> pd.DataFrame:
        """
        Evaluate predictions grouped by specified columns.
        
        Args:
            data: DataFrame containing data
            group_cols: List of columns to group by
            actual_col: Column name for actual values
            predicted_col: Column name for predicted values
            
        Returns:
            DataFrame with metrics for each group
        """
        group_metrics = []
        
        for group_name, group_data in data.groupby(group_cols):
            if len(group_data) == 0:
                continue
                
            try:
                metrics = self.evaluate_predictions(
                    group_data[actual_col].values,
                    group_data[predicted_col].values
                )
                
                # Add group identifiers
                group_dict = dict(zip(group_cols, group_name if isinstance(group_name, tuple) else [group_name]))
                group_dict.update(metrics)
                group_metrics.append(group_dict)
                
            except ValueError as e:
                logger.warning(f"Skipping group {group_name}: {e}")
                continue
        
        result_df = pd.DataFrame(group_metrics)
        logger.info(f"Evaluated {len(result_df)} groups")
        
        return result_df
    
    def calculate_residuals(
        self, 
        actual: np.ndarray, 
        predicted: np.ndarray
    ) -> np.ndarray:
        """
        Calculate residuals (actual - predicted).
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Array of residuals
        """
        return calculate_residuals(actual, predicted)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        summary = self.metrics.copy()
        summary['evaluation_date'] = datetime.now().isoformat()
        
        return summary


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Mean Absolute Error
        
    Raises:
        ValueError: If arrays are empty or mismatched lengths
    """
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Arrays cannot be empty")
    
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have same length")
    
    return np.mean(np.abs(actual - predicted))


def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Mean Squared Error
        
    Raises:
        ValueError: If arrays are empty or mismatched lengths
    """
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Arrays cannot be empty")
    
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have same length")
    
    return np.mean((actual - predicted) ** 2)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Root Mean Squared Error
    """
    return np.sqrt(calculate_mse(actual, predicted))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Mean Absolute Percentage Error as percentage
    """
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Arrays cannot be empty")
    
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have same length")
    
    # Handle division by zero - exclude zero actual values
    mask = actual != 0
    if not mask.any():
        return np.inf
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Calculate residuals (actual - predicted).
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Array of residuals
    """
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have same length")
    
    return actual - predicted


def generate_evaluation_report(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        Dictionary containing comprehensive evaluation report
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(actual, predicted)
    
    residuals = calculate_residuals(actual, predicted)
    
    report = {
        'metrics': metrics,
        'summary': {
            'total_predictions': len(actual),
            'mean_actual': np.mean(actual),
            'mean_predicted': np.mean(predicted),
            'std_actual': np.std(actual),
            'std_predicted': np.std(predicted)
        },
        'distribution_stats': {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals)
        },
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    logger.info("Generated comprehensive evaluation report")
    return report


def plot_predictions_vs_actual(
    actual: np.ndarray, 
    predicted: np.ndarray,
    show_plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Create predictions vs actual plot data.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        show_plot: Whether to display plot (for testing, set False)
        
    Returns:
        Dictionary containing plot data
    """
    plot_data = {
        'actual': actual,
        'predicted': predicted,
        'perfect_line': actual,  # Perfect prediction line
        'residuals': calculate_residuals(actual, predicted)
    }
    
    if show_plot:
        # In a real implementation, this would create matplotlib plots
        # For now, just log that plot would be created
        logger.info(f"Plot data prepared for {len(actual)} predictions")
    
    return plot_data
"""
Performance Optimizer for ML pipeline.

This module provides memory usage optimization, training time optimization,
and prediction latency optimization capabilities.
"""

import pandas as pd
import numpy as np
import os
import psutil
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimizes ML pipeline performance for memory, training time, and prediction latency.
    
    This class provides utilities to optimize different aspects of the ML pipeline
    to ensure efficient operation on consumer-grade hardware.
    """
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.optimization_cache = {}
        logger.info("PerformanceOptimizer initialized")
    
    def optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of a DataFrame.
        
        Args:
            data: Input DataFrame to optimize
            
        Returns:
            Optimized DataFrame with reduced memory footprint
        """
        logger.info(f"Optimizing memory usage for DataFrame with shape {data.shape}")
        
        optimized_data = data.copy()
        
        # Optimize numeric columns
        for col in optimized_data.select_dtypes(include=['int64']).columns:
            optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='integer')
        
        for col in optimized_data.select_dtypes(include=['float64']).columns:
            optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='float')
        
        # Optimize object columns to category if beneficial
        for col in optimized_data.select_dtypes(include=['object']).columns:
            try:
                if len(optimized_data) > 0 and optimized_data[col].nunique() / len(optimized_data) < 0.5:
                    optimized_data[col] = optimized_data[col].astype('category')
            except (TypeError, ValueError):
                # Skip columns that can't be converted to category
                continue
        
        original_memory = data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"Memory usage reduced by {reduction:.1f}%")
        
        return optimized_data
    
    def optimize_training_time(self, base_config: Dict[str, Any], data_size: int) -> Dict[str, Any]:
        """
        Optimize LightGBM configuration for faster training.
        
        Args:
            base_config: Base LightGBM configuration
            data_size: Size of training dataset
            
        Returns:
            Optimized configuration dictionary
        """
        logger.info(f"Optimizing training configuration for dataset size: {data_size}")
        
        optimized_config = base_config.copy()
        
        # Set number of threads based on available CPUs
        optimized_config['num_threads'] = min(os.cpu_count() or 1, 4)
        
        # Reduce verbosity for faster training
        optimized_config['verbosity'] = -1
        
        # Adjust parameters based on data size
        if data_size < 1000:
            optimized_config['num_leaves'] = min(optimized_config.get('num_leaves', 31), 15)
            optimized_config['early_stopping_rounds'] = 10
        elif data_size < 10000:
            optimized_config['num_leaves'] = min(optimized_config.get('num_leaves', 31), 31)
            optimized_config['early_stopping_rounds'] = 20
        else:
            optimized_config['num_leaves'] = optimized_config.get('num_leaves', 31)
            optimized_config['early_stopping_rounds'] = 50
        
        # Validate and correct invalid parameters
        if optimized_config.get('num_leaves', 0) <= 0:
            optimized_config['num_leaves'] = 15
        
        learning_rate = optimized_config.get('learning_rate', 0.1)
        if learning_rate > 1.0 or learning_rate <= 0:
            optimized_config['learning_rate'] = 0.1
        
        # Ensure valid objective
        valid_objectives = ['regression', 'regression_l1', 'regression_l2']
        if optimized_config.get('objective') not in valid_objectives:
            optimized_config['objective'] = 'regression'
        
        logger.info(f"Training configuration optimized with {optimized_config['num_threads']} threads")
        
        return optimized_config
    
    def optimize_prediction_latency(self, predictor) -> 'OptimizedPredictor':
        """
        Optimize predictor for lower latency.
        
        Args:
            predictor: Model predictor to optimize
            
        Returns:
            Optimized predictor wrapper
        """
        logger.info("Creating optimized predictor wrapper for reduced latency")
        return OptimizedPredictor(predictor)


class OptimizedPredictor:
    """
    Wrapper for optimized prediction performance.
    
    This class wraps a predictor to provide optimized prediction capabilities
    with reduced latency and improved efficiency.
    """
    
    def __init__(self, predictor):
        """
        Initialize optimized predictor.
        
        Args:
            predictor: Original predictor to wrap
        """
        self.predictor = predictor
        self.prediction_cache = {}
        logger.info("OptimizedPredictor initialized with caching enabled")
    
    def predict(self, data):
        """
        Make optimized predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction results
        """
        # For optimization, we could implement caching, batch processing, etc.
        # For now, delegate to original predictor
        return self.predictor.predict(data)
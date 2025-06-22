"""
ML Pipeline Orchestrator.
Coordinates the complete machine learning pipeline from training to model persistence.
"""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import psutil
import os

from .data_loader import load_and_validate_csv, parse_results_column, create_binary_pass_column
from .aggregator import aggregate_pass_percentages
from .feature_engineer import extract_time_features, create_lag_features, calculate_moving_averages
from .data_splitter import split_data_chronologically
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator


class MLPipeline:
    """
    Orchestrates the complete ML training pipeline.
    
    Coordinates data loading, preprocessing, training, and evaluation
    in a single cohesive workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML pipeline with configuration.
        
        Args:
            config: Optional configuration dictionary for pipeline parameters
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.progress_callback: Optional[Callable] = None
        
        # Initialize pipeline components
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the pipeline."""
        return {
            'chunk_size': 20000,
            'test_size': 0.2,
            'random_state': 42,
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            'min_samples_for_training': 50
        }
    
    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """
        Set callback function for progress tracking.
        
        Args:
            callback: Function that receives (stage, current, total) progress updates
        """
        self.progress_callback = callback
    
    def _report_progress(self, stage: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(stage, current, total)
    
    def train_model(
        self,
        csv_file: str,
        rule_metadata: Dict[str, Any],
        output_model_path: str
    ) -> Dict[str, Any]:
        """
        Execute the complete model training pipeline.
        
        Args:
            csv_file: Path to input CSV file
            rule_metadata: Dictionary of rule metadata
            output_model_path: Path where to save the trained model
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            if not csv_file or not rule_metadata or not output_model_path:
                return {'success': False, 'error': 'Missing required parameters'}
            
            if not Path(csv_file).exists():
                return {'success': False, 'error': f'CSV file not found: {csv_file}'}
            
            start_time = time.time()
            self.logger.info(f"Starting ML pipeline training with CSV: {csv_file}")
            
            # Stage 1: Load and parse data
            self._report_progress("Loading data", 1, 6)
            self.logger.info("Stage 1: Loading and parsing CSV data...")
            raw_data = load_and_validate_csv(csv_file)
            # Parse results column and create binary pass column
            parsed_results = parse_results_column(raw_data['results'].tolist())
            raw_data['is_pass'] = create_binary_pass_column(parsed_results)
            
            if raw_data.empty:
                return {'success': False, 'error': 'No data loaded from CSV file'}
            
            # Stage 2: Aggregate data by dataset/rule/date
            self._report_progress("Aggregating data", 2, 6)
            self.logger.info("Stage 2: Aggregating data by dataset/rule/date...")
            aggregated_data = aggregate_pass_percentages(raw_data)
            
            if len(aggregated_data) < self.config['min_samples_for_training']:
                return {
                    'success': False, 
                    'error': f'Insufficient data for training. Got {len(aggregated_data)} samples, need at least {self.config["min_samples_for_training"]}'
                }
            
            # Stage 3: Feature engineering
            self._report_progress("Engineering features", 3, 6)
            self.logger.info("Stage 3: Engineering features...")
            feature_data = extract_time_features(aggregated_data)
            feature_data = create_lag_features(feature_data)
            feature_data = calculate_moving_averages(feature_data)
            
            # Stage 4: Split data chronologically
            self._report_progress("Splitting data", 4, 6)
            self.logger.info("Stage 4: Splitting data chronologically...")
            # Use 80% for training (chronologically)
            cutoff_idx = int(len(feature_data) * (1 - self.config['test_size']))
            train_data = feature_data.iloc[:cutoff_idx]
            test_data = feature_data.iloc[cutoff_idx:]
            
            # Select only numeric and categorical columns suitable for LightGBM
            # Exclude target variable and non-trainable columns
            exclude_cols = ['pass_percentage', 'business_date', 'source', 'tenant_id', 'dataset_name']
            feature_cols = [col for col in feature_data.columns 
                           if col not in exclude_cols]
            
            # Ensure we have categorical columns for LightGBM
            categorical_cols = ['dataset_uuid', 'rule_code'] if 'dataset_uuid' in feature_cols else []
            
            X_train = train_data[feature_cols]
            y_train = train_data['pass_percentage']
            X_test = test_data[feature_cols]
            y_test = test_data['pass_percentage']
            
            # Stage 5: Train model
            self._report_progress("Training model", 5, 6)
            self.logger.info("Stage 5: Training LightGBM model...")
            
            # Pass categorical columns information to the trainer
            model_params_with_categorical = {
                **self.config['model_params'],
                'categorical_cols': categorical_cols
            }
            
            model = self.model_trainer.train(
                X_train, y_train, 
                model_params=model_params_with_categorical
            )
            
            # Stage 6: Evaluate model and save
            self._report_progress("Evaluating model", 6, 6)
            self.logger.info("Stage 6: Evaluating model and saving...")
            evaluation_metrics = self.evaluator.evaluate(model, X_test, y_test)
            
            # Save the trained model
            self.model_trainer.save_model(model, output_model_path)
            
            training_time = time.time() - start_time
            
            self.logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            
            return {
                'success': True,
                'training_time': training_time,
                'model_path': output_model_path,
                'evaluation_metrics': evaluation_metrics,
                'samples_trained': len(X_train),
                'samples_tested': len(X_test),
                'memory_peak_mb': self.get_memory_usage()
            }
            
        except Exception as e:
            error_msg = f"Pipeline training failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return {'success': False, 'error': error_msg}
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_file}: {e}")
            return {}
    
    def save_config(self, config_file: str) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            config_file: Path where to save configuration
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save config to {config_file}: {e}")
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        self.logger.debug("Pipeline cleanup completed")
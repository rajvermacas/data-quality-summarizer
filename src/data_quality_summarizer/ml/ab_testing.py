"""
A/B Testing Framework for ML model comparison.

Stage 3 enhancement: Statistical A/B testing framework for comparing
model performance in production with traffic splitting and significance testing.
"""

import pandas as pd
import numpy as np
import json
import os
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ABTestingService:
    """
    A/B testing service for comparing ML model performance.
    
    Provides traffic splitting, experiment management, and statistical
    significance testing for model comparisons.
    """
    
    def __init__(self, storage_path: str = './ab_experiments'):
        """
        Initialize A/B testing service.
        
        Args:
            storage_path: Directory to store experiment data
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.experiments_file = os.path.join(storage_path, 'experiments.json')
        self.experiments = self._load_experiments()
        
        logger.info(f"ABTestingService initialized with storage: {storage_path}")
    
    def create_experiment(
        self,
        control_model_version: str,
        treatment_model_version: str,
        traffic_split: float,
        experiment_name: Optional[str] = None,
        duration_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new A/B test experiment.
        
        Args:
            control_model_version: Control (baseline) model version
            treatment_model_version: Treatment (new) model version
            traffic_split: Fraction of traffic to send to treatment (0.0-1.0)
            experiment_name: Optional human-readable experiment name
            duration_days: Optional experiment duration in days
            metadata: Optional experiment metadata
            
        Returns:
            Unique experiment ID
        """
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError(f"traffic_split must be between 0.0 and 1.0, got {traffic_split}")
        
        experiment_id = str(uuid.uuid4())
        
        experiment = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name or f"Experiment {experiment_id[:8]}",
            'control_model': control_model_version,
            'treatment_model': treatment_model_version,
            'traffic_split': traffic_split,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'duration_days': duration_days,
            'metadata': metadata or {},
            'results': {
                'control': [],
                'treatment': []
            }
        }
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        logger.info(f"Created experiment {experiment_id}: {control_model_version} vs {treatment_model_version}")
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> None:
        """
        Start an experiment (begin routing traffic).
        
        Args:
            experiment_id: Experiment ID to start
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment['status'] = 'running'
        experiment['started_at'] = datetime.now().isoformat()
        
        self._save_experiments()
        
        logger.info(f"Started experiment {experiment_id}")
    
    def stop_experiment(self, experiment_id: str) -> None:
        """
        Stop an experiment (stop routing traffic).
        
        Args:
            experiment_id: Experiment ID to stop
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment['status'] = 'stopped'
        experiment['stopped_at'] = datetime.now().isoformat()
        
        self._save_experiments()
        
        logger.info(f"Stopped experiment {experiment_id}")
    
    def archive_experiment(self, experiment_id: str) -> None:
        """
        Archive an experiment.
        
        Args:
            experiment_id: Experiment ID to archive
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment['status'] = 'archived'
        experiment['archived_at'] = datetime.now().isoformat()
        
        self._save_experiments()
        
        logger.info(f"Archived experiment {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.
        
        Args:
            experiment_id: Experiment ID to retrieve
            
        Returns:
            Experiment dictionary
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.experiments[experiment_id].copy()
    
    def route_prediction(
        self,
        experiment_id: str,
        user_id: str,
        request_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route a prediction request to control or treatment.
        
        Uses consistent hashing to ensure same user always gets same treatment.
        
        Args:
            experiment_id: Experiment ID for routing
            user_id: Unique identifier for the user/request
            request_data: Optional request data for routing logic
            
        Returns:
            'control' or 'treatment'
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != 'running':
            # Default to control if experiment not running
            return 'control'
        
        # Use consistent hashing based on user_id
        hash_input = f"{experiment_id}:{user_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Normalize to [0, 1]
        normalized_hash = (hash_value % 1000000) / 1000000.0
        
        # Route based on traffic split
        if normalized_hash < experiment['traffic_split']:
            return 'treatment'
        else:
            return 'control'
    
    def record_results(
        self,
        experiment_id: str,
        group: str,
        results: Union[List[float], np.ndarray]
    ) -> None:
        """
        Record experiment results for analysis.
        
        Args:
            experiment_id: Experiment ID
            group: 'control' or 'treatment'
            results: List of metric values (e.g., prediction accuracy)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if group not in ['control', 'treatment']:
            raise ValueError(f"group must be 'control' or 'treatment', got '{group}'")
        
        experiment = self.experiments[experiment_id]
        
        # Convert to list if numpy array
        if isinstance(results, np.ndarray):
            results = results.tolist()
        
        # Extend existing results
        experiment['results'][group].extend(results)
        
        self._save_experiments()
        
        logger.info(f"Recorded {len(results)} results for {group} in experiment {experiment_id}")
    
    def evaluate_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Evaluate experiment results with statistical significance testing.
        
        Args:
            experiment_id: Experiment ID to evaluate
            
        Returns:
            Dictionary containing statistical analysis results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        control_results = np.array(experiment['results']['control'])
        treatment_results = np.array(experiment['results']['treatment'])
        
        if len(control_results) == 0 or len(treatment_results) == 0:
            return {
                'error': 'Insufficient data',
                'control_count': len(control_results),
                'treatment_count': len(treatment_results)
            }
        
        # Calculate basic statistics
        control_mean = np.mean(control_results)
        treatment_mean = np.mean(treatment_results)
        control_std = np.std(control_results)
        treatment_std = np.std(treatment_results)
        
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_results, control_results, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_results) - 1) * control_std**2 + 
                             (len(treatment_results) - 1) * treatment_std**2) / 
                            (len(control_results) + len(treatment_results) - 2))
        
        if pooled_std > 0:
            effect_size = (treatment_mean - control_mean) / pooled_std
        else:
            effect_size = 0.0
        
        # Calculate confidence interval for difference in means
        std_error = np.sqrt(control_std**2 / len(control_results) + 
                           treatment_std**2 / len(treatment_results))
        
        # 95% confidence interval
        degrees_freedom = len(control_results) + len(treatment_results) - 2
        t_critical = stats.t.ppf(0.975, degrees_freedom)
        
        difference = treatment_mean - control_mean
        margin_error = t_critical * std_error
        
        confidence_interval = {
            'lower': difference - margin_error,
            'upper': difference + margin_error
        }
        
        # Determine significance
        significant = p_value < 0.05
        
        logger.info(f"Experiment {experiment_id} evaluation: p={p_value:.4f}, significant={significant}")
        
        return {
            'experiment_id': experiment_id,
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'control_std': float(control_std),
            'treatment_std': float(treatment_std),
            'control_count': len(control_results),
            'treatment_count': len(treatment_results),
            'difference': float(difference),
            'effect_size': float(effect_size),
            'p_value': float(p_value),
            'significant': significant,
            'confidence_interval': confidence_interval,
            't_statistic': float(t_stat),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all active (running) experiments.
        
        Returns:
            List of active experiment dictionaries
        """
        active = []
        for exp_id, experiment in self.experiments.items():
            if experiment['status'] == 'running':
                active.append(experiment.copy())
        
        return active
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get a summary of experiment performance.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Summary dictionary with key metrics
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        evaluation = self.evaluate_experiment(experiment_id)
        
        summary = {
            'experiment_id': experiment_id,
            'experiment_name': experiment['experiment_name'],
            'status': experiment['status'],
            'created_at': experiment['created_at'],
            'models': {
                'control': experiment['control_model'],
                'treatment': experiment['treatment_model']
            },
            'traffic_split': experiment['traffic_split']
        }
        
        if 'error' not in evaluation:
            summary.update({
                'sample_sizes': {
                    'control': evaluation['control_count'],
                    'treatment': evaluation['treatment_count']
                },
                'performance': {
                    'control_mean': evaluation['control_mean'],
                    'treatment_mean': evaluation['treatment_mean'],
                    'improvement': evaluation['difference'],
                    'improvement_percent': (evaluation['difference'] / evaluation['control_mean'] * 100) if evaluation['control_mean'] != 0 else 0
                },
                'statistical_significance': {
                    'p_value': evaluation['p_value'],
                    'significant': evaluation['significant'],
                    'effect_size': evaluation['effect_size']
                }
            })
        else:
            summary['evaluation_status'] = evaluation['error']
        
        return summary
    
    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load experiments from disk."""
        if not os.path.exists(self.experiments_file):
            return {}
        
        try:
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load experiments file, starting fresh")
            return {}
    
    def _save_experiments(self) -> None:
        """Save experiments to disk."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
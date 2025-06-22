"""
Hyperparameter Optimization Engine for ML pipeline.

Stage 3 enhancement: Automated hyperparameter tuning using cross-validation
with support for grid search, random search, and Bayesian optimization.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import logging
import itertools
import random

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization engine for LightGBM models.
    
    Provides multiple optimization strategies including grid search,
    random search, and Bayesian optimization with cross-validation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        logger.info(f"HyperparameterOptimizer initialized with random_state={random_state}")
    
    def optimize(
        self, 
        data: pd.DataFrame,
        param_space: Dict[str, List[Any]],
        cv_folds: int = 5,
        strategy: str = 'grid',
        max_evals: Optional[int] = None,
        time_limit_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using cross-validation.
        
        Args:
            data: Training data with features and target
            param_space: Dictionary defining parameter search space
            cv_folds: Number of cross-validation folds
            strategy: Optimization strategy ('grid', 'random', 'bayesian')
            max_evals: Maximum number of evaluations (for random/bayesian)
            time_limit_minutes: Time limit for optimization in minutes
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        # Prepare data
        target_col = 'pass_percentage'
        categorical_cols = [col for col in ['dataset_uuid', 'rule_code'] 
                           if col in data.columns]
        feature_cols = [col for col in data.columns 
                       if col not in categorical_cols + [target_col]]
        
        X = data[feature_cols + categorical_cols]
        y = data[target_col]
        
        logger.info(f"Starting {strategy} optimization with {len(X)} samples, {cv_folds} folds")
        
        if strategy == 'grid':
            result = self._grid_search(X, y, param_space, cv_folds, categorical_cols, time_limit_minutes)
        elif strategy == 'random':
            max_evals = max_evals or 50
            result = self._random_search(X, y, param_space, cv_folds, categorical_cols, max_evals, time_limit_minutes)
        elif strategy == 'bayesian':
            max_evals = max_evals or 30
            result = self._bayesian_search(X, y, param_space, cv_folds, categorical_cols, max_evals, time_limit_minutes)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        optimization_time = time.time() - start_time
        result['optimization_time'] = optimization_time
        result['strategy'] = strategy
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s, best score: {result['best_score']:.4f}")
        
        return result
    
    def bayesian_search(
        self,
        data: pd.DataFrame,
        param_space: Dict[str, List[Any]],
        n_trials: int = 30,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Bayesian optimization for efficient parameter search.
        
        Args:
            data: Training data with features and target
            param_space: Dictionary defining parameter search space
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing optimization results
        """
        return self.optimize(
            data=data,
            param_space=param_space,
            cv_folds=cv_folds,
            strategy='bayesian',
            max_evals=n_trials
        )
    
    def _grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, List[Any]],
        cv_folds: int,
        categorical_cols: List[str],
        time_limit_minutes: Optional[int]
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        param_grid = list(ParameterGrid(param_space))
        
        logger.info(f"Grid search: evaluating {len(param_grid)} parameter combinations")
        
        best_score = float('inf')
        best_params = None
        cv_results = []
        start_time = time.time()
        
        for i, params in enumerate(param_grid):
            # Check time limit
            if time_limit_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes > time_limit_minutes:
                    logger.warning(f"Time limit reached after {i} evaluations")
                    break
            
            score = self._evaluate_params(X, y, params, cv_folds, categorical_cols)
            cv_results.append({
                'params': params.copy(),
                'score': score,
                'evaluation_order': i
            })
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
            
            if i % 10 == 0:
                logger.debug(f"Grid search progress: {i+1}/{len(param_grid)}, best score: {best_score:.4f}")
        
        result = {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results,
            'total_evaluations': len(cv_results)
        }
        
        # Add early stop reason if time limit was hit
        if time_limit_minutes and len(cv_results) < len(param_grid):
            result['early_stop_reason'] = 'time_limit'
        
        return result
    
    def _random_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, List[Any]],
        cv_folds: int,
        categorical_cols: List[str],
        max_evals: int,
        time_limit_minutes: Optional[int]
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        logger.info(f"Random search: evaluating up to {max_evals} parameter combinations")
        
        best_score = float('inf')
        best_params = None
        cv_results = []
        start_time = time.time()
        
        for i in range(max_evals):
            # Check time limit
            if time_limit_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes > time_limit_minutes:
                    logger.warning(f"Time limit reached after {i} evaluations")
                    break
            
            # Sample random parameters
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = random.choice(param_values)
            
            score = self._evaluate_params(X, y, params, cv_folds, categorical_cols)
            cv_results.append({
                'params': params.copy(),
                'score': score,
                'evaluation_order': i
            })
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
            
            if i % 10 == 0:
                logger.debug(f"Random search progress: {i+1}/{max_evals}, best score: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results,
            'total_evaluations': len(cv_results)
        }
    
    def _bayesian_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, List[Any]],
        cv_folds: int,
        categorical_cols: List[str],
        max_evals: int,
        time_limit_minutes: Optional[int]
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Note: This is a simplified implementation. For production use,
        consider using libraries like scikit-optimize or Optuna.
        """
        logger.info(f"Bayesian search: evaluating up to {max_evals} parameter combinations")
        
        best_score = float('inf')
        best_params = None
        trials_history = []
        start_time = time.time()
        
        # Start with random exploration
        exploration_trials = min(5, max_evals // 3)
        
        for i in range(max_evals):
            # Check time limit
            if time_limit_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes > time_limit_minutes:
                    logger.warning(f"Time limit reached after {i} evaluations")
                    break
            
            if i < exploration_trials:
                # Exploration phase: random sampling
                params = {}
                for param_name, param_values in param_space.items():
                    params[param_name] = random.choice(param_values)
            else:
                # Exploitation phase: focus on promising regions
                params = self._select_next_params(trials_history, param_space)
            
            score = self._evaluate_params(X, y, params, cv_folds, categorical_cols)
            trials_history.append({
                'params': params.copy(),
                'score': score,
                'trial_number': i
            })
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
            
            if i % 5 == 0:
                logger.debug(f"Bayesian search progress: {i+1}/{max_evals}, best score: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trials_history': trials_history,
            'total_evaluations': len(trials_history)
        }
    
    def _select_next_params(self, trials_history: List[Dict], param_space: Dict) -> Dict:
        """
        Select next parameters based on previous trials (simplified acquisition function).
        """
        if len(trials_history) < 3:
            # Not enough history, use random
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = random.choice(param_values)
            return params
        
        # Simple strategy: find best trials and sample around them
        best_trials = sorted(trials_history, key=lambda x: x['score'])[:3]
        base_trial = random.choice(best_trials)
        
        # Modify parameters slightly from best trial
        params = base_trial['params'].copy()
        
        # Randomly modify 1-2 parameters
        param_names = list(param_space.keys())
        num_changes = random.randint(1, min(2, len(param_names)))
        params_to_change = random.sample(param_names, num_changes)
        
        for param_name in params_to_change:
            params[param_name] = random.choice(param_space[param_name])
        
        return params
    
    def _evaluate_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_folds: int,
        categorical_cols: List[str]
    ) -> float:
        """
        Evaluate parameters using cross-validation.
        
        Returns:
            Mean absolute error (lower is better)
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Convert categorical columns to codes for LightGBM
            X_train_processed = X_train.copy()
            X_val_processed = X_val.copy()
            
            categorical_feature_indices = []
            for i, col in enumerate(X.columns):
                if col in categorical_cols:
                    categorical_feature_indices.append(i)
                    # Convert to category and then to codes
                    X_train_processed[col] = X_train_processed[col].astype('category').cat.codes
                    X_val_processed[col] = X_val_processed[col].astype('category').cat.codes
            
            # Create LightGBM datasets
            lgb_train = lgb.Dataset(
                X_train_processed, 
                y_train,
                categorical_feature=categorical_feature_indices
            )
            lgb_val = lgb.Dataset(
                X_val_processed,
                y_val,
                categorical_feature=categorical_feature_indices,
                reference=lgb_train
            )
            
            # Train model with given parameters
            model_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'verbose': -1,
                **params
            }
            
            model = lgb.train(
                model_params,
                lgb_train,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Make predictions and calculate score
            y_pred = model.predict(X_val_processed)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
        
        return np.mean(cv_scores)
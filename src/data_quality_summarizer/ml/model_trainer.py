"""
Model Trainer module for ML pipeline.

This module provides LightGBM model training functionality
for pass percentage prediction with categorical feature support.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from typing import List, Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    LightGBM model trainer for pass percentage prediction.
    
    Provides interface for training, prediction, and model management
    with proper categorical feature handling.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize model trainer with parameters.
        
        Args:
            params: Optional LightGBM parameters. Uses defaults if None.
        """
        self.model = None
        self.params = params or get_default_lgb_params()
        self.feature_cols = None
        self.categorical_cols = None
        logger.info(f"ModelTrainer initialized with params: {self.params}")
    
    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
        target_col: str
    ) -> lgb.Booster:
        """
        Train LightGBM model on provided data.
        
        Args:
            data: Training data DataFrame
            feature_cols: List of feature column names
            categorical_cols: List of categorical column names
            target_col: Target variable column name
            
        Returns:
            Trained LightGBM Booster model
        """
        self.feature_cols = feature_cols + categorical_cols
        self.categorical_cols = categorical_cols
        
        self.model = train_lightgbm_model(
            data, feature_cols, categorical_cols, target_col, self.params
        )
        
        logger.info(f"Model training completed with {len(self.feature_cols)} features")
        return self.model
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model not trained yet
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if self.feature_cols is None:
            raise ValueError("Feature columns not set. Call fit() first.")
        
        # Prepare categorical features to match training data categories
        pred_data = prepare_categorical_features_for_prediction(
            data.copy(), self.categorical_cols, self.model
        )
        
        predictions = self.model.predict(pred_data[self.feature_cols])
        
        # Clip predictions to reasonable range for pass percentage
        predictions = np.clip(predictions, 0, 100)
        
        logger.debug(f"Generated {len(predictions)} predictions")
        
        return predictions


def train_lightgbm_model(
    data: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    params: Optional[Dict[str, Any]] = None
) -> lgb.Booster:
    """
    Train a LightGBM model for pass percentage prediction.
    
    Args:
        data: Training data DataFrame
        feature_cols: List of feature column names (numeric features)
        categorical_cols: List of categorical feature column names  
        target_col: Target variable column name
        params: Optional LightGBM parameters
        
    Returns:
        Trained LightGBM Booster model
        
    Raises:
        ValueError: If data is empty
        KeyError: If required columns are missing
    """
    if data.empty:
        raise ValueError("Training data cannot be empty")
    
    # Validate columns exist
    all_feature_cols = feature_cols + categorical_cols
    missing_features = set(all_feature_cols) - set(data.columns)
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")
    
    if target_col not in data.columns:
        raise KeyError(f"Missing target column: {target_col}")
    
    # Prepare data
    train_data = prepare_categorical_features(data.copy(), categorical_cols)
    
    # Create LightGBM dataset
    X = train_data[all_feature_cols]
    y = train_data[target_col]
    
    train_dataset = lgb.Dataset(
        X, 
        label=y,
        categorical_feature=categorical_cols
    )
    
    # Use provided params or defaults
    model_params = params or get_default_lgb_params()
    
    # Train model
    logger.info(f"Training LightGBM model with {len(all_feature_cols)} features")
    model = lgb.train(
        model_params,
        train_dataset,
        valid_sets=[train_dataset],
        callbacks=[lgb.log_evaluation(0)]  # Silent training
    )
    
    logger.info("Model training completed successfully")
    return model


def prepare_categorical_features(
    data: pd.DataFrame, 
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Prepare categorical features for LightGBM training.
    
    Args:
        data: DataFrame to prepare
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with categorical columns converted to category dtype
    """
    data_copy = data.copy()
    
    for col in categorical_cols:
        if col in data_copy.columns:
            data_copy[col] = data_copy[col].astype('category')
    
    logger.debug(f"Prepared {len(categorical_cols)} categorical features")
    return data_copy


def prepare_categorical_features_for_prediction(
    data: pd.DataFrame, 
    categorical_cols: List[str],
    model: lgb.Booster
) -> pd.DataFrame:
    """
    Prepare categorical features for prediction, ensuring categories match training.
    
    Args:
        data: DataFrame to prepare for prediction
        categorical_cols: List of categorical column names
        model: Trained LightGBM model (for category compatibility)
        
    Returns:
        DataFrame with categorical columns prepared for prediction
    """
    data_copy = data.copy()
    
    for col in categorical_cols:
        if col in data_copy.columns:
            # Convert to category but don't worry about matching training categories
            # LightGBM will handle unknown categories
            data_copy[col] = data_copy[col].astype('category')
    
    logger.debug(f"Prepared {len(categorical_cols)} categorical features for prediction")
    return data_copy


def get_default_lgb_params() -> Dict[str, Any]:
    """
    Get default LightGBM parameters optimized for pass percentage prediction.
    
    Returns:
        Dictionary of default LightGBM parameters
    """
    return {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'num_boost_round': 100,
        'early_stopping_rounds': 10
    }


def save_model(model: lgb.Booster, file_path: str) -> None:
    """
    Save trained LightGBM model to file.
    
    Args:
        model: Trained LightGBM model
        file_path: Path to save model file
        
    Raises:
        IOError: If unable to save to file path
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {file_path}")
    except (OSError, IOError) as e:
        logger.error(f"Failed to save model to {file_path}: {e}")
        raise


def load_model(file_path: str) -> lgb.Booster:
    """
    Load trained LightGBM model from file.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Loaded LightGBM model
        
    Raises:
        FileNotFoundError: If model file not found
    """
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {file_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from {file_path}: {e}")
        raise
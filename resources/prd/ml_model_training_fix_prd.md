# **PRD: ML Model Training Issue Fix - Data Quality Summarizer**

**Version:** 1.0  
**Author:** Claude Code Analysis  
**Status:** Approved  
**Date:** 2025-06-22  
**Related:** [Session Scratchpad](../context/session-scratchpad.md) | [Predictive Model PRD](predictive_model_prd.md)

---

## **1. Executive Summary**

This PRD addresses a **CRITICAL** issue discovered in the Data Quality Summarizer's ML prediction pipeline where the trained LightGBM model consistently predicts 0.0% pass percentage for ALL inputs, regardless of feature values. The issue was identified through comprehensive validation analysis showing Mean Absolute Error of 51.9% and complete absence of learning from features.

### **Impact Assessment**
- **Severity:** CRITICAL - Complete model failure
- **Affected Component:** ML Pipeline (`src/data_quality_summarizer/ml/`)
- **Current State:** Model training completes successfully but produces non-functional predictions
- **Business Impact:** Predictive data quality monitoring is completely non-functional

---

## **2. Problem Statement**

### **2.1 Current Issue**
The ML model training pipeline appears to work correctly but produces a model that exhibits the following critical failures:

- **All Predictions Identical**: 991 test predictions, all returning exactly 0.0%
- **No Feature Learning**: Model ignores all input features and produces constant output
- **High Error Rate**: Mean Absolute Error of 51.9% on validation data
- **No Variance**: Prediction range 0.0% - 0.0% vs actual range 0.0% - 100.0%

### **2.2 Evidence from Comprehensive Analysis**

#### **Validation Results (from `simple_validation.py`)**
```
Total predictions: 991
Unique predictions: 1 (all identical)
All predictions same: TRUE
Actual range: 0.0% - 100.0%
Predicted range: 0.0% - 0.0%
MAE (Mean Absolute Error): 51.9%
RMSE (Root Mean Squared Error): 71.9%
```

#### **Visualization Analysis**
**9-Panel Comprehensive Analysis (`model_prediction_analysis.png`)**:
1. **Predicted vs Actual Scatter**: All points at y=0, showing no correlation
2. **Error Distribution**: Heavily skewed, mean error 51.9%
3. **Error by Rule Code**: Consistent poor performance across all rules (101-104)
4. **Distribution Comparison**: Predicted (single spike at 0), Actual (distributed 0-100%)
5. **Time Series**: Consistent errors across all dates
6. **Error by Dataset**: All datasets show identical poor performance
7. **Performance Metrics**: MAE: 51.9%, MSE: 5176.8, RMSE: 71.9
8. **Residuals Plot**: All residuals negative, systematic bias
9. **Quality Assessment**: ❌ CRITICAL - All predictions identical

**4-Panel Detailed Error Analysis (`detailed_error_analysis.png`)**:
1. **Error Heatmap**: Consistent high errors across rule codes and months
2. **Cumulative Error Distribution**: 50% of predictions have >50% error
3. **Error-Colored Scatter**: High error magnitude for all non-zero actuals
4. **Detailed Statistics Table**: Comprehensive metrics showing model failure

---

## **3. Root Cause Analysis**

### **3.1 Identified Technical Issues**

#### **A. Data Preprocessing Problems**
**Location**: `src/data_quality_summarizer/ml/aggregator.py:52-70`
- **Issue**: Target variable (`pass_percentage`) may have insufficient variance
- **Evidence**: Empty groups default to 0.0% (`handle_empty_groups()`)
- **Impact**: Model may be learning to predict the dominant class (0.0%)

#### **B. Feature Engineering Issues**
**Location**: `src/data_quality_summarizer/ml/feature_engineer.py:52-111`
- **Issue**: Lag features return `pd.NA` for insufficient historical data
- **Evidence**: Lines 89-104 show NaN handling for missing lag dates
- **Impact**: Most features are NaN/0.0, providing no signal to model

#### **C. Model Training Configuration**
**Location**: `src/data_quality_summarizer/ml/model_trainer.py:412-430`
- **Issue**: Default LightGBM parameters may be inadequate
- **Current Parameters**:
  ```python
  'num_boost_round': 100,
  'early_stopping_rounds': 10,  # Too aggressive
  'learning_rate': 0.05         # May be too low
  ```
- **Impact**: Training may stop before meaningful learning occurs

#### **D. Feature Mismatch (Critical)**
**Location**: `src/data_quality_summarizer/ml/predictor.py:147-204`
- **Issue**: Training uses 11 features (9 numeric + 2 categorical) but prediction inconsistently handles categorical features
- **Evidence**: Comments in `_prepare_model_input()` indicate "fixes 9 vs 11 feature mismatch"
- **Impact**: Inconsistent feature handling between training and prediction

### **3.2 Data Quality Hypothesis**
**Training Data Analysis**:
- **Source**: `demo_subset.csv` (999 rows from `large_100k_test.csv`)
- **Groups**: 991 unique (dataset_uuid, rule_code, business_date) combinations
- **Issue**: Limited historical depth for lag features and moving averages
- **Impact**: Most engineered features are zero/NaN, reducing model signal

---

## **4. Technical Requirements**

### **4.1 Data Validation & Preprocessing (Priority: CRITICAL)**

#### **R4.1.1 Target Variable Analysis**
- **Requirement**: Add comprehensive target distribution validation before training
- **Implementation**: 
  - Log pass_percentage statistics (min, max, mean, std, null count)
  - Validate sufficient variance exists (std > 0.1)
  - Flag datasets with >90% zero values
- **Location**: `model_trainer.py:295-317` (in `train_lightgbm_model()`)

#### **R4.1.2 Feature Imputation Strategy**
- **Requirement**: Replace NaN values in engineered features with domain-appropriate defaults
- **Implementation**:
  - Lag features: Use dataset-rule historical average or 50.0%
  - Moving averages: Use global dataset average or 50.0%
  - Time features: Keep as-is (never NaN)
- **Location**: `feature_engineer.py:52-192` (in all feature functions)

#### **R4.1.3 Data Quality Gates**
- **Requirement**: Implement training data quality checks
- **Validation Rules**:
  - Minimum 50 samples per dataset-rule combination
  - At least 30% non-zero pass_percentage values
  - Feature matrix rank > 0.8 * feature_count
- **Location**: New function in `model_trainer.py`

### **4.2 Feature Engineering Improvements (Priority: HIGH)**

#### **R4.2.1 Robust Lag Feature Calculation**
- **Current Issue**: Lines 88-104 in `feature_engineer.py` use strict date matching
- **Requirement**: Implement nearest-neighbor lag feature calculation
- **Implementation**:
  ```python
  # Instead of exact date match, find closest prior date within window
  def find_closest_lag_value(group_sorted, current_date, lag_days, tolerance_days=3):
      target_date = current_date - pd.Timedelta(days=lag_days)
      valid_dates = group_sorted[group_sorted['business_date'] <= target_date]
      if len(valid_dates) > 0:
          closest_row = valid_dates.iloc[-1]  # Most recent prior date
          return closest_row['pass_percentage']
      return np.nan
  ```

#### **R4.2.2 Enhanced Moving Average Calculation**
- **Current Issue**: Lines 145-153 require exact window size
- **Requirement**: Use available data when full window not available
- **Implementation**: Change `window_values` to use `min_periods=1` logic

#### **R4.2.3 Feature Importance Logging**
- **Requirement**: Log feature importance after training
- **Implementation**: Add `model.feature_importance()` analysis in training pipeline
- **Output**: Feature importance visualization saved to `model_diagnostics/`

### **4.3 Model Training Configuration (Priority: HIGH)**

#### **R4.3.1 Optimized LightGBM Parameters**
- **Current Parameters** (lines 419-429 in `model_trainer.py`):
  ```python
  # BEFORE (problematic)
  'num_boost_round': 100,
  'early_stopping_rounds': 10,
  'learning_rate': 0.05
  ```
- **Required Parameters**:
  ```python
  # AFTER (optimized for this use case)
  'objective': 'regression',
  'metric': 'mae',
  'num_leaves': 31,
  'learning_rate': 0.1,              # Increased from 0.05
  'feature_fraction': 0.9,
  'bagging_fraction': 0.8,
  'bagging_freq': 5,
  'min_data_in_leaf': 10,            # NEW - prevent overfitting
  'min_sum_hessian_in_leaf': 1e-3,   # NEW - stability
  'verbosity': 0,                    # Enable training logs
  'num_boost_round': 300,            # Increased from 100
  'early_stopping_rounds': 50,       # Increased from 10
  'valid_sets': [train_dataset],     # Enable validation monitoring
  'callbacks': [lgb.log_evaluation(10)]  # Log every 10 rounds
  ```

#### **R4.3.2 Training Diagnostics**
- **Requirement**: Comprehensive training monitoring
- **Implementation**:
  1. **Pre-training validation**:
     - Feature correlation matrix
     - Target variable distribution analysis
     - Sample size validation per group
  2. **During training**:
     - Validation loss tracking
     - Feature importance evolution
     - Prediction distribution monitoring
  3. **Post-training validation**:
     - Prediction range validation (should not be constant)
     - Cross-validation with different splits
     - Synthetic data validation

### **4.4 Prediction Pipeline Fix (Priority: CRITICAL)**

#### **R4.4.1 Feature Consistency Fix**
- **Current Issue**: `predictor.py:164-174` mentions "9 vs 11 feature mismatch"
- **Requirement**: Ensure identical feature handling between training and prediction
- **Implementation**:
  1. Extract feature selection logic to shared utility function
  2. Use identical categorical feature preparation in both pipelines
  3. Add feature count validation in prediction pipeline

#### **R4.4.2 Categorical Feature Handling**
- **Location**: `predictor.py:192-194`
- **Current Issue**: Inconsistent categorical preparation
- **Requirement**: Match training categorical preparation exactly
- **Implementation**:
  ```python
  # Ensure categorical features use same dtype and categories as training
  for col in categorical_cols:
      if col in data_copy.columns:
          # Must match training categories exactly
          data_copy[col] = data_copy[col].astype('category')
  ```

### **4.5 Fallback Strategy (Priority: MEDIUM)**

#### **R4.5.1 Baseline Predictor**
- **Requirement**: Implement historical average fallback
- **Implementation**: If model predicts constant values, fall back to dataset-rule historical average
- **Location**: New class `BaselinePredictor` in `predictor.py`

#### **R4.5.2 Prediction Validation**
- **Requirement**: Runtime prediction quality checks
- **Validation Rules**:
  - Reject if all predictions in batch are identical
  - Warning if prediction outside [0, 100] range
  - Error if prediction is NaN or infinite
- **Location**: `_validate_and_clip_prediction()` enhancement

---

## **5. Implementation Plan**

### **5.1 Phase 1: Data Preprocessing Fix (Priority: CRITICAL)**

#### **Task 5.1.1: Add Target Variable Validation**
- **File**: `src/data_quality_summarizer/ml/model_trainer.py`
- **Location**: Lines 295-317 (in `train_lightgbm_model()`)
- **Changes**:
  ```python
  def validate_training_data(data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
      """Validate training data quality before model training."""
      target_stats = {
          'count': len(data[target_col]),
          'null_count': data[target_col].isnull().sum(),
          'zero_count': (data[target_col] == 0).sum(),
          'mean': data[target_col].mean(),
          'std': data[target_col].std(),
          'min': data[target_col].min(),
          'max': data[target_col].max()
      }
      
      # Validation checks
      if target_stats['std'] < 0.1:
          logger.warning("Low target variance detected", std=target_stats['std'])
      
      zero_percentage = (target_stats['zero_count'] / target_stats['count']) * 100
      if zero_percentage > 90:
          logger.warning("High zero percentage in target", percentage=zero_percentage)
      
      return target_stats
  ```

#### **Task 5.1.2: Implement Feature Imputation**
- **File**: `src/data_quality_summarizer/ml/feature_engineer.py`
- **Location**: Lines 88-104 (lag features), 145-153 (moving averages)
- **Changes**: Replace `pd.NA` with domain-appropriate defaults

#### **Task 5.1.3: Add Data Quality Gates**
- **File**: `src/data_quality_summarizer/ml/model_trainer.py`
- **Location**: New function before training
- **Implementation**: Minimum sample size and variance validation

### **5.2 Phase 2: Model Configuration Fix (Priority: HIGH)**

#### **Task 5.2.1: Update LightGBM Parameters**
- **File**: `src/data_quality_summarizer/ml/model_trainer.py`
- **Location**: Lines 412-430 (in `get_default_lgb_params()`)
- **Changes**: Apply optimized parameters from R4.3.1

#### **Task 5.2.2: Add Training Diagnostics**
- **File**: `src/data_quality_summarizer/ml/model_trainer.py`
- **Location**: Lines 295-357 (in `train_lightgbm_model()`)
- **Implementation**: Add comprehensive logging and validation

### **5.3 Phase 3: Feature Engineering Robustness (Priority: HIGH)**

#### **Task 5.3.1: Improve Lag Feature Calculation**
- **File**: `src/data_quality_summarizer/ml/feature_engineer.py`
- **Location**: Lines 52-111 (in `create_lag_features()`)
- **Implementation**: Nearest-neighbor lag calculation

#### **Task 5.3.2: Enhanced Moving Averages**
- **File**: `src/data_quality_summarizer/ml/feature_engineer.py`
- **Location**: Lines 113-159 (in `calculate_moving_averages()`)
- **Implementation**: Flexible window size handling

### **5.4 Phase 4: Prediction Pipeline Fix (Priority: CRITICAL)**

#### **Task 5.4.1: Fix Feature Consistency**
- **File**: `src/data_quality_summarizer/ml/predictor.py`
- **Location**: Lines 147-204 (in `_prepare_model_input()`)
- **Implementation**: Ensure 11-feature consistency

#### **Task 5.4.2: Enhance Prediction Validation**
- **File**: `src/data_quality_summarizer/ml/predictor.py`
- **Location**: Lines 206-230 (in `_validate_and_clip_prediction()`)
- **Implementation**: Add constant prediction detection

### **5.5 Phase 5: Testing & Validation (Priority: HIGH)**

#### **Task 5.5.1: Create Synthetic Data Tests**
- **File**: New test file `tests/test_ml_training_fix.py`
- **Implementation**: Test with controlled synthetic data to verify learning

#### **Task 5.5.2: Add Training Diagnostics Script**
- **File**: New script `debug_training_diagnostics.py`
- **Implementation**: Comprehensive training analysis and visualization

#### **Task 5.5.3: Enhanced Validation Pipeline**
- **File**: Update `simple_validation.py`
- **Implementation**: Add constant prediction detection and feature analysis

---

## **6. Success Criteria**

### **6.1 Primary Success Metrics**

#### **S6.1.1 Prediction Variance**
- **Current**: All predictions = 0.0% (1 unique value)
- **Target**: Predictions show variance > 1.0% standard deviation
- **Measurement**: Standard deviation of predictions on validation set

#### **S6.1.2 Mean Absolute Error**
- **Current**: MAE = 51.9%
- **Target**: MAE < 15% on validation data
- **Measurement**: `simple_validation.py` output

#### **S6.1.3 Prediction Range**
- **Current**: 0.0% - 0.0%
- **Target**: Reasonable range (5% - 95% typical)
- **Measurement**: Min/max of validation predictions

### **6.2 Secondary Success Metrics**

#### **S6.2.1 Feature Importance**
- **Target**: Non-zero importance for at least 70% of features
- **Measurement**: LightGBM feature_importance() output

#### **S6.2.2 Model Convergence**
- **Target**: Training loss decreases over iterations
- **Measurement**: LightGBM validation metrics during training

#### **S6.2.3 Cross-Dataset Performance**
- **Target**: Model performs reasonably across different dataset UUIDs
- **Measurement**: Per-dataset MAE analysis

### **6.3 Quality Gates**

#### **Q6.3.1 No Constant Predictions**
- **Validation**: Automated check that predictions have std > 0.1%
- **Location**: Enhanced `simple_validation.py`

#### **Q6.3.2 Feature Consistency**
- **Validation**: Training and prediction use identical feature count and types
- **Location**: Unit tests in ML pipeline

#### **Q6.3.3 Regression Testing**
- **Validation**: Existing functionality (data summarization) remains unaffected
- **Location**: Full test suite `python -m pytest`

---

## **7. Implementation Context**

### **7.1 Project Environment**
- **Working Directory**: `/root/projects/data-quality-summarizer`
- **Virtual Environment**: Configured with all dependencies
- **Testing**: 86% coverage, 302 test cases across 14 files
- **Performance Target**: <2min runtime, <1GB memory for 100k rows

### **7.2 Key Files and Locations**

#### **Core ML Pipeline Files**:
- **Model Training**: `src/data_quality_summarizer/ml/model_trainer.py` (476 lines)
- **Feature Engineering**: `src/data_quality_summarizer/ml/feature_engineer.py` (192 lines)
- **Prediction Service**: `src/data_quality_summarizer/ml/predictor.py` (230+ lines)
- **Pipeline Orchestration**: `src/data_quality_summarizer/ml/pipeline.py` (179+ lines)
- **Data Aggregation**: `src/data_quality_summarizer/ml/aggregator.py` (84 lines)

#### **Validation and Debug Files**:
- **Current Validation**: `simple_validation.py` (141 lines)
- **Visualization**: `create_prediction_graphs.py` (308 lines)
- **Feature Debug**: `test_feature_debug.py` (78 lines)

#### **Test Data**:
- **Training Data**: `demo_subset.csv` (999 rows)
- **Rules Metadata**: `demo_rules.json`
- **Model File**: `demo_model.pkl` (non-functional)
- **Validation Results**: `validation_report.csv` (991 rows)

### **7.3 Session Context**
This PRD is based on comprehensive analysis conducted in a previous session that included:

1. **Codebase Analysis**: Complete review of ML pipeline implementation
2. **Model Training**: Successful training of LightGBM model (1.40 seconds)
3. **Validation Analysis**: 991 predictions analyzed showing complete model failure
4. **Visualization**: Two comprehensive charts generated showing error patterns
5. **Root Cause Investigation**: Detailed code review identifying specific issues

### **7.4 Command References**

#### **Training Commands**:
```bash
# Train model (currently produces non-functional model)
python -m src train-model demo_subset.csv demo_rules.json --output-model demo_model.pkl

# Single prediction (returns 0.0%)
python -m src predict --model demo_model.pkl --dataset-uuid uuid123 --rule-code 101 --date 2024-01-15

# Validation analysis
python simple_validation.py

# Visualization
python create_prediction_graphs.py
```

#### **Development Commands**:
```bash
# Environment setup
python -m venv venv && source venv/bin/activate
pip install -e .

# Testing
python -m pytest --cov=src --cov-report=term-missing

# Type checking
mypy src/
```

---

## **8. Risk Assessment**

### **8.1 Technical Risks**

#### **R8.1.1 Training Data Insufficiency**
- **Risk**: 999 training samples may be insufficient for robust learning
- **Mitigation**: Implement data augmentation or use larger training set
- **Probability**: Medium | **Impact**: High

#### **R8.1.2 Feature Engineering Complexity**
- **Risk**: Lag features and moving averages may introduce data leakage
- **Mitigation**: Strict temporal validation in feature engineering
- **Probability**: Low | **Impact**: High

#### **R8.1.3 Model Architecture Mismatch**
- **Risk**: LightGBM may not be suitable for this time series regression task
- **Mitigation**: Implement alternative models (Random Forest, XGBoost)
- **Probability**: Low | **Impact**: Medium

### **8.2 Implementation Risks**

#### **R8.2.1 Regression Introduction**
- **Risk**: Fixes may break existing functionality
- **Mitigation**: Comprehensive test suite execution before deployment
- **Probability**: Medium | **Impact**: High

#### **R8.2.2 Performance Degradation**
- **Risk**: Enhanced feature engineering may increase training time
- **Mitigation**: Performance benchmarking and optimization
- **Probability**: Medium | **Impact**: Low

### **8.3 Mitigation Strategies**

1. **Incremental Implementation**: Implement fixes in phases with validation at each step
2. **Comprehensive Testing**: Both unit tests and integration tests for all changes
3. **Rollback Plan**: Maintain current codebase state for easy rollback
4. **Performance Monitoring**: Track training time and memory usage during fixes

---

## **9. Dependencies and Assumptions**

### **9.1 Technical Dependencies**
- **Python Environment**: Virtual environment with LightGBM, pandas, scikit-learn
- **Data Availability**: Access to `demo_subset.csv` and `demo_rules.json`
- **Development Tools**: pytest, mypy, visualization libraries (matplotlib, seaborn)

### **9.2 Assumptions**
1. **Data Quality**: Training data represents real-world patterns
2. **Historical Depth**: Sufficient historical data exists for lag features
3. **Model Architecture**: LightGBM is appropriate for regression task
4. **Performance Requirements**: Existing performance targets remain valid
5. **Testing Environment**: Current test infrastructure can validate fixes

### **9.3 External Dependencies**
- **LightGBM Library**: Version compatibility with current environment
- **Pandas**: Support for categorical feature handling
- **File System**: Write access to model and diagnostic output directories

---

## **10. Appendix**

### **10.1 Error Evidence**

#### **Complete Validation Output**:
```
🔍 MODEL VALIDATION ANALYSIS
========================================
📊 Loading test data...
   • Loaded 999 records
🔄 Parsing actual results...
   • Found 991 unique dataset/rule/date combinations
🤖 Loading trained model...
🎯 Making predictions and comparing...

📈 VALIDATION RESULTS:
   • Total predictions: 991
   • Unique predictions: 1
   • All predictions same: True
   • Actual range: 0.0% - 100.0%
   • Predicted range: 0.0% - 0.0%
   • MAE (Mean Absolute Error): 51.90%
   • RMSE (Root Mean Squared Error): 71.94%

🚨 ANALYSIS:
   ❌ CRITICAL ISSUE: All predictions are identical!
   → This suggests the model is not learning from features
   → Possible causes: insufficient training data, feature scaling issues, or overfitting
```

### **10.2 Code Analysis Summary**

#### **Key Files Analyzed**:
1. **model_trainer.py** (476 lines): Main training logic with parameter configuration
2. **feature_engineer.py** (192 lines): Feature creation with NaN handling issues
3. **predictor.py** (230+ lines): Prediction service with feature mismatch comments
4. **pipeline.py** (179+ lines): Training orchestration with feature selection logic
5. **aggregator.py** (84 lines): Data aggregation with empty group defaults

#### **Critical Code Locations**:
- **Feature Mismatch**: `predictor.py:152` comments about "9 vs 11 feature mismatch"
- **NaN Handling**: `feature_engineer.py:89-104` strict date matching causing NaN values
- **Parameter Config**: `model_trainer.py:419-429` potentially inadequate LightGBM settings
- **Empty Groups**: `aggregator.py:73-84` defaulting to 0.0% for empty groups

### **10.3 Visualization Analysis**

The comprehensive visualization analysis generated two detailed charts:

1. **`model_prediction_analysis.png`** (9-panel analysis):
   - Predicted vs Actual scatter plot showing all predictions at y=0
   - Error distribution heavily skewed with mean error 51.9%
   - Consistent poor performance across all rule codes (101-104)
   - Time series analysis showing consistent errors across all dates

2. **`detailed_error_analysis.png`** (4-panel error breakdown):
   - Error heatmap showing consistent high errors across dimensions
   - Cumulative error distribution with 50% of predictions having >50% error
   - Error-colored scatter plot highlighting magnitude of prediction failures
   - Detailed statistics table confirming systematic model bias

These visualizations provide conclusive evidence that the model is not learning meaningful patterns and requires the systematic fixes outlined in this PRD.

---

**Document Status**: Ready for Implementation  
**Next Steps**: Begin Phase 1 implementation with target variable validation and feature imputation  
**Expected Resolution Timeline**: 2-3 development cycles with comprehensive testing at each phase
# **PRD: Predictive Model for Data Quality Pass Percentage**

**Version:** 1.0

**Author:** Cascade

**Status:** Approved

## **1. Introduction**

This document outlines the requirements for a new predictive modeling feature within the Data Quality Summarizer application. The goal is to forecast the daily pass percentage of data quality rules for specific datasets. This will enable proactive monitoring and help teams anticipate potential data quality issues before they impact downstream systems. By predicting the likelihood of rule failures, we can allocate resources more effectively and improve overall data reliability.

## **2. Problem Statement**

Currently, data quality is monitored reactively. Teams can only analyze rule performance after the executions have completed. There is no mechanism to forecast potential drops in data quality for a given day.

This project aims to solve that problem by building a machine learning model that can predict the expected pass percentage for a given `dataset_uuid` and `rule_code` on a future `business_date`.

## **3. Goals and Objectives**

### **3.1. Goals**

*   To develop a robust regression model that predicts the daily pass percentage of a data quality rule on a specific dataset.
*   To create a prediction service that takes `dataset_uuid`, `rule_code`, and `business_date` as input and returns a predicted pass percentage.
*   To ensure the solution is efficient and can run on standard, CPU-only server hardware.
*   To provide a clear and maintainable codebase for the modeling pipeline.

### **3.2. Non-Goals**

*   This model will not predict the outcome of a single, individual rule execution (i.e., it won't predict a specific "Pass" or "Fail").
*   The initial version will not have a user interface for visualizing predictions. This will be a backend service.
*   The model will not perform real-time predictions. It is designed for forecasting on a daily basis.

## **4. Functional Requirements**

*   **Prediction Service**: A function or class must be exposed that accepts the following inputs:
    *   `dataset_uuid` (string)
    *   `rule_code` (integer or string)
    *   `business_date` (string or date object)
*   The service will return a single floating-point number representing the predicted pass percentage (e.g., `87.5`).

## **5. Technical Design and Implementation Details**

This section details the step-by-step technical plan for building the model.

### **5.1. Data Source**

The primary data source for training the model will be the historical execution logs stored in `large_test.csv`. This file contains records of every rule execution.

### **5.2. Modeling Approach: Regression**

This problem will be framed as a **regression task**, not classification. The model's objective is to predict a continuous numerical value (the pass percentage), not a discrete category.

### **5.3. Data Preparation and Feature Engineering**

This is the core of the project. The raw log data will be transformed into a structured dataset suitable for training a time-series model.

1.  **Load Data**: The `large_test.csv` file will be loaded into a Pandas DataFrame.
2.  **Parse `results` Column**: The JSON string in the `results` column will be parsed to extract the "Pass" or "Fail" outcome. A new binary column, `is_pass`, will be created (`1` for "Pass", `0` for "Fail").
3.  **Calculate the Target Variable (`pass_percentage`)**:
    *   The data will be grouped by `dataset_uuid`, `rule_code`, and `business_date`.
    *   For each group, we will calculate the pass percentage using the formula:
        `pass_percentage = (SUM(is_pass) / COUNT(is_pass)) * 100`
    *   This aggregated DataFrame will be the primary dataset for our model.

4.  **Feature Creation**: To give the model predictive power, the following features will be engineered:
    *   **Time-Based Features**: From the `business_date`, we will extract:
        *   Day of the week
        *   Day of the month
        *   Week of the year
        *   Month
    *   **Lag Features**: We will create features that represent the pass percentage from previous days (e.g., 1 day ago, 2 days ago, 7 days ago). This helps the model understand recent trends.
    *   **Moving Averages**: We will calculate rolling averages of the pass percentage over different windows (e.g., 3-day moving average, 7-day moving average). This helps smooth out noise and identify longer-term trends.
    *   **Categorical Features**: `dataset_uuid` and `rule_code` will be treated as categorical features.

### **5.4. Model Selection**

*   **Primary Model**: We will use a **Gradient Boosting Machine (GBM)**.
*   **Library**: We will use the **LightGBM** library. This choice is based on its high performance, speed, and efficiency on CPU-only machines. It also has excellent built-in support for categorical features, which simplifies our pipeline.

### **5.5. Training and Evaluation**

1.  **Data Splitting**: The data must be split chronologically to simulate a real-world forecasting scenario. We will select a cutoff date; all data before the cutoff will be for training, and all data after will be for testing. A random split is not appropriate for this time-series problem.
2.  **Training**: The LightGBM model will be trained on the training dataset.
3.  **Evaluation Metric**: The model's performance will be evaluated using the **Mean Absolute Error (MAE)**. This metric tells us, on average, how many percentage points our prediction is off from the actual value.

### **5.6. Hardware and Dependencies**

*   **Hardware**: The entire solution will be designed to run efficiently on a standard, **CPU-only, consumer-grade machine**. No GPU is required.
*   **Python Libraries**: The following libraries will be added to the project's dependencies (`pyproject.toml`):
    *   `pandas`
    *   `scikit-learn`
    *   `lightgbm`

## **6. Success Metrics**

*   The primary success metric will be a low Mean Absolute Error (MAE) on the held-out test set. A target MAE will be established after an initial baseline model is built.
*   The prediction service successfully returns a prediction when given a valid input.
*   The code is well-documented and includes unit tests for the data preparation and prediction logic.

---

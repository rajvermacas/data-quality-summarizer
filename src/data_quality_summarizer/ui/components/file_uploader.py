"""
File upload component for data quality summarizer web interface.

Handles CSV and JSON file uploads with validation for Stage 1 requirements.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional
from io import BytesIO


def validate_csv_file(file_content: bytes) -> Dict[str, Any]:
    """
    Validate uploaded CSV file format and required columns.
    
    Args:
        file_content: Raw file content as bytes
        
    Returns:
        Dict containing validation results with keys:
        - is_valid: Boolean indicating if file is valid
        - error: Error message if validation failed
        - row_count: Number of data rows if valid
    """
    try:
        # Convert bytes to string and read CSV
        df = pd.read_csv(BytesIO(file_content))
        
        # Required columns for data quality summarizer
        required_columns = [
            'source', 'tenant_id', 'dataset_uuid', 'dataset_name',
            'business_date', 'rule_code', 'results',
            'level_of_execution', 'attribute_name'
        ]
        
        # Check for missing columns
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            return {
                "is_valid": False,
                "error": f"Missing required columns: {', '.join(missing_columns)}"
            }
        
        # Validation passed
        return {
            "is_valid": True,
            "row_count": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"CSV parsing error: {str(e)}"
        }


def validate_json_file(file_content: bytes) -> Dict[str, Any]:
    """
    Validate uploaded JSON file format for rule metadata.
    
    Args:
        file_content: Raw file content as bytes
        
    Returns:
        Dict containing validation results with keys:
        - is_valid: Boolean indicating if file is valid
        - error: Error message if validation failed
        - rule_count: Number of rules if valid
    """
    try:
        # Parse JSON content
        rules_data = json.loads(file_content.decode('utf-8'))
        
        # Validate it's a dictionary (rule code -> metadata mapping)
        if not isinstance(rules_data, dict):
            return {
                "is_valid": False,
                "error": "JSON file must contain a dictionary of rule metadata"
            }
        
        # Basic validation - each rule should have name and description
        for rule_code, rule_info in rules_data.items():
            if not isinstance(rule_info, dict):
                return {
                    "is_valid": False,
                    "error": f"Rule {rule_code} metadata must be a dictionary"
                }
        
        # Validation passed
        return {
            "is_valid": True,
            "rule_count": len(rules_data),
            "rules": list(rules_data.keys())
        }
        
    except json.JSONDecodeError as e:
        return {
            "is_valid": False,
            "error": f"Invalid JSON format: {str(e)}"
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"JSON validation error: {str(e)}"
        }


def create_file_uploader() -> Dict[str, Any]:
    """
    Create file uploader UI components for CSV and JSON files.
    
    Returns:
        Dict containing uploader state with uploaded files
    """
    try:
        import streamlit as st
        
        # Create columns for side-by-side upload
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data File")
            csv_file = st.file_uploader(
                "Upload CSV data file",
                type=['csv'],
                help="Upload your data quality check results in CSV format"
            )
        
        with col2:
            st.subheader("📋 Rule Metadata")
            json_file = st.file_uploader(
                "Upload JSON rule metadata",
                type=['json'],
                help="Upload rule definitions and metadata in JSON format"
            )
        
        return {
            "csv_file": csv_file,
            "json_file": json_file
        }
        
    except ImportError:
        # For testing without Streamlit
        return {
            "csv_file": None,
            "json_file": None
        }
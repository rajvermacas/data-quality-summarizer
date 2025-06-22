"""
Data loader module for ML pipeline.

Handles CSV loading, validation, and JSON parsing of results column.
"""
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


def load_and_validate_csv(file_path: str) -> pd.DataFrame:
    """
    Load and validate CSV file structure.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with validated structure
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If file doesn't exist
    """
    required_columns = [
        'source', 'tenant_id', 'dataset_uuid', 'dataset_name',
        'business_date', 'rule_code', 'results', 'level_of_execution',
        'attribute_name'
    ]
    
    try:
        df = pd.read_csv(file_path)
        logger.info("CSV file loaded successfully", rows=len(df), file=file_path)
    except FileNotFoundError:
        logger.error("CSV file not found", file=file_path)
        raise
    
    # Check for required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error("Missing required columns", missing=list(missing_columns))
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("CSV structure validated successfully", columns=len(df.columns))
    return df


def parse_results_column(json_strings: List[str]) -> List[Optional[Dict[str, Any]]]:
    """
    Parse JSON strings from results column.
    
    Args:
        json_strings: List of JSON string values
        
    Returns:
        List of parsed dictionaries or None for invalid JSON
    """
    parsed_results = []
    
    for json_str in json_strings:
        try:
            parsed = json.loads(json_str)
            parsed_results.append(parsed)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON result", json_string=json_str)
            parsed_results.append(None)
    
    logger.info("JSON parsing completed", 
                total=len(json_strings), 
                successful=sum(1 for r in parsed_results if r is not None))
    
    return parsed_results


def create_binary_pass_column(parsed_results: List[Optional[Dict[str, Any]]]) -> List[int]:
    """
    Create binary pass column from parsed results.
    
    Handles both 'status' and 'result' fields in JSON with case-insensitive matching.
    
    Args:
        parsed_results: List of parsed JSON dictionaries
        
    Returns:
        List of binary values (1 for pass, 0 for fail)
    """
    binary_column = []
    
    for result in parsed_results:
        if result is None:
            # Default to fail for malformed results
            binary_column.append(0)
        else:
            # Check for both 'status' and 'result' fields (case-insensitive)
            status = result.get('status', '').lower()
            result_field = result.get('result', '').lower()
            
            # Consider it a pass if either field indicates pass
            if status == 'pass' or result_field == 'pass':
                binary_column.append(1)
            else:
                binary_column.append(0)
    
    pass_count = sum(binary_column)
    logger.info("Binary pass column created", 
                total=len(binary_column), 
                passes=pass_count, 
                pass_rate=pass_count/len(binary_column) if binary_column else 0)
    
    return binary_column


def normalize_rule_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize rule codes to integer format with comprehensive logging.
    
    Args:
        df: DataFrame with 'rule_code' column
        
    Returns:
        DataFrame with normalized integer rule codes
        
    Raises:
        ValueError: If no valid rule codes remain after conversion
    """
    from src.data_quality_summarizer.rules import validate_and_convert_rule_code
    
    result_df = df.copy()
    initial_count = len(result_df)
    
    # Apply conversion function
    result_df['rule_code'] = result_df['rule_code'].apply(
        validate_and_convert_rule_code
    )
    
    # Remove rows with invalid rule codes
    result_df = result_df.dropna(subset=['rule_code'])
    final_count = len(result_df)
    
    # Log conversion results
    converted_count = initial_count - final_count
    if converted_count > 0:
        logger.warning("Dropped rows with invalid rule codes", 
                      dropped=converted_count, 
                      initial=initial_count, 
                      final=final_count)
    
    if final_count == 0:
        raise ValueError("No valid rule codes found after conversion")
    
    # Ensure integer type
    result_df['rule_code'] = result_df['rule_code'].astype(int)
    
    logger.info("Rule code normalization completed", 
                initial=initial_count, 
                final=final_count, 
                success_rate=final_count/initial_count if initial_count > 0 else 0)
    return result_df
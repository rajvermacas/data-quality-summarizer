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
            status = result.get('status', '').lower()
            if status == 'pass':
                binary_column.append(1)
            else:
                binary_column.append(0)
    
    pass_count = sum(binary_column)
    logger.info("Binary pass column created", 
                total=len(binary_column), 
                passes=pass_count, 
                pass_rate=pass_count/len(binary_column) if binary_column else 0)
    
    return binary_column
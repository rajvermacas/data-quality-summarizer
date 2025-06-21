"""
Input validation module for ML prediction service.

This module provides validation and sanitization for prediction service inputs,
ensuring proper parameter types and formats before processing.
"""

from datetime import datetime, date
from typing import Union, Dict, Any


class InputValidator:
    """
    Validates and sanitizes inputs for the prediction service.
    
    Handles validation of dataset_uuid, rule_code, and business_date parameters
    with proper error handling and type conversion.
    """
    
    def __init__(self):
        """Initialize the InputValidator."""
        pass
    
    def validate_dataset_uuid(self, dataset_uuid: Any) -> str:
        """
        Validate and sanitize dataset_uuid parameter.
        
        Args:
            dataset_uuid: The dataset UUID to validate
            
        Returns:
            str: Cleaned dataset UUID string
            
        Raises:
            ValueError: If dataset_uuid is invalid
        """
        if not isinstance(dataset_uuid, str):
            raise ValueError("dataset_uuid must be a string")
        
        # Strip whitespace
        cleaned_uuid = dataset_uuid.strip()
        
        if not cleaned_uuid:
            raise ValueError("dataset_uuid cannot be empty")
        
        return cleaned_uuid
    
    def validate_rule_code(self, rule_code: Any) -> str:
        """
        Validate and sanitize rule_code parameter.
        
        Args:
            rule_code: The rule code to validate (string or integer)
            
        Returns:
            str: Rule code as string
            
        Raises:
            ValueError: If rule_code is invalid
        """
        if isinstance(rule_code, str):
            # Strip whitespace
            cleaned_code = rule_code.strip()
            if not cleaned_code:
                raise ValueError("rule_code cannot be empty")
            return cleaned_code
        elif isinstance(rule_code, int):
            return str(rule_code)
        else:
            raise ValueError("rule_code must be a string or integer")
    
    def validate_business_date(self, business_date: Any) -> date:
        """
        Validate and convert business_date parameter.
        
        Args:
            business_date: The business date to validate
            
        Returns:
            date: Validated date object
            
        Raises:
            ValueError: If business_date is invalid
        """
        if business_date is None:
            raise ValueError("business_date cannot be None")
        
        if isinstance(business_date, date) and not isinstance(business_date, datetime):
            return business_date
        elif isinstance(business_date, datetime):
            return business_date.date()
        elif isinstance(business_date, str):
            try:
                parsed_date = datetime.strptime(business_date, "%Y-%m-%d")
                return parsed_date.date()
            except ValueError:
                raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {business_date}")
        else:
            raise ValueError("business_date must be a string, date, or datetime object")
    
    def validate_all_inputs(
        self, 
        dataset_uuid: Any, 
        rule_code: Any, 
        business_date: Any
    ) -> Dict[str, Any]:
        """
        Validate all prediction service inputs.
        
        Args:
            dataset_uuid: The dataset UUID
            rule_code: The rule code 
            business_date: The business date
            
        Returns:
            Dict containing validated inputs
            
        Raises:
            ValueError: If any input is invalid
        """
        return {
            "dataset_uuid": self.validate_dataset_uuid(dataset_uuid),
            "rule_code": self.validate_rule_code(rule_code),
            "business_date": self.validate_business_date(business_date)
        }
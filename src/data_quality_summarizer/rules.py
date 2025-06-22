"""
Rule metadata management system for data quality summarizer.

This module handles loading, validation, and lookup of rule metadata
from JSON configuration files. It provides utilities for enriching
data quality results with rule information.

Features:
- Fast rule metadata lookup by code
- Comprehensive validation of rule metadata structure
- Efficient caching of loaded metadata
- Graceful handling of missing rule codes with warnings
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

from .constants import VALID_CATEGORIES

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RuleMetadata:
    """
    Data class representing rule metadata information.

    Attributes:
        rule_code: Unique identifier for the rule
        rule_name: Human-readable rule name (e.g., "ROW_COUNT")
        rule_type: Type of rule (e.g., "DATASET", "ATTRIBUTE", "BUSINESS_RULE")
        dimension: Quality dimension (e.g., "Completeness", "Correctness")
        rule_description: Detailed description of what the rule validates
        category: Rule category (1-4)
    """

    rule_code: int
    rule_name: str
    rule_type: str
    dimension: str
    rule_description: str
    category: int

    def __str__(self) -> str:
        """Return string representation of the rule."""
        return f"RuleMetadata(code={self.rule_code}, name='{self.rule_name}')"


def validate_and_convert_rule_code(rule_code: Any) -> Optional[int]:
    """
    Validate and convert rule code to integer format.
    
    Supports multiple input formats:
    - Integer: 1, 2, 3 (returned as-is)
    - String with R prefix: 'R001', 'R002' (converted to 1, 2)
    - String numeric: '001', '002' (converted to 1, 2)
    
    Args:
        rule_code: Rule code in any supported format
        
    Returns:
        Integer rule code, or None if invalid
    """
    if isinstance(rule_code, int):
        return rule_code
    elif isinstance(rule_code, str):
        try:
            # Handle 'R001' format
            if rule_code.startswith('R'):
                return int(rule_code[1:])
            # Handle direct string numbers
            else:
                return int(rule_code)
        except ValueError:
            logger.warning(f"Invalid rule code format (cannot convert to int): {rule_code}")
            return None
    else:
        logger.warning(f"Unexpected rule code type: {type(rule_code)}")
        return None


def validate_rule_metadata(rule_code: int, metadata: Dict[str, Any]) -> None:
    """
    Validate rule metadata structure and values.

    Args:
        rule_code: The rule code being validated
        metadata: Dictionary containing rule metadata

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = (
        "rule_name",
        "rule_type",
        "dimension",
        "rule_description",
        "category",
    )

    # Check for required fields
    for field in required_fields:
        if field not in metadata:
            raise ValueError(
                f"Missing required field '{field}' for rule code {rule_code}"
            )

    # Validate rule_type (must be a non-empty string)
    rule_type = metadata["rule_type"]
    if not isinstance(rule_type, str) or not rule_type.strip():
        raise ValueError(
            f"rule_type must be a non-empty string for rule code {rule_code}"
        )

    # Validate category
    category = metadata["category"]
    if not isinstance(category, str) or category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {VALID_CATEGORIES} for rule code {rule_code}")


def load_rule_metadata(json_file_path: str) -> Dict[int, RuleMetadata]:
    """
    Load rule metadata from JSON file.

    Args:
        json_file_path: Path to the JSON file containing rule metadata

    Returns:
        Dictionary mapping rule codes to RuleMetadata objects

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
        ValueError: If rule metadata validation fails
    """
    json_path = Path(json_file_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Rule metadata file not found: {json_file_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {json_file_path}: {e}")
        raise

    rules_dict = {}

    for rule_code_str, metadata in raw_metadata.items():
        # Use the new conversion function to handle both string and integer formats
        rule_code = validate_and_convert_rule_code(rule_code_str)
        if rule_code is None:
            continue

        # Validate metadata structure
        validate_rule_metadata(rule_code, metadata)

        # Create RuleMetadata object
        rule = RuleMetadata(
            rule_code=rule_code,
            rule_name=metadata["rule_name"],
            rule_type=metadata["rule_type"],
            dimension=metadata["dimension"],
            rule_description=metadata["rule_description"],
            category=metadata["category"],
        )

        rules_dict[rule_code] = rule
        logger.debug(f"Loaded rule metadata: {rule}")

    logger.info(
        f"Successfully loaded {len(rules_dict)} rule metadata entries from "
        f"{json_file_path}"
    )
    return rules_dict


def get_rule_by_code(
    rules_dict: Dict[int, RuleMetadata], rule_code: int
) -> Optional[RuleMetadata]:
    """
    Lookup rule metadata by rule code.

    Args:
        rules_dict: Dictionary of rule metadata
        rule_code: Rule code to lookup

    Returns:
        RuleMetadata object if found, None otherwise
    """
    if rule_code not in rules_dict:
        logger.warning(f"Rule code {rule_code} not found in metadata")
        return None

    return rules_dict[rule_code]


def enrich_with_rule_metadata(
    data: Dict[str, Any], rules_dict: Dict[int, RuleMetadata]
) -> Dict[str, Any]:
    """
    Enrich data dictionary with rule metadata information.

    Args:
        data: Dictionary containing at least a 'rule_code' field
        rules_dict: Dictionary of rule metadata

    Returns:
        New dictionary with original data plus rule metadata fields

    Raises:
        KeyError: If 'rule_code' field is missing from input data
    """
    if "rule_code" not in data:
        raise KeyError("Input data must contain 'rule_code' field")

    # Copy original data
    enriched = data.copy()

    rule_code = data["rule_code"]
    rule = get_rule_by_code(rules_dict, rule_code)

    if rule is not None:
        # Add rule metadata fields
        enriched["rule_name"] = rule.rule_name
        enriched["rule_type"] = rule.rule_type
        enriched["dimension"] = rule.dimension
        enriched["rule_description"] = rule.rule_description
        enriched["category"] = rule.category
    else:
        # Add None values for missing rule metadata
        enriched["rule_name"] = None
        enriched["rule_type"] = None
        enriched["dimension"] = None
        enriched["rule_description"] = None
        enriched["category"] = None

    return enriched

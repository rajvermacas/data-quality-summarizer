"""
Constants for the data quality summarizer system.

This module contains system-wide constants used across multiple components.
"""

# Valid rule categories - string format
VALID_CATEGORIES = {"C1", "C2", "C3", "C4"}

# Category descriptions for reference
CATEGORY_DESCRIPTIONS = {
    "C1": "Critical - High priority data quality issues",
    "C2": "High - Important data quality issues", 
    "C3": "Medium - Moderate data quality issues",
    "C4": "Low - Minor data quality issues"
}
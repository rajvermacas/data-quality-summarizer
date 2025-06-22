#!/usr/bin/env python3
"""
Generate test rule metadata for CI/CD pipeline testing.

This script creates realistic rule metadata that matches the expected schema
for the ML pipeline testing in automated environments.
"""

import argparse
import json
from pathlib import Path


def generate_test_rules(output_path: str):
    """
    Generate test rule metadata JSON file.
    
    Args:
        output_path: Path to save the JSON file
    """
    # Generate realistic rule metadata matching the expected schema
    rules = {
        "1": {
            "rule_name": "Completeness Check",
            "rule_type": "Completeness",
            "dimension": "Completeness",
            "rule_description": "Validates that required fields are not null or empty",
            "category": "C1"
        },
        "2": {
            "rule_name": "Format Validation",
            "rule_type": "Validity",
            "dimension": "Validity",
            "rule_description": "Validates data format compliance with expected patterns",
            "category": "V1"
        },
        "3": {
            "rule_name": "Range Check",
            "rule_type": "Accuracy",
            "dimension": "Accuracy",
            "rule_description": "Validates that numeric values are within expected ranges",
            "category": "A1"
        },
        "4": {
            "rule_name": "Uniqueness Check",
            "rule_type": "Uniqueness",
            "dimension": "Uniqueness",
            "rule_description": "Validates that values are unique where required",
            "category": "U1"
        },
        "5": {
            "rule_name": "Consistency Check",
            "rule_type": "Consistency",
            "dimension": "Consistency",
            "rule_description": "Validates data consistency across related fields",
            "category": "CON1"
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(rules, f, indent=2)
    
    print(f"Generated {len(rules)} test rules and saved to {output_path}")
    
    # Print rules summary for verification
    print("Rules summary:")
    for rule_id, rule_data in rules.items():
        print(f"  Rule {rule_id}: {rule_data['rule_name']} ({rule_data['rule_type']})")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Generate test rule metadata for CI/CD pipeline')
    parser.add_argument('--output', type=str, default='test_rules.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    generate_test_rules(args.output)


if __name__ == '__main__':
    main()
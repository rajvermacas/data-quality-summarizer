"""
Test module for rules.py - Rule metadata management system.

Following TDD approach: RED → GREEN → REFACTOR
These tests will fail initially until src/rules.py is implemented.
"""

import json
import pytest
import tempfile
import os
import logging

from data_quality_summarizer.rules import (
    RuleMetadata,
    load_rule_metadata,
    get_rule_by_code,
    validate_rule_metadata,
    enrich_with_rule_metadata,
)
from data_quality_summarizer.constants import VALID_CATEGORIES


class TestRuleMetadata:
    """Test cases for RuleMetadata class."""

    def test_rule_metadata_creation(self):
        """Test creating a RuleMetadata instance."""
        rule = RuleMetadata(
            rule_code=101,
            rule_name="ROW_COUNT",
            rule_type="DATASET",
            dimension="Completeness",
            rule_description="Validates row count meets expectations",
            category="C1",
        )

        assert rule.rule_code == 101
        assert rule.rule_name == "ROW_COUNT"
        assert rule.rule_type == "DATASET"
        assert rule.dimension == "Completeness"
        assert rule.rule_description == "Validates row count meets expectations"
        assert rule.category == "C1"

    def test_rule_metadata_string_representation(self):
        """Test string representation of RuleMetadata."""
        rule = RuleMetadata(
            rule_code=101,
            rule_name="ROW_COUNT",
            rule_type="DATASET",
            dimension="Completeness",
            rule_description="Validates row count meets expectations",
            category="C1",
        )

        str_repr = str(rule)
        assert "ROW_COUNT" in str_repr
        assert "101" in str_repr


class TestRuleMetadataLoading:
    """Test cases for rule metadata loading from JSON."""

    @pytest.fixture
    def sample_rule_metadata_json(self):
        """Create sample rule metadata JSON for testing."""
        return {
            "101": {
                "rule_name": "ROW_COUNT",
                "rule_type": "DATASET",
                "dimension": "Completeness",
                "rule_description": "Validates row count meets expectations",
                "category": "C1",
            },
            "102": {
                "rule_name": "NULL_CHECK",
                "rule_type": "ATTRIBUTE",
                "dimension": "Correctness",
                "rule_description": "Checks for null values in attributes",
                "category": "C2",
            },
            "103": {
                "rule_name": "DATA_TYPE_CHECK",
                "rule_type": "ATTRIBUTE",
                "dimension": "Correctness",
                "rule_description": "Validates data types match schema",
                "category": "C1",
            },
        }

    @pytest.fixture
    def temp_json_file(self, sample_rule_metadata_json):
        """Create temporary JSON file with sample rule metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_rule_metadata_json, f, indent=2)
            temp_file_name = f.name
        yield temp_file_name
        os.unlink(temp_file_name)

    def test_load_rule_metadata_success(self, temp_json_file):
        """Test successful loading of rule metadata from JSON file."""
        rules_dict = load_rule_metadata(temp_json_file)

        assert len(rules_dict) == 3
        assert 101 in rules_dict
        assert 102 in rules_dict
        assert 103 in rules_dict

        # Check rule 101
        rule_101 = rules_dict[101]
        assert isinstance(rule_101, RuleMetadata)
        assert rule_101.rule_name == "ROW_COUNT"
        assert rule_101.rule_type == "DATASET"
        assert rule_101.dimension == "Completeness"
        assert rule_101.category == "C1"

    def test_load_rule_metadata_file_not_found(self):
        """Test loading rule metadata when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_rule_metadata("nonexistent_file.json")

    def test_load_rule_metadata_invalid_json(self):
        """Test loading rule metadata with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json content }")
            f.flush()

            with pytest.raises(json.JSONDecodeError):
                load_rule_metadata(f.name)

        os.unlink(f.name)

    def test_load_rule_metadata_missing_required_fields(self):
        """Test loading rule metadata with missing required fields."""
        incomplete_metadata = {
            "101": {
                "rule_name": "ROW_COUNT",
                # Missing rule_type, dimension, rule_description, category
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(incomplete_metadata, f)
            f.flush()

            with pytest.raises(ValueError, match="Missing required field"):
                load_rule_metadata(f.name)

        os.unlink(f.name)


class TestRuleLookup:
    """Test cases for rule lookup functionality."""

    @pytest.fixture
    def rules_dict(self):
        """Create a sample rules dictionary for testing."""
        return {
            101: RuleMetadata(
                rule_code=101,
                rule_name="ROW_COUNT",
                rule_type="DATASET",
                dimension="Completeness",
                rule_description="Validates row count meets expectations",
                category="C1",
            ),
            102: RuleMetadata(
                rule_code=102,
                rule_name="NULL_CHECK",
                rule_type="ATTRIBUTE",
                dimension="Correctness",
                rule_description="Checks for null values in attributes",
                category="C2",
            ),
        }

    def test_get_rule_by_code_success(self, rules_dict):
        """Test successful rule lookup by code."""
        rule = get_rule_by_code(rules_dict, 101)
        assert rule is not None
        assert rule.rule_name == "ROW_COUNT"
        assert rule.rule_code == 101

    def test_get_rule_by_code_not_found(self, rules_dict):
        """Test rule lookup when code doesn't exist."""
        rule = get_rule_by_code(rules_dict, 999)
        assert rule is None

    def test_get_rule_by_code_with_logging(self, rules_dict, caplog):
        """Test rule lookup logs warning for missing rule codes."""
        with caplog.at_level(logging.WARNING):
            rule = get_rule_by_code(rules_dict, 999)
            assert rule is None
            assert "Rule code 999 not found" in caplog.text


class TestRuleMetadataValidation:
    """Test cases for rule metadata validation."""

    def test_validate_rule_metadata_valid(self):
        """Test validation of valid rule metadata."""
        metadata = {
            "rule_name": "ROW_COUNT",
            "rule_type": "DATASET",
            "dimension": "Completeness",
            "rule_description": "Validates row count meets expectations",
            "category": "C1",
        }

        # Should not raise any exception
        validate_rule_metadata(101, metadata)

    def test_validate_rule_metadata_missing_rule_name(self):
        """Test validation fails when rule_name is missing."""
        metadata = {
            "rule_type": "DATASET",
            "dimension": "Completeness",
            "rule_description": "Validates row count meets expectations",
            "category": "C1",
        }

        with pytest.raises(ValueError, match="Missing required field 'rule_name'"):
            validate_rule_metadata(101, metadata)

    def test_validate_rule_metadata_invalid_rule_type(self):
        """Test validation fails for invalid rule_type (empty string)."""
        metadata = {
            "rule_name": "ROW_COUNT",
            "rule_type": "",
            "dimension": "Completeness",
            "rule_description": "Validates row count meets expectations",
            "category": "C1",
        }

        with pytest.raises(
            ValueError, match="rule_type must be a non-empty string"
        ):
            validate_rule_metadata(101, metadata)

    def test_validate_rule_metadata_custom_rule_type(self):
        """Test validation passes for custom rule_type values."""
        metadata = {
            "rule_name": "BUSINESS_VALIDATION",
            "rule_type": "BUSINESS_RULE",
            "dimension": "Accuracy",
            "rule_description": "Custom business rule validation",
            "category": "C2",
        }

        # Should not raise any exception
        validate_rule_metadata(105, metadata)

    def test_validate_rule_metadata_invalid_category(self):
        """Test validation fails for invalid category."""
        metadata = {
            "rule_name": "ROW_COUNT",
            "rule_type": "DATASET",
            "dimension": "Completeness",
            "rule_description": "Validates row count meets expectations",
            "category": "C5",  # Invalid category (must be C1-C4)
        }

        with pytest.raises(ValueError, match="category must be one of"):
            validate_rule_metadata(101, metadata)

    def test_validate_rule_metadata_invalid_category_integer(self):
        """Test validation fails for integer category (old format)."""
        metadata = {
            "rule_name": "ROW_COUNT",
            "rule_type": "DATASET",
            "dimension": "Completeness",
            "rule_description": "Validates row count meets expectations",
            "category": 1,  # Should be string "C1", not integer
        }

        with pytest.raises(ValueError, match="category must be one of"):
            validate_rule_metadata(101, metadata)

    def test_validate_rule_metadata_valid_all_categories(self):
        """Test validation passes for all valid string categories."""
        for category in VALID_CATEGORIES:
            metadata = {
                "rule_name": "TEST_RULE",
                "rule_type": "DATASET",
                "dimension": "Completeness",
                "rule_description": "Test rule description",
                "category": category,
            }
            # Should not raise any exception
            validate_rule_metadata(101, metadata)

    def test_validate_rule_metadata_invalid_category_wrong_format(self):
        """Test validation fails for wrong category string format."""
        invalid_categories = ["1", "C5", "c1", "Category1", ""]
        
        for invalid_category in invalid_categories:
            metadata = {
                "rule_name": "TEST_RULE",
                "rule_type": "DATASET", 
                "dimension": "Completeness",
                "rule_description": "Test rule description",
                "category": invalid_category,
            }
            with pytest.raises(ValueError, match="category must be one of"):
                validate_rule_metadata(101, metadata)


class TestRuleEnrichment:
    """Test cases for enriching data with rule metadata."""

    @pytest.fixture
    def rules_dict(self):
        """Create a sample rules dictionary for testing."""
        return {
            101: RuleMetadata(
                rule_code=101,
                rule_name="ROW_COUNT",
                rule_type="DATASET",
                dimension="Completeness",
                rule_description="Validates row count meets expectations",
                category="C1",
            ),
            102: RuleMetadata(
                rule_code=102,
                rule_name="NULL_CHECK",
                rule_type="ATTRIBUTE",
                dimension="Correctness",
                rule_description="Checks for null values in attributes",
                category="C2",
            ),
        }

    def test_enrich_with_rule_metadata_success(self, rules_dict):
        """Test successful enrichment of data with rule metadata."""
        input_data = {
            "rule_code": 101,
            "source": "test_system",
            "tenant_id": "tenant_1",
        }

        enriched = enrich_with_rule_metadata(input_data, rules_dict)

        assert enriched["rule_code"] == 101
        assert enriched["rule_name"] == "ROW_COUNT"
        assert enriched["rule_type"] == "DATASET"
        assert enriched["dimension"] == "Completeness"
        assert enriched["rule_description"] == "Validates row count meets expectations"
        assert enriched["category"] == "C1"
        assert enriched["source"] == "test_system"  # Original data preserved
        assert enriched["tenant_id"] == "tenant_1"  # Original data preserved

    def test_enrich_with_rule_metadata_missing_rule(self, rules_dict, caplog):
        """Test enrichment when rule code is not found."""
        input_data = {
            "rule_code": 999,  # Non-existent rule
            "source": "test_system",
            "tenant_id": "tenant_1",
        }

        with caplog.at_level(logging.WARNING):
            enriched = enrich_with_rule_metadata(input_data, rules_dict)

            # Should preserve original data but add None values for missing rule fields
            assert enriched["rule_code"] == 999
            assert enriched["rule_name"] is None
            assert enriched["rule_type"] is None
            assert enriched["dimension"] is None
            assert enriched["rule_description"] is None
            assert enriched["category"] is None
            assert enriched["source"] == "test_system"
            assert enriched["tenant_id"] == "tenant_1"

            assert "Rule code 999 not found" in caplog.text

    def test_enrich_with_rule_metadata_no_rule_code(self, rules_dict):
        """Test enrichment when input data has no rule_code field."""
        input_data = {"source": "test_system", "tenant_id": "tenant_1"}

        with pytest.raises(KeyError, match="rule_code"):
            enrich_with_rule_metadata(input_data, rules_dict)


class TestRuleMetadataIntegration:
    """Integration tests for rule metadata system."""

    def test_full_workflow_load_and_lookup(self, tmp_path):
        """Test complete workflow: load metadata and perform lookups."""
        # Create test rule metadata file
        metadata = {
            "201": {
                "rule_name": "UNIQUENESS_CHECK",
                "rule_type": "ATTRIBUTE",
                "dimension": "Uniqueness",
                "rule_description": "Validates unique values in attribute",
                "category": "C3",
            },
            "202": {
                "rule_name": "RANGE_CHECK",
                "rule_type": "ATTRIBUTE",
                "dimension": "Correctness",
                "rule_description": "Validates values are within expected range",
                "category": "C1",
            },
        }

        json_file = tmp_path / "test_rules.json"
        with open(json_file, "w") as f:
            json.dump(metadata, f)

        # Load metadata
        rules_dict = load_rule_metadata(str(json_file))
        assert len(rules_dict) == 2

        # Test lookups
        rule_201 = get_rule_by_code(rules_dict, 201)
        assert rule_201.rule_name == "UNIQUENESS_CHECK"
        assert rule_201.dimension == "Uniqueness"

        rule_202 = get_rule_by_code(rules_dict, 202)
        assert rule_202.rule_name == "RANGE_CHECK"
        assert rule_202.category == "C1"

        # Test enrichment
        data = {"rule_code": 201, "value": "test"}
        enriched = enrich_with_rule_metadata(data, rules_dict)
        assert enriched["rule_name"] == "UNIQUENESS_CHECK"
        assert enriched["value"] == "test"

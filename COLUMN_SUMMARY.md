# Data Quality Summarizer - Output CSV Column Reference

## Column Definitions (27 Total)

### Identity Columns
- **source** - Data source identifier
- **tenant_id** - Tenant/organization identifier  
- **dataset_uuid** - Unique dataset identifier
- **dataset_name** - Human-readable dataset name
- **rule_code** - Numeric rule identifier
- **rule_name** - Human-readable rule name

### Rule Metadata
- **rule_type** - Type/category of the rule
- **dimension** - Data dimension being validated
- **rule_description** - Detailed rule description
- **category** - Rule grouping/classification

### Time Grouping
- **week_group** - Sequential week number for grouping
- **business_week_start_date** - Start date of the business week
- **business_week_end_date** - End date of the business week  
- **business_date_latest** - Most recent business date in the week

### Volume Metrics
- **dataset_record_count_total** - Total records in dataset
- **filtered_record_count_total** - Records after filtering

### Quality Results
- **pass_count** - Number of records that passed the rule
- **fail_count** - Number of records that failed the rule
- **warning_count** - Number of records with warnings
- **fail_rate** - Percentage of failures (fail_count / (pass_count + fail_count + warning_count) * 100)

### Trend Analysis
- **previous_period_fail_rate** - Fail rate from previous comparable period
- **trend_flag** - Trend direction: "up", "down", or "equal" - If trend is up then it means failure rate has increased.

### Execution Context
- **last_execution_level** - Level at which rule was last executed

## Week Grouping Logic

The system groups data quality results by **business weeks** with sequential numbering:

- **Week Group 1**: Earliest week containing data
- **Week Group 2**: Next chronological week
- **Week Group N**: Latest week containing data

Each week group spans exactly 7 days (Monday to Sunday) and includes:
- Start/end date boundaries
- All rule executions within that period
- Aggregated pass/fail/warning counts
- Calculated fail rates and trends

This enables time-series analysis of data quality patterns across weekly intervals.
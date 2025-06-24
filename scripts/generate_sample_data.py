#!/usr/bin/env python3
"""
Script to generate comprehensive sample data for the Data Quality Summarizer.
Creates 5000+ rows across 15 datasets and 7 rules from C1-C4 categories.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_data():
    """Generate sample CSV data with realistic data quality check results."""
    
    # Define the datasets
    datasets = [
        {"uuid": "ds001", "name": "Customer Master Data", "source": "CRM_SYSTEM"},
        {"uuid": "ds002", "name": "Product Catalog", "source": "ERP_SYSTEM"},
        {"uuid": "ds003", "name": "Sales Transactions", "source": "POS_SYSTEM"},
        {"uuid": "ds004", "name": "Inventory Records", "source": "WMS_SYSTEM"},
        {"uuid": "ds005", "name": "Employee Directory", "source": "HR_SYSTEM"},
        {"uuid": "ds006", "name": "Financial Accounts", "source": "FINANCE_SYSTEM"},
        {"uuid": "ds007", "name": "Marketing Campaigns", "source": "MARKETING_SYSTEM"},
        {"uuid": "ds008", "name": "Support Tickets", "source": "SUPPORT_SYSTEM"},
        {"uuid": "ds009", "name": "Vendor Information", "source": "PROCUREMENT_SYSTEM"},
        {"uuid": "ds010", "name": "Order Management", "source": "ORDER_SYSTEM"},
        {"uuid": "ds011", "name": "Shipping Records", "source": "LOGISTICS_SYSTEM"},
        {"uuid": "ds012", "name": "Quality Metrics", "source": "QMS_SYSTEM"},
        {"uuid": "ds013", "name": "Compliance Data", "source": "COMPLIANCE_SYSTEM"},
        {"uuid": "ds014", "name": "Asset Management", "source": "ASSET_SYSTEM"},
        {"uuid": "ds015", "name": "Project Portfolio", "source": "PPM_SYSTEM"}
    ]
    
    # Define rule codes (numeric format expected by ingestion)
    rule_codes = [1, 2, 3, 4, 5, 6, 7]
    
    # Define tenant IDs
    tenant_ids = ["tenant_001", "tenant_002", "tenant_003"]
    
    # Define execution levels and attributes
    execution_levels = ["DATASET", "ATTRIBUTE", "RECORD"]
    attributes = [
        "customer_id", "email", "phone", "address", "name", 
        "product_code", "price", "quantity", "date_created",
        "amount", "status", "category", "description"
    ]
    
    # Generate date range (12 months of data)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    def random_date():
        """Generate a random date within the range."""
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        return start_date + timedelta(days=random_days)
    
    def generate_results():
        """Generate realistic pass/fail results."""
        # Bias toward more passes than failures for realistic data
        if random.random() < 0.75:  # 75% chance of pass
            return json.dumps({"result": "Pass"})
        else:
            return json.dumps({"result": "Fail"})
    
    # Generate the CSV data
    rows = []
    target_rows = 5200  # Slightly over 5000 to ensure we meet requirement
    
    for i in range(target_rows):
        dataset = random.choice(datasets)
        tenant_id = random.choice(tenant_ids)
        rule_code = random.choice(rule_codes)
        business_date = random_date().isoformat()
        level_of_execution = random.choice(execution_levels)
        attribute_name = random.choice(attributes) if level_of_execution == "ATTRIBUTE" else ""
        
        # Generate random record counts
        dataset_record_count = random.randint(1000, 50000)
        filtered_record_count = random.randint(int(dataset_record_count * 0.8), dataset_record_count)
        context_id = f"ctx_{random.randint(1000, 9999)}"
        
        row = [
            dataset["source"],           # source
            tenant_id,                   # tenant_id  
            dataset["uuid"],             # dataset_uuid
            dataset["name"],             # dataset_name
            business_date,               # business_date
            dataset_record_count,        # dataset_record_count
            rule_code,                   # rule_code
            level_of_execution,          # level_of_execution
            attribute_name,              # attribute_name
            generate_results(),          # results (JSON string)
            context_id,                  # context_id
            filtered_record_count        # filtered_record_count
        ]
        rows.append(row)
    
    return rows


def main():
    """Generate and save the sample data."""
    print("Generating sample data with 5000+ rows across 15 datasets and 7 rules...")
    
    # Generate the data
    rows = generate_sample_data()
    
    # Define output path
    output_path = Path(__file__).parent.parent / "sample_input.csv"
    
    # Write to CSV
    headers = [
        "source", "tenant_id", "dataset_uuid", "dataset_name",
        "business_date", "dataset_record_count", "rule_code", 
        "level_of_execution", "attribute_name", "results", 
        "context_id", "filtered_record_count"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"✅ Generated {len(rows)} rows of sample data")
    print(f"📁 Saved to: {output_path}")
    
    # Print summary statistics
    datasets_count = len(set(row[2] for row in rows))  # unique dataset_uuids
    rules_count = len(set(row[5] for row in rows))     # unique rule_codes
    tenants_count = len(set(row[1] for row in rows))   # unique tenant_ids
    
    print(f"📊 Summary:")
    print(f"   - Datasets: {datasets_count}")
    print(f"   - Rules: {rules_count}")
    print(f"   - Tenants: {tenants_count}")
    print(f"   - Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
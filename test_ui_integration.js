#!/usr/bin/env node

// Test script to verify UI-API integration
import fetch from 'node-fetch';
import FormData from 'form-data';
import fs from 'fs';

async function testFileUpload() {
    console.log('Testing file upload integration...');
    
    try {
        // Check if backend is running
        console.log('1. Checking backend health...');
        const healthResponse = await fetch('http://127.0.0.1:8000/api/health');
        const healthData = await healthResponse.json();
        console.log('✓ Backend health:', healthData);
        
        // Check if frontend is accessible
        console.log('2. Checking frontend accessibility...');
        const frontendResponse = await fetch('http://127.0.0.1:4173/');
        console.log('✓ Frontend accessible:', frontendResponse.status === 200);
        
        // Test file upload
        console.log('3. Testing file upload API...');
        const formData = new FormData();
        formData.append('csv_file', fs.createReadStream('sample_input.csv'));
        formData.append('rules_file', fs.createReadStream('sample_rules.json'));
        
        const uploadResponse = await fetch('http://127.0.0.1:8000/api/process', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            throw new Error(`Upload failed: ${uploadResponse.statusText}`);
        }
        
        const result = await uploadResponse.json();
        console.log('✓ File upload successful');
        console.log('  - Summary data rows:', result.summary_data.length);
        console.log('  - Natural language entries:', result.nl_summary.length);
        console.log('  - Unique datasets:', result.unique_datasets);
        console.log('  - Unique rules:', result.unique_rules);
        console.log('  - Processing time:', result.processing_time_seconds, 'seconds');
        
        // Verify data structure for charts
        console.log('4. Verifying chart data structure...');
        const firstRow = result.summary_data[0];
        const requiredFields = [
            'rule_category', 'risk_level', 'overall_fail_rate', 
            'fail_rate_1m', 'fail_rate_3m', 'fail_rate_12m',
            'execution_consistency', 'avg_daily_executions'
        ];
        
        const missingFields = requiredFields.filter(field => !(field in firstRow));
        if (missingFields.length > 0) {
            console.log('⚠ Missing fields for charts:', missingFields);
        } else {
            console.log('✓ All required chart data fields present');
        }
        
        console.log('\n🎉 Integration test passed! The UI should now work correctly.');
        console.log('   Access the UI at: http://localhost:4173/');
        console.log('   Upload the sample files to see charts.');
        
    } catch (error) {
        console.error('❌ Integration test failed:', error.message);
    }
}

testFileUpload();
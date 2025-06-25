# Product Requirements Document: UI Chart Visualization Fix

## Document Information
- **Document Version**: 1.0
- **Created**: 2025-06-25
- **Author**: Claude Code Assistant
- **Status**: Draft
- **Priority**: High

## Executive Summary

The Data Quality Summarizer React UI has a critical issue where chart visualizations are not working after successful file upload and processing. The root cause is a data structure mismatch between the backend API response and the frontend chart component expectations. This PRD outlines the comprehensive testing strategy and component analysis required to resolve the visualization issues.

## Problem Statement

### Current Issue
- Users can successfully upload CSV and rules files
- Backend processing completes successfully (5,200 rows processed in ~0.45 seconds)
- API returns structured data with 315 summary entries
- **Charts tab displays broken/empty visualizations**
- Data table and other components may also be affected

### Impact
- **User Experience**: Poor user experience as primary data visualization feature is non-functional
- **Product Value**: Significantly reduces the value proposition of the web interface
- **Adoption**: May drive users back to CLI-only usage

## Root Cause Analysis

### Primary Issues Identified

1. **Field Name Mismatch**
   - **API Returns**: `fail_rate_total`, `fail_count_total`, `pass_count_total`
   - **Frontend Expects**: `overall_fail_rate`, `total_failures`, `total_passes`

2. **Missing Calculated Fields**
   - **Missing**: `risk_level`, `rule_category`, `improvement_needed`
   - **Impact**: Charts and metrics calculations fail

3. **Type Definition Inconsistency**
   - **Issue**: `SummaryRow` interface in `types/common.ts` doesn't match actual API response
   - **Impact**: Type safety compromised, runtime errors

4. **Data Transformation Gap**
   - **Issue**: No transformation layer between API response and UI consumption
   - **Impact**: Direct usage of incompatible data structures

## Component Analysis & Testing Requirements

### 🚨 HIGH PRIORITY COMPONENTS

#### 1. SummaryCharts.tsx
**Location**: `src/data_quality_summarizer/ui/visualizations/SummaryCharts.tsx`

**Issues**:
- References non-existent fields: `rule_category`, `risk_level`, `overall_fail_rate`
- Recharts integration expects specific data structure
- Chart data transformation logic assumes incorrect field names

**Testing Requirements**:
- Verify chart rendering with actual API data
- Test all 4 chart types: Bar, Pie, Line, Scatter
- Validate data transformation logic
- Test responsive behavior and error handling

#### 2. ResultsPage.tsx
**Location**: `src/data_quality_summarizer/ui/pages/ResultsPage.tsx`

**Issues**:
- `calculateMetrics()` function references missing fields
- Metrics calculations will fail with actual API data
- Tab switching to charts will break

**Testing Requirements**:
- Test metrics calculation with real API response
- Verify tab switching functionality
- Test overview metrics display
- Validate download functionality

#### 3. SummaryRow Type Definition
**Location**: `src/data_quality_summarizer/ui/types/common.ts`

**Issues**:
- Interface doesn't match actual API response structure
- Defines 49 fields but API returns different field names
- Type safety is compromised

**Testing Requirements**:
- Align interface with actual API response
- Validate type safety across all components
- Test runtime type checking

### 🔍 MEDIUM PRIORITY COMPONENTS

#### 4. App.tsx
**Location**: `src/data_quality_summarizer/ui/App.tsx`

**Testing Requirements**:
- Test complete file upload flow
- Verify state management and transitions
- Test error handling and recovery
- Validate data passing between components

#### 5. DataTable.tsx
**Location**: `src/data_quality_summarizer/ui/components/DataTable.tsx`

**Potential Issues**:
- May reference incorrect field names
- Sorting and filtering may break with actual data

**Testing Requirements**:
- Test table rendering with real API data
- Verify sorting functionality across all columns
- Test search/filter functionality
- Validate pagination with large datasets
- Test CSV export feature

#### 6. Backend Integration
**Location**: `src/data_quality_summarizer/ui/backend_integration.py`

**Testing Requirements**:
- Validate API response structure consistency
- Test field naming conventions
- Verify data type consistency
- Test error response handling

### ✅ LOW PRIORITY COMPONENTS

#### 7. FileUpload Components
- **FileUpload.tsx**: File validation, drag-and-drop
- **FileUploadPage.tsx**: Information display, layout

#### 8. Processing Components
- **ProcessingPage.tsx**: Progress display, status messages
- **ProgressBar.tsx**: Progress visualization

#### 9. ML Pipeline Components
- **MLPipelinePage.tsx**: ML functionality interface

## Technical Requirements

### Data Contract Standardization

#### Required API Response Fields
```typescript
interface SummaryRow {
  // Identity fields
  source: string
  tenant_id: string
  dataset_uuid: string
  dataset_name: string
  rule_code: number
  
  // Rule metadata (from backend conversion)
  rule_name: string
  rule_type: string
  dimension: string
  rule_description: string
  category: string
  
  // Counts (from aggregation)
  pass_count_total: number
  fail_count_total: number
  pass_count_1m: number
  fail_count_1m: number
  pass_count_3m: number
  fail_count_3m: number
  pass_count_12m: number
  fail_count_12m: number
  
  // Rates (calculated)
  fail_rate_total: number
  fail_rate_1m: number
  fail_rate_3m: number
  fail_rate_12m: number
  
  // Status
  trend_flag: string
  business_date_latest: string
  last_execution_level: string
}
```

#### Required Calculated Fields
```typescript
interface EnhancedSummaryRow extends SummaryRow {
  // Calculated for UI
  overall_fail_rate: number        // = fail_rate_total
  total_failures: number           // = fail_count_total
  total_passes: number             // = pass_count_total
  rule_category: string            // = category
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW'
  improvement_needed: boolean
  latest_business_date: string     // = business_date_latest
}
```

### Implementation Strategy

#### Phase 1: Data Layer Fix (High Priority)
1. **Update Type Definitions**
   - Fix `SummaryRow` interface to match API response
   - Add data transformation interface

2. **Add Data Transformation Layer**
   - Create utility to transform API response to UI format
   - Add field mapping and calculated field generation

3. **Update Backend Response** (if needed)
   - Ensure consistent field naming
   - Add missing calculated fields

#### Phase 2: Component Updates (High Priority)
1. **Fix SummaryCharts.tsx**
   - Update field references to match API response
   - Add data validation and error handling

2. **Fix ResultsPage.tsx**
   - Update `calculateMetrics()` function
   - Fix field references throughout component

#### Phase 3: Comprehensive Testing (Medium Priority)
1. **Integration Testing**
   - Test complete upload-to-visualization flow
   - Validate all chart types render correctly

2. **Component Testing**
   - Test individual components with real data
   - Validate error handling and edge cases

#### Phase 4: Quality Assurance (Low Priority)
1. **Cross-browser Testing**
2. **Performance Testing**
3. **Accessibility Testing**

## Success Criteria

### Primary Goals
- [ ] Charts render correctly with real data
- [ ] All 4 chart types display proper visualizations
- [ ] Metrics calculations work with API response
- [ ] Complete upload-to-visualization flow works

### Secondary Goals
- [ ] Data table displays correctly
- [ ] Search and filtering work properly
- [ ] Export functionality works
- [ ] Error handling is robust

### Performance Requirements
- Charts should render within 2 seconds of tab switch
- UI should remain responsive during data processing
- Memory usage should stay reasonable for large datasets

## Test Cases

### High Priority Test Cases

#### TC-001: Chart Rendering
- **Objective**: Verify all charts render with real API data
- **Steps**: Upload sample files, navigate to charts tab
- **Expected**: All 4 charts display with proper data
- **Status**: Currently Failing

#### TC-002: Metrics Calculation
- **Objective**: Verify overview metrics calculate correctly
- **Steps**: Upload files, check overview tab metrics
- **Expected**: All metrics display correct values
- **Status**: Likely Failing

#### TC-003: Data Table Display
- **Objective**: Verify data table shows correct information
- **Steps**: Upload files, navigate to table tab
- **Expected**: Table shows all 315 rows with correct field values
- **Status**: Unknown

### Medium Priority Test Cases

#### TC-004: Complete Flow
- **Objective**: End-to-end upload to visualization
- **Steps**: Full upload and navigation workflow
- **Expected**: Seamless user experience

#### TC-005: Error Handling
- **Objective**: Graceful handling of data issues
- **Steps**: Various error scenarios
- **Expected**: User-friendly error messages

## Risk Assessment

### High Risk
- **Data Structure Changes**: Risk of breaking existing functionality
- **Type Safety**: Runtime errors if types don't match

### Medium Risk
- **Performance Impact**: Data transformation overhead
- **Browser Compatibility**: Chart rendering across browsers

### Low Risk
- **User Adoption**: Users may prefer CLI if UI is unreliable

## Timeline

### Immediate (1-2 days)
- Fix data contract and type definitions
- Update chart components
- Basic testing

### Short Term (3-5 days)
- Comprehensive component testing
- Integration testing
- Performance validation

### Medium Term (1 week)
- Full QA cycle
- Documentation updates
- User acceptance testing

## Dependencies

### Internal Dependencies
- Backend API stability
- React/TypeScript compatibility
- Recharts library updates

### External Dependencies
- Browser compatibility
- Node.js/npm ecosystem stability

## Success Metrics

### Technical Metrics
- 0 console errors on charts tab
- <2 second chart render time
- 100% test coverage for affected components

### User Experience Metrics
- Charts display correctly for all test cases
- Users can successfully complete upload-to-visualization flow
- No user-reported visualization issues

## Appendix

### Related Documents
- [Original Implementation PRD](./data-quality-summarizer-prd.md)
- [UI Implementation Plan](../development_plan/ui_implementation_3_stage_tdd_plan_20250624_053810.md)

### Technical References
- API Response Schema
- React Component Architecture
- Recharts Documentation

### Change Log
- **v1.0** (2025-06-25): Initial document creation
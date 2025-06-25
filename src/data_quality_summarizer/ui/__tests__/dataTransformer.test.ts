import { transformSummaryData, calculateRiskLevel, calculateImprovementNeeded, calculateExecutionConsistency, ApiSummaryRow, EnhancedSummaryRow } from '../utils/dataTransformer'

describe('Data Transformation Layer', () => {
  // Helper function to create mock API response data
  const mockApiResponse = (): ApiSummaryRow => ({
    // Identity fields
    source: 'test_source',
    tenant_id: 'tenant_123',
    dataset_uuid: 'uuid-123',
    dataset_name: 'Test Dataset',
    rule_code: 'R001',
    
    // Rule metadata (from backend conversion)
    rule_name: 'Test Rule',
    rule_type: 'completeness',
    dimension: 'accuracy',
    rule_description: 'Test rule description',
    category: 'Data Quality',
    
    // Counts (from aggregation)
    pass_count_total: 850,
    fail_count_total: 150,
    pass_count_1m: 280,
    fail_count_1m: 20,
    pass_count_3m: 420,
    fail_count_3m: 80,
    pass_count_12m: 850,
    fail_count_12m: 150,
    
    // Rates (calculated)
    fail_rate_total: 0.15,
    fail_rate_1m: 0.067,
    fail_rate_3m: 0.16,
    fail_rate_12m: 0.15,
    
    // Status
    trend_flag: 'improving',
    business_date_latest: '2024-01-15',
    last_execution_level: 'dataset',
    
    // Additional fields from backend
    dataset_record_count_latest: 1000,
    filtered_record_count_latest: 950
  })

  describe('transformSummaryData', () => {
    it('should transform API response to UI format with field mapping', () => {
      const apiResponse = mockApiResponse()
      const transformed = transformSummaryData([apiResponse])
      
      expect(transformed).toHaveLength(1)
      const result = transformed[0]
      
      // Test field mappings
      expect(result.overall_fail_rate).toBe(apiResponse.fail_rate_total)
      expect(result.total_failures).toBe(apiResponse.fail_count_total)
      expect(result.total_passes).toBe(apiResponse.pass_count_total)
      expect(result.rule_category).toBe(apiResponse.category)
      expect(result.latest_business_date).toBe(apiResponse.business_date_latest)
    })

    it('should calculate risk_level based on fail_rate_total', () => {
      const lowRiskData = { ...mockApiResponse(), fail_rate_total: 0.05 }
      const mediumRiskData = { ...mockApiResponse(), fail_rate_total: 0.15 }
      const highRiskData = { ...mockApiResponse(), fail_rate_total: 0.35 }
      
      const [lowRisk] = transformSummaryData([lowRiskData])
      const [mediumRisk] = transformSummaryData([mediumRiskData])
      const [highRisk] = transformSummaryData([highRiskData])
      
      expect(lowRisk.risk_level).toBe('LOW')
      expect(mediumRisk.risk_level).toBe('MEDIUM')
      expect(highRisk.risk_level).toBe('HIGH')
    })

    it('should calculate improvement_needed based on trends and fail rate', () => {
      const needsImprovement = { ...mockApiResponse(), fail_rate_total: 0.25, trend_flag: 'worsening' }
      const noImprovement = { ...mockApiResponse(), fail_rate_total: 0.05, trend_flag: 'stable' }
      
      const [needs] = transformSummaryData([needsImprovement])
      const [noNeeds] = transformSummaryData([noImprovement])
      
      expect(needs.improvement_needed).toBe(true)
      expect(noNeeds.improvement_needed).toBe(false)
    })

    it('should calculate avg_daily_executions and execution_consistency', () => {
      const apiData = {
        ...mockApiResponse(),
        pass_count_total: 900,
        fail_count_total: 100,
        pass_count_1m: 300,
        fail_count_1m: 0,
        business_date_latest: '2024-01-15'
      }
      
      const [result] = transformSummaryData([apiData])
      
      // Should calculate based on 30-day month
      expect(result.avg_daily_executions).toBeCloseTo(10, 1) // (300 + 0) / 30
      expect(result.execution_consistency).toBeGreaterThan(0)
      expect(result.execution_consistency).toBeLessThanOrEqual(1)
    })

    it('should handle missing fields gracefully', () => {
      const incompleteResponse = {
        source: 'test',
        tenant_id: 'test',
        dataset_uuid: 'test',
        dataset_name: 'test',
        rule_code: 'test',
        // Missing many required fields
      } as Partial<ApiSummaryRow>
      
      // Skip validation for this test as we're testing handling of incomplete data
      expect(() => transformSummaryData([incompleteResponse as ApiSummaryRow], true)).not.toThrow()
      
      const [result] = transformSummaryData([incompleteResponse as ApiSummaryRow], true)
      expect(result.risk_level).toBe('UNKNOWN')
      expect(result.improvement_needed).toBe(false)
      expect(result.avg_daily_executions).toBe(0)
    })

    it('should preserve all original fields from API response', () => {
      const apiResponse = mockApiResponse()
      const [transformed] = transformSummaryData([apiResponse])
      
      // Check that original fields are preserved
      expect(transformed.source).toBe(apiResponse.source)
      expect(transformed.tenant_id).toBe(apiResponse.tenant_id)
      expect(transformed.dataset_uuid).toBe(apiResponse.dataset_uuid)
      expect(transformed.dataset_name).toBe(apiResponse.dataset_name)
      expect(transformed.rule_code).toBe(apiResponse.rule_code)
    })

    it('should handle empty array input', () => {
      const result = transformSummaryData([])
      expect(result).toEqual([])
    })

    it('should handle multiple records correctly', () => {
      const apiResponses = [
        mockApiResponse(),
        { ...mockApiResponse(), rule_code: 'R002', fail_rate_total: 0.25 },
        { ...mockApiResponse(), rule_code: 'R003', fail_rate_total: 0.35 }
      ]
      
      const transformed = transformSummaryData(apiResponses)
      
      expect(transformed).toHaveLength(3)
      expect(transformed[0].risk_level).toBe('MEDIUM')
      expect(transformed[1].risk_level).toBe('HIGH')
      expect(transformed[2].risk_level).toBe('HIGH')
    })
  })

  describe('calculateRiskLevel', () => {
    it('should return LOW for fail rate < 0.1', () => {
      expect(calculateRiskLevel(0)).toBe('LOW')
      expect(calculateRiskLevel(0.05)).toBe('LOW')
      expect(calculateRiskLevel(0.099)).toBe('LOW')
    })

    it('should return MEDIUM for fail rate 0.1-0.2', () => {
      expect(calculateRiskLevel(0.1)).toBe('MEDIUM')
      expect(calculateRiskLevel(0.15)).toBe('MEDIUM')
      expect(calculateRiskLevel(0.2)).toBe('MEDIUM')
    })

    it('should return HIGH for fail rate > 0.2', () => {
      expect(calculateRiskLevel(0.21)).toBe('HIGH')
      expect(calculateRiskLevel(0.5)).toBe('HIGH')
      expect(calculateRiskLevel(1)).toBe('HIGH')
    })

    it('should handle undefined/null values', () => {
      expect(calculateRiskLevel(undefined)).toBe('UNKNOWN')
      expect(calculateRiskLevel(null)).toBe('UNKNOWN')
      expect(calculateRiskLevel(NaN)).toBe('UNKNOWN')
    })
  })

  describe('calculateImprovementNeeded', () => {
    it('should return true for high fail rate with worsening trend', () => {
      expect(calculateImprovementNeeded(0.25, 'worsening')).toBe(true)
      expect(calculateImprovementNeeded(0.3, 'worsening')).toBe(true)
    })

    it('should return true for medium fail rate with worsening trend', () => {
      expect(calculateImprovementNeeded(0.15, 'worsening')).toBe(true)
    })

    it('should return false for low fail rate regardless of trend', () => {
      expect(calculateImprovementNeeded(0.05, 'worsening')).toBe(false)
      expect(calculateImprovementNeeded(0.05, 'stable')).toBe(false)
      expect(calculateImprovementNeeded(0.05, 'improving')).toBe(false)
    })

    it('should return false for improving trend regardless of fail rate', () => {
      expect(calculateImprovementNeeded(0.25, 'improving')).toBe(false)
      expect(calculateImprovementNeeded(0.15, 'improving')).toBe(false)
    })

    it('should handle missing values gracefully', () => {
      expect(calculateImprovementNeeded(undefined, 'worsening')).toBe(false)
      expect(calculateImprovementNeeded(0.25, undefined)).toBe(true) // High fail rate
      expect(calculateImprovementNeeded(undefined, undefined)).toBe(false)
    })
  })

  describe('calculateExecutionConsistency', () => {
    it('should return 1 for perfectly consistent executions', () => {
      // Same number of executions each month
      expect(calculateExecutionConsistency(100, 100, 100)).toBe(1)
      expect(calculateExecutionConsistency(50, 50, 50)).toBe(1)
    })

    it('should return lower value for inconsistent executions', () => {
      // Varying execution counts
      const consistency = calculateExecutionConsistency(100, 50, 25)
      expect(consistency).toBeGreaterThan(0)
      expect(consistency).toBeLessThan(1)
    })

    it('should handle zero executions', () => {
      expect(calculateExecutionConsistency(0, 0, 0)).toBe(0)
      expect(calculateExecutionConsistency(100, 0, 0)).toBeGreaterThan(0)
      expect(calculateExecutionConsistency(0, 100, 0)).toBeGreaterThan(0)
    })

    it('should handle missing values', () => {
      expect(calculateExecutionConsistency(undefined, 100, 100)).toBe(0)
      expect(calculateExecutionConsistency(100, undefined, 100)).toBe(0)
      expect(calculateExecutionConsistency(100, 100, undefined)).toBe(0)
    })
  })

  describe('Type Safety and Validation', () => {
    it('should validate required fields in API response by default', () => {
      const invalidData = { invalid: 'data' } as any
      
      // Should throw validation error by default
      expect(() => transformSummaryData([invalidData])).toThrow()
    })

    it('should handle invalid data gracefully when validation is skipped', () => {
      const invalidData = { invalid: 'data' } as any
      
      // Should handle invalid data without throwing when validation is skipped
      const result = transformSummaryData([invalidData], true)
      expect(result).toHaveLength(1)
      expect(result[0].risk_level).toBe('UNKNOWN')
    })

    it('should ensure all EnhancedSummaryRow fields are present', () => {
      const apiResponse = mockApiResponse()
      const [transformed] = transformSummaryData([apiResponse])
      
      // Check all required UI fields are present
      expect(transformed).toHaveProperty('overall_fail_rate')
      expect(transformed).toHaveProperty('total_failures')
      expect(transformed).toHaveProperty('total_passes')
      expect(transformed).toHaveProperty('rule_category')
      expect(transformed).toHaveProperty('risk_level')
      expect(transformed).toHaveProperty('improvement_needed')
      expect(transformed).toHaveProperty('latest_business_date')
      expect(transformed).toHaveProperty('avg_daily_executions')
      expect(transformed).toHaveProperty('execution_consistency')
    })

    it('should validate API response structure when enabled', () => {
      const validData = mockApiResponse()
      
      // Should not throw for valid data
      expect(() => transformSummaryData([validData])).not.toThrow()
    })

    it('should throw meaningful error for invalid API response', () => {
      const invalidData = {
        source: 'test',
        // Missing other required fields
      }
      
      expect(() => transformSummaryData([invalidData as any])).toThrow(/Invalid API response format/)
    })
  })
})
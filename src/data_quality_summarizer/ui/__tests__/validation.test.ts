import { 
  validateApiSummaryRow, 
  validateApiSummaryData, 
  validateApiProcessingResult,
  safeValidateApiSummaryRow,
  isApiSummaryRow 
} from '../utils/validation'

describe('Runtime Validation', () => {
  const validApiRow = {
    source: 'test_source',
    tenant_id: 'tenant_123',
    dataset_uuid: 'uuid-123',
    dataset_name: 'Test Dataset',
    rule_code: 'R001',
    rule_name: 'Test Rule',
    rule_type: 'completeness',
    dimension: 'accuracy',
    rule_description: 'Test rule description',
    category: 'Data Quality',
    pass_count_total: 850,
    fail_count_total: 150,
    pass_count_1m: 280,
    fail_count_1m: 20,
    pass_count_3m: 420,
    fail_count_3m: 80,
    pass_count_12m: 850,
    fail_count_12m: 150,
    fail_rate_total: 0.15,
    fail_rate_1m: 0.067,
    fail_rate_3m: 0.16,
    fail_rate_12m: 0.15,
    trend_flag: 'improving',
    business_date_latest: '2024-01-15',
    last_execution_level: 'dataset'
  }

  describe('validateApiSummaryRow', () => {
    it('should validate a correct API response', () => {
      const result = validateApiSummaryRow(validApiRow)
      expect(result).toEqual(validApiRow)
    })

    it('should provide defaults for optional fields', () => {
      const minimalRow = {
        source: 'test',
        tenant_id: 'tenant_123',
        dataset_uuid: 'uuid-123',
        dataset_name: 'Test Dataset',
        rule_code: 'R001'
      }
      
      const result = validateApiSummaryRow(minimalRow)
      expect(result.rule_name).toBe('')
      expect(result.category).toBe('')
      expect(result.pass_count_total).toBe(0)
      expect(result.fail_rate_total).toBe(0)
    })

    it('should throw error for missing required fields', () => {
      const invalidRow = {
        source: 'test',
        // Missing required fields
      }
      
      expect(() => validateApiSummaryRow(invalidRow)).toThrow('Invalid API response format')
    })

    it('should throw error for incorrect field types', () => {
      const invalidRow = {
        ...validApiRow,
        pass_count_total: 'not a number' // Should be number
      }
      
      expect(() => validateApiSummaryRow(invalidRow)).toThrow()
    })
  })

  describe('validateApiSummaryData', () => {
    it('should validate an array of API rows', () => {
      const data = [validApiRow, { ...validApiRow, rule_code: 'R002' }]
      const result = validateApiSummaryData(data)
      
      expect(result).toHaveLength(2)
      expect(result[0].rule_code).toBe('R001')
      expect(result[1].rule_code).toBe('R002')
    })

    it('should throw error with row index for invalid data', () => {
      const data = [
        validApiRow,
        { source: 'invalid' } // Missing required fields
      ]
      
      expect(() => validateApiSummaryData(data)).toThrow('Validation failed for row 1')
    })
  })

  describe('validateApiProcessingResult', () => {
    it('should validate complete API response', () => {
      const apiResponse = {
        summary_data: [validApiRow],
        nl_summary: ['Test summary'],
        total_rows_processed: 1000,
        processing_time_seconds: 2.5,
        memory_usage_mb: 50,
        unique_datasets: 1,
        unique_rules: 1,
        time_range: {
          start_date: '2024-01-01',
          end_date: '2024-01-15'
        }
      }
      
      const result = validateApiProcessingResult(apiResponse)
      expect(result).toEqual(apiResponse)
    })

    it('should throw error for invalid processing result', () => {
      const invalidResponse = {
        summary_data: 'not an array',
        // Other fields missing
      }
      
      expect(() => validateApiProcessingResult(invalidResponse)).toThrow('Invalid API response')
    })
  })

  describe('safeValidateApiSummaryRow', () => {
    it('should return success result for valid data', () => {
      const result = safeValidateApiSummaryRow(validApiRow)
      
      expect(result.success).toBe(true)
      if (result.success) {
        expect(result.data).toEqual(validApiRow)
      }
    })

    it('should return error result for invalid data', () => {
      const result = safeValidateApiSummaryRow({ invalid: 'data' })
      
      expect(result.success).toBe(false)
      if (!result.success) {
        expect(result.error).toContain('Invalid API response format')
      }
    })
  })

  describe('isApiSummaryRow', () => {
    it('should return true for valid API row structure', () => {
      expect(isApiSummaryRow(validApiRow)).toBe(true)
    })

    it('should return false for invalid structures', () => {
      expect(isApiSummaryRow(null)).toBe(false)
      expect(isApiSummaryRow(undefined)).toBe(false)
      expect(isApiSummaryRow('string')).toBe(false)
      expect(isApiSummaryRow(123)).toBe(false)
      expect(isApiSummaryRow({})).toBe(false)
      expect(isApiSummaryRow({ source: 'test' })).toBe(false) // Missing other required fields
    })

    it('should check required fields exist', () => {
      const partialRow = {
        source: 'test',
        tenant_id: 'tenant',
        dataset_uuid: 'uuid',
        dataset_name: 'name',
        // Missing rule_code
      }
      
      expect(isApiSummaryRow(partialRow)).toBe(false)
    })
  })
})
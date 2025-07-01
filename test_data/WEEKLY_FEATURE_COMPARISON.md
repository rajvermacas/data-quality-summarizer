# Weekly Aggregation Feature: Grouping Comparison

## Summary of Results

| Grouping | Rows Generated | Period Length | Date Range Example | Customer ID Rule (202) Fail Rate |
|----------|----------------|---------------|-------------------|-----------------------------------|
| **1-week** | 42 rows | 7 days | 2024-01-01 to 2024-01-07 | Week 0: 100%, Week 1: 0% |
| **2-week** | 21 rows | 14 days | 2024-01-01 to 2024-01-14 | Period 0: 67% |
| **4-week** | 14 rows | 28 days | 2024-01-01 to 2024-01-28 | Period 0: 40% |

## Key Observations

### 1. **Granularity vs. Noise Trade-off**
- **1-week**: High granularity, captures rapid quality changes
- **2-week**: Balanced view, reduces weekly fluctuations  
- **4-week**: Monthly perspective, smooths out short-term variations

### 2. **Trend Detection Sensitivity**
- **1-week**: Detects immediate improvements (100% → 0% fail rate)
- **2-week**: Shows moderate overall trend (67% combined rate)
- **4-week**: Reveals longer-term patterns (40% average over month)

### 3. **Business Alignment**
- **1-week**: Ideal for operational monitoring and alerting
- **2-week**: Good for sprint-based development cycles
- **4-week**: Perfect for monthly business reviews

### 4. **Data Volume Impact**
- Same 49 input rows generate different aggregation levels
- Configurable grouping allows analysis at multiple time horizons
- Memory usage remains constant (~117MB) regardless of grouping

## Feature Advantages

✅ **Flexible Configuration**: Single `--weeks N` parameter\
✅ **Streaming Architecture**: Maintains performance for large datasets\
✅ **Accurate Trends**: Period-over-period comparison with proper fail rate calculation\
✅ **Business Date Integrity**: Uses latest actual date per period\
✅ **Multiple Output Formats**: CSV + Natural Language\
✅ **Backwards Compatible**: Default `--weeks 1` preserves existing behavior

## Use Case Recommendations

| Scenario | Recommended Grouping | Rationale |
|----------|---------------------|-----------|
| **Real-time Monitoring** | `--weeks 1` | Immediate issue detection |
| **Sprint Reviews** | `--weeks 2` | Matches 2-week development cycles |
| **Monthly Reports** | `--weeks 4` | Executive summaries and trends |
| **Quarterly Analysis** | `--weeks 12` | Long-term pattern analysis |
| **Custom Business Cycles** | `--weeks N` | Match specific business needs |

This flexible weekly aggregation feature replaces the rigid 1m/3m/12m time windows with configurable, business-aligned reporting periods.
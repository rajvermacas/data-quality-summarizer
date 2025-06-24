import React, { useState } from 'react'
import { SummaryCharts } from '../visualizations/SummaryCharts'
import { DataTable } from '../components/DataTable'
import { ProcessingResult } from '../types/common'
import { 
  BarChart3, Database, Clock, Activity, TrendingUp, 
  AlertTriangle, CheckCircle, FileText, Download, 
  RefreshCw, Brain 
} from 'lucide-react'

interface ResultsPageProps {
  result: ProcessingResult
  onStartOver: () => void
  onViewMLPipeline: () => void
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ result, onStartOver, onViewMLPipeline }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'charts' | 'table' | 'natural-language'>('overview')

  const calculateMetrics = () => {
    const totalRules = result.summary_data.length
    const highRiskRules = result.summary_data.filter(row => row.risk_level === 'HIGH').length
    const improvementNeeded = result.summary_data.filter(row => row.improvement_needed).length
    
    const avgFailRate = result.summary_data.reduce((sum, row) => sum + row.overall_fail_rate, 0) / totalRules
    const worstFailRate = Math.max(...result.summary_data.map(row => row.overall_fail_rate))
    
    const totalFailures = result.summary_data.reduce((sum, row) => sum + row.total_failures, 0)
    const totalPasses = result.summary_data.reduce((sum, row) => sum + row.total_passes, 0)

    return {
      totalRules,
      highRiskRules,
      improvementNeeded,
      avgFailRate,
      worstFailRate,
      totalFailures,
      totalPasses,
      overallHealthScore: Math.round((1 - avgFailRate) * 100)
    }
  }

  const metrics = calculateMetrics()

  const downloadSummary = (format: 'csv' | 'txt') => {
    if (format === 'csv') {
      const headers = Object.keys(result.summary_data[0]).join(',')
      const csvContent = [
        headers,
        ...result.summary_data.map(row => Object.values(row).map(val => 
          typeof val === 'string' && val.includes(',') ? `"${val}"` : val
        ).join(','))
      ].join('\n')
      
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'data_quality_summary.csv'
      link.click()
      URL.revokeObjectURL(url)
    } else {
      const content = result.nl_summary.join('\n')
      const blob = new Blob([content], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'data_quality_summary.txt'
      link.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className="results-page">
      <div className="results-header">
        <div className="header-content">
          <h2>Processing Results</h2>
          <p>Analysis complete! Here's your data quality summary and insights.</p>
        </div>
        <div className="header-actions">
          <button className="button button-secondary" onClick={onStartOver}>
            <RefreshCw size={16} />
            Process New Files
          </button>
          <button className="button" onClick={onViewMLPipeline}>
            <Brain size={16} />
            ML Pipeline
          </button>
        </div>
      </div>

      <div className="tabs">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'charts', label: 'Charts', icon: TrendingUp },
          { id: 'table', label: 'Data Table', icon: Database },
          { id: 'natural-language', label: 'Natural Language', icon: FileText }
        ].map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id as any)}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          )
        })}
      </div>

      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-content">
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-icon">
                  <Database size={24} />
                </div>
                <div className="metric-info">
                  <h3>{result.unique_datasets}</h3>
                  <p>Unique Datasets</p>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  <Activity size={24} />
                </div>
                <div className="metric-info">
                  <h3>{result.unique_rules}</h3>
                  <p>Data Quality Rules</p>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  <CheckCircle size={24} />
                </div>
                <div className="metric-info">
                  <h3>{metrics.overallHealthScore}%</h3>
                  <p>Health Score</p>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  <AlertTriangle size={24} />
                </div>
                <div className="metric-info">
                  <h3>{metrics.highRiskRules}</h3>
                  <p>High Risk Rules</p>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  <Clock size={24} />
                </div>
                <div className="metric-info">
                  <h3>{result.processing_time_seconds.toFixed(1)}s</h3>
                  <p>Processing Time</p>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  <Database size={24} />
                </div>
                <div className="metric-info">
                  <h3>{result.memory_usage_mb.toFixed(0)}MB</h3>
                  <p>Memory Used</p>
                </div>
              </div>
            </div>

            <div className="summary-cards">
              <div className="summary-card">
                <h3>Processing Summary</h3>
                <ul>
                  <li><strong>Total Rows Processed:</strong> {result.total_rows_processed.toLocaleString()}</li>
                  <li><strong>Time Range:</strong> {result.time_range.start_date} to {result.time_range.end_date}</li>
                  <li><strong>Average Fail Rate:</strong> {(metrics.avgFailRate * 100).toFixed(2)}%</li>
                  <li><strong>Worst Fail Rate:</strong> {(metrics.worstFailRate * 100).toFixed(2)}%</li>
                </ul>
              </div>

              <div className="summary-card">
                <h3>Quality Insights</h3>
                <ul>
                  <li><strong>Total Failures:</strong> {metrics.totalFailures.toLocaleString()}</li>
                  <li><strong>Total Passes:</strong> {metrics.totalPasses.toLocaleString()}</li>
                  <li><strong>Rules Needing Improvement:</strong> {metrics.improvementNeeded}</li>
                  <li><strong>Overall Success Rate:</strong> {((metrics.totalPasses / (metrics.totalPasses + metrics.totalFailures)) * 100).toFixed(2)}%</li>
                </ul>
              </div>
            </div>

            <div className="download-section">
              <h3>Download Results</h3>
              <div className="download-buttons">
                <button className="download-btn" onClick={() => downloadSummary('csv')}>
                  <Download size={16} />
                  Download CSV Summary
                </button>
                <button className="download-btn" onClick={() => downloadSummary('txt')}>
                  <Download size={16} />
                  Download Natural Language Summary
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'charts' && (
          <div className="charts-content">
            <SummaryCharts data={result.summary_data} />
          </div>
        )}

        {activeTab === 'table' && (
          <div className="table-content">
            <DataTable data={result.summary_data} title="Data Quality Summary" />
          </div>
        )}

        {activeTab === 'natural-language' && (
          <div className="nl-content">
            <div className="nl-header">
              <h3>Natural Language Summary</h3>
              <p>LLM-optimized sentences for each data quality rule assessment:</p>
            </div>
            <div className="nl-text">
              {result.nl_summary.map((sentence, index) => (
                <p key={index} className="nl-sentence">
                  {sentence}
                </p>
              ))}
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .results-page {
          max-width: 1200px;
          margin: 0 auto;
        }

        .results-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          flex-wrap: wrap;
          gap: 20px;
        }

        .header-content h2 {
          color: #333;
          margin: 0 0 5px 0;
        }

        .header-content p {
          color: #666;
          margin: 0;
        }

        .header-actions {
          display: flex;
          gap: 10px;
        }

        .button {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .tabs {
          display: flex;
          border-bottom: 1px solid #e9ecef;
          margin-bottom: 30px;
          overflow-x: auto;
        }

        .tab {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 20px;
          border: none;
          background: none;
          cursor: pointer;
          font-size: 16px;
          color: #6c757d;
          border-bottom: 2px solid transparent;
          transition: all 0.3s ease;
          white-space: nowrap;
        }

        .tab.active {
          color: #007bff;
          border-bottom-color: #007bff;
        }

        .tab:hover {
          color: #007bff;
        }

        .tab-content {
          min-height: 400px;
        }

        /* Overview Tab Styles */
        .overview-content {
          display: flex;
          flex-direction: column;
          gap: 30px;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
        }

        .metric-card {
          background: white;
          border-radius: 8px;
          padding: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
          display: flex;
          align-items: center;
          gap: 15px;
        }

        .metric-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 50px;
          height: 50px;
          background: #e3f2fd;
          border-radius: 50%;
          color: #007bff;
        }

        .metric-info h3 {
          margin: 0;
          color: #333;
          font-size: 24px;
          font-weight: bold;
        }

        .metric-info p {
          margin: 5px 0 0 0;
          color: #666;
          font-size: 14px;
        }

        .summary-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
        }

        .summary-card {
          background: white;
          border-radius: 8px;
          padding: 25px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .summary-card h3 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .summary-card ul {
          margin: 0;
          padding: 0;
          list-style: none;
        }

        .summary-card li {
          padding: 8px 0;
          border-bottom: 1px solid #f8f9fa;
        }

        .summary-card li:last-child {
          border-bottom: none;
        }

        .download-section {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 25px;
          border: 1px solid #e9ecef;
        }

        .download-section h3 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .download-buttons {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
        }

        .download-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #007bff;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: background-color 0.3s ease;
        }

        .download-btn:hover {
          background: #0056b3;
        }

        /* Natural Language Tab Styles */
        .nl-content {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .nl-header {
          margin-bottom: 25px;
        }

        .nl-header h3 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .nl-header p {
          margin: 0;
          color: #666;
        }

        .nl-text {
          max-height: 600px;
          overflow-y: auto;
          border: 1px solid #e9ecef;
          border-radius: 6px;
          padding: 20px;
          background: #f8f9fa;
        }

        .nl-sentence {
          margin: 0 0 15px 0;
          line-height: 1.6;
          color: #333;
          padding: 10px;
          background: white;
          border-radius: 4px;
          border-left: 3px solid #007bff;
        }

        .nl-sentence:last-child {
          margin-bottom: 0;
        }

        @media (max-width: 768px) {
          .results-header {
            flex-direction: column;
            align-items: stretch;
          }

          .header-actions {
            justify-content: center;
          }

          .metrics-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          }

          .download-buttons {
            justify-content: center;
          }
        }
      `}</style>
    </div>
  )
}
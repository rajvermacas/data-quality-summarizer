import React, { useState, useMemo } from 'react'
import { ChevronUp, ChevronDown, Search, Download } from 'lucide-react'
import { SummaryRow } from '../types/common'

interface DataTableProps {
  data: SummaryRow[]
  title?: string
}

type SortKey = keyof SummaryRow
type SortDirection = 'asc' | 'desc'

export const DataTable: React.FC<DataTableProps> = ({ data, title = "Data Summary" }) => {
  const [sortKey, setSortKey] = useState<SortKey>('overall_fail_rate')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 20

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDirection('desc')
    }
    setCurrentPage(1)
  }

  const filteredAndSortedData = useMemo(() => {
    let filtered = data.filter(row => 
      row.dataset_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.rule_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.rule_description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.source.toLowerCase().includes(searchTerm.toLowerCase())
    )

    return filtered.sort((a, b) => {
      const aVal = a[sortKey]
      const bVal = b[sortKey]
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
      }
      
      const aStr = String(aVal).toLowerCase()
      const bStr = String(bVal).toLowerCase()
      
      if (sortDirection === 'asc') {
        return aStr.localeCompare(bStr)
      } else {
        return bStr.localeCompare(aStr)
      }
    })
  }, [data, searchTerm, sortKey, sortDirection])

  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage
    return filteredAndSortedData.slice(startIndex, startIndex + itemsPerPage)
  }, [filteredAndSortedData, currentPage])

  const totalPages = Math.ceil(filteredAndSortedData.length / itemsPerPage)

  const exportToCsv = () => {
    const headers = [
      'Dataset Name', 'Rule Code', 'Rule Description', 'Source', 'Overall Fail Rate',
      '1M Fail Rate', '3M Fail Rate', '12M Fail Rate', 'Risk Level', 'Total Failures',
      'Total Passes', 'Latest Business Date'
    ]
    
    const csvContent = [
      headers.join(','),
      ...filteredAndSortedData.map(row => [
        `"${row.dataset_name}"`,
        row.rule_code,
        `"${row.rule_description}"`,
        row.source,
        row.overall_fail_rate.toFixed(4),
        row.fail_rate_1m.toFixed(4),
        row.fail_rate_3m.toFixed(4),
        row.fail_rate_12m.toFixed(4),
        row.risk_level,
        row.total_failures,
        row.total_passes,
        row.latest_business_date
      ].join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${title.replace(/\s+/g, '_').toLowerCase()}_export.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  const SortButton: React.FC<{ column: SortKey, children: React.ReactNode }> = ({ column, children }) => (
    <button
      className={`sort-button ${sortKey === column ? 'active' : ''}`}
      onClick={() => handleSort(column)}
    >
      {children}
      {sortKey === column && (
        sortDirection === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />
      )}
    </button>
  )

  const getRiskLevelColor = (risk: string) => {
    switch (risk?.toUpperCase()) {
      case 'HIGH': return '#dc3545'
      case 'MEDIUM': return '#fd7e14'
      case 'LOW': return '#28a745'
      default: return '#6c757d'
    }
  }

  const formatFailRate = (rate: number) => {
    return (rate * 100).toFixed(2) + '%'
  }

  return (
    <div className="data-table-container">
      <div className="table-header">
        <div className="header-left">
          <h3>{title}</h3>
          <span className="record-count">
            {filteredAndSortedData.length} of {data.length} records
          </span>
        </div>
        <div className="header-actions">
          <div className="search-box">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search datasets, rules, or sources..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value)
                setCurrentPage(1)
              }}
            />
          </div>
          <button className="export-button" onClick={exportToCsv}>
            <Download size={16} />
            Export CSV
          </button>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th><SortButton column="dataset_name">Dataset</SortButton></th>
              <th><SortButton column="rule_code">Rule</SortButton></th>
              <th><SortButton column="rule_description">Description</SortButton></th>
              <th><SortButton column="overall_fail_rate">Overall</SortButton></th>
              <th><SortButton column="fail_rate_1m">1M</SortButton></th>
              <th><SortButton column="fail_rate_3m">3M</SortButton></th>
              <th><SortButton column="fail_rate_12m">12M</SortButton></th>
              <th><SortButton column="risk_level">Risk</SortButton></th>
              <th><SortButton column="total_failures">Failures</SortButton></th>
              <th><SortButton column="latest_business_date">Latest Date</SortButton></th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, index) => (
              <tr key={`${row.dataset_uuid}-${row.rule_code}-${index}`}>
                <td title={row.dataset_name}>
                  <div className="dataset-cell">
                    <strong>{row.dataset_name.length > 25 
                      ? row.dataset_name.substring(0, 25) + '...' 
                      : row.dataset_name
                    }</strong>
                    <small>{row.source}</small>
                  </div>
                </td>
                <td className="rule-cell">
                  <code>{row.rule_code}</code>
                </td>
                <td title={row.rule_description}>
                  {row.rule_description.length > 40 
                    ? row.rule_description.substring(0, 40) + '...' 
                    : row.rule_description
                  }
                </td>
                <td className="fail-rate-cell">
                  <span className={`fail-rate ${row.overall_fail_rate > 0.1 ? 'high' : row.overall_fail_rate > 0.05 ? 'medium' : 'low'}`}>
                    {formatFailRate(row.overall_fail_rate)}
                  </span>
                </td>
                <td className="fail-rate-cell">
                  <span className={`fail-rate ${row.fail_rate_1m > 0.1 ? 'high' : row.fail_rate_1m > 0.05 ? 'medium' : 'low'}`}>
                    {formatFailRate(row.fail_rate_1m)}
                  </span>
                </td>
                <td className="fail-rate-cell">
                  <span className={`fail-rate ${row.fail_rate_3m > 0.1 ? 'high' : row.fail_rate_3m > 0.05 ? 'medium' : 'low'}`}>
                    {formatFailRate(row.fail_rate_3m)}
                  </span>
                </td>
                <td className="fail-rate-cell">
                  <span className={`fail-rate ${row.fail_rate_12m > 0.1 ? 'high' : row.fail_rate_12m > 0.05 ? 'medium' : 'low'}`}>
                    {formatFailRate(row.fail_rate_12m)}
                  </span>
                </td>
                <td>
                  <span 
                    className="risk-badge"
                    style={{ backgroundColor: getRiskLevelColor(row.risk_level) }}
                  >
                    {row.risk_level}
                  </span>
                </td>
                <td className="number-cell">{row.total_failures.toLocaleString()}</td>
                <td className="date-cell">{row.latest_business_date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button 
            className="pagination-button"
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
          >
            First
          </button>
          <button 
            className="pagination-button"
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage === 1}
          >
            Previous
          </button>
          <span className="pagination-info">
            Page {currentPage} of {totalPages}
          </span>
          <button 
            className="pagination-button"
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            Next
          </button>
          <button 
            className="pagination-button"
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
          >
            Last
          </button>
        </div>
      )}

      <style jsx>{`
        .data-table-container {
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
          overflow: hidden;
        }

        .table-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          background: #f8f9fa;
          border-bottom: 1px solid #e9ecef;
          flex-wrap: wrap;
          gap: 15px;
        }

        .header-left h3 {
          margin: 0;
          color: #333;
        }

        .record-count {
          color: #666;
          font-size: 14px;
          margin-left: 10px;
        }

        .header-actions {
          display: flex;
          align-items: center;
          gap: 15px;
        }

        .search-box {
          display: flex;
          align-items: center;
          gap: 8px;
          background: white;
          border: 1px solid #ced4da;
          border-radius: 6px;
          padding: 8px 12px;
        }

        .search-box input {
          border: none;
          outline: none;
          width: 250px;
          font-size: 14px;
        }

        .export-button {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #007bff;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: background-color 0.3s ease;
        }

        .export-button:hover {
          background: #0056b3;
        }

        .table-wrapper {
          overflow-x: auto;
        }

        .data-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }

        .data-table th {
          background: #f8f9fa;
          border-bottom: 2px solid #e9ecef;
          padding: 0;
          text-align: left;
          white-space: nowrap;
        }

        .sort-button {
          width: 100%;
          background: none;
          border: none;
          padding: 12px 8px;
          text-align: left;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: space-between;
          color: #495057;
          font-weight: 600;
          transition: background-color 0.2s ease;
        }

        .sort-button:hover {
          background: #e9ecef;
        }

        .sort-button.active {
          color: #007bff;
          background: #e3f2fd;
        }

        .data-table td {
          padding: 12px 8px;
          border-bottom: 1px solid #e9ecef;
          vertical-align: top;
        }

        .data-table tr:hover {
          background: #f8f9fa;
        }

        .dataset-cell {
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-width: 200px;
        }

        .dataset-cell small {
          color: #666;
          font-size: 12px;
        }

        .rule-cell code {
          background: #f8f9fa;
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 12px;
          border: 1px solid #e9ecef;
        }

        .fail-rate-cell {
          text-align: center;
        }

        .fail-rate {
          padding: 2px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: bold;
        }

        .fail-rate.low {
          background: #d4edda;
          color: #155724;
        }

        .fail-rate.medium {
          background: #fff3cd;
          color: #856404;
        }

        .fail-rate.high {
          background: #f8d7da;
          color: #721c24;
        }

        .risk-badge {
          padding: 2px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: bold;
          color: white;
        }

        .number-cell {
          text-align: right;
          font-family: monospace;
        }

        .date-cell {
          font-family: monospace;
          font-size: 13px;
        }

        .pagination {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 10px;
          padding: 20px;
          background: #f8f9fa;
          border-top: 1px solid #e9ecef;
        }

        .pagination-button {
          background: white;
          border: 1px solid #ced4da;
          padding: 8px 12px;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .pagination-button:hover:not(:disabled) {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        .pagination-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .pagination-info {
          margin: 0 15px;
          color: #495057;
          font-weight: 500;
        }

        @media (max-width: 768px) {
          .table-header {
            flex-direction: column;
            align-items: stretch;
          }

          .header-actions {
            flex-direction: column;
          }

          .search-box input {
            width: 100%;
          }
        }
      `}</style>
    </div>
  )
}
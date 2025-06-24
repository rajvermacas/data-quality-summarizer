import React from 'react'
import { FileUpload } from '../components/FileUpload'

interface FileUploadPageProps {
  onFileUpload: (csvFile: File, rulesFile: File) => void
}

export const FileUploadPage: React.FC<FileUploadPageProps> = ({ onFileUpload }) => {
  return (
    <div className="page">
      <FileUpload onFileUpload={onFileUpload} />
      
      <div className="info-section">
        <h3>System Requirements</h3>
        <ul>
          <li><strong>CSV File:</strong> Up to 100MB, containing data quality check results</li>
          <li><strong>Rules File:</strong> JSON format, up to 5MB, containing rule metadata</li>
          <li><strong>Processing Time:</strong> Typically 1-2 minutes for 100K rows</li>
          <li><strong>Memory Usage:</strong> System uses streaming processing to stay under 1GB RAM</li>
        </ul>

        <h3>Expected CSV Columns</h3>
        <div className="columns-grid">
          <div className="column-group">
            <h4>Identity Fields</h4>
            <ul>
              <li>source</li>
              <li>tenant_id</li>
              <li>dataset_uuid</li>
              <li>dataset_name</li>
            </ul>
          </div>
          <div className="column-group">
            <h4>Rule & Date Fields</h4>
            <ul>
              <li>rule_code</li>
              <li>business_date (ISO format)</li>
              <li>results (JSON string)</li>
            </ul>
          </div>
          <div className="column-group">
            <h4>Execution Context</h4>
            <ul>
              <li>level_of_execution</li>
              <li>attribute_name</li>
            </ul>
          </div>
        </div>

        <h3>Sample Rules JSON Structure</h3>
        <pre className="code-block">
{`{
  "R001": {
    "category": "Completeness",
    "description": "Check for missing values",
    "severity": "HIGH"
  },
  "R002": {
    "category": "Validity", 
    "description": "Validate data formats",
    "severity": "MEDIUM"
  }
}`}
        </pre>
      </div>

      <style jsx>{`
        .page {
          max-width: 1000px;
          margin: 0 auto;
        }

        .info-section {
          margin-top: 50px;
          padding-top: 30px;
          border-top: 1px solid #e9ecef;
        }

        .info-section h3 {
          color: #333;
          margin-bottom: 15px;
        }

        .info-section ul {
          margin: 0 0 20px 0;
          padding-left: 20px;
        }

        .info-section li {
          margin-bottom: 8px;
          line-height: 1.5;
        }

        .columns-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
          margin: 20px 0;
        }

        .column-group {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 6px;
        }

        .column-group h4 {
          margin: 0 0 10px 0;
          color: #495057;
          font-size: 14px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .column-group ul {
          margin: 0;
          padding-left: 15px;
        }

        .column-group li {
          font-family: monospace;
          font-size: 13px;
          margin-bottom: 5px;
        }

        .code-block {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 6px;
          padding: 15px;
          font-family: monospace;
          font-size: 13px;
          overflow-x: auto;
          margin: 15px 0;
        }
      `}</style>
    </div>
  )
}
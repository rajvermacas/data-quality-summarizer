import React, { useState, useRef } from 'react'
import { Upload, File, AlertCircle } from 'lucide-react'

interface FileUploadProps {
  onFileUpload: (csvFile: File, rulesFile: File) => void
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [rulesFile, setRulesFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState<'csv' | 'rules' | null>(null)
  const [errors, setErrors] = useState<string[]>([])

  const csvInputRef = useRef<HTMLInputElement>(null)
  const rulesInputRef = useRef<HTMLInputElement>(null)

  const validateFile = (file: File, type: 'csv' | 'rules'): string[] => {
    const errors: string[] = []
    
    if (type === 'csv') {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        errors.push('CSV file must have .csv extension')
      }
      if (file.size > 100 * 1024 * 1024) { // 100MB limit
        errors.push('CSV file must be smaller than 100MB')
      }
    } else {
      if (!file.name.toLowerCase().endsWith('.json')) {
        errors.push('Rules file must have .json extension')
      }
      if (file.size > 5 * 1024 * 1024) { // 5MB limit
        errors.push('Rules file must be smaller than 5MB')
      }
    }

    return errors
  }

  const handleFileChange = (file: File, type: 'csv' | 'rules') => {
    const validationErrors = validateFile(file, type)
    
    if (validationErrors.length > 0) {
      setErrors(validationErrors)
      return
    }

    setErrors([])
    
    if (type === 'csv') {
      setCsvFile(file)
    } else {
      setRulesFile(file)
    }
  }

  const handleDragOver = (e: React.DragEvent, type: 'csv' | 'rules') => {
    e.preventDefault()
    setDragOver(type)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(null)
  }

  const handleDrop = (e: React.DragEvent, type: 'csv' | 'rules') => {
    e.preventDefault()
    setDragOver(null)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileChange(files[0], type)
    }
  }

  const handleSubmit = () => {
    if (!csvFile || !rulesFile) {
      setErrors(['Please select both CSV file and rules file'])
      return
    }

    onFileUpload(csvFile, rulesFile)
  }

  const canSubmit = csvFile && rulesFile && errors.length === 0

  return (
    <div className="file-upload-container">
      <h2>Upload Files</h2>
      <p>Select your CSV data file and rules metadata JSON file to begin processing.</p>

      <div className="upload-grid">
        {/* CSV File Upload */}
        <div className="upload-section">
          <h3>CSV Data File</h3>
          <div
            className={`upload-area ${dragOver === 'csv' ? 'drag-over' : ''}`}
            onDragOver={(e) => handleDragOver(e, 'csv')}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(e, 'csv')}
            onClick={() => csvInputRef.current?.click()}
          >
            <Upload size={48} />
            {csvFile ? (
              <div>
                <File size={24} />
                <p><strong>{csvFile.name}</strong></p>
                <p>{(csvFile.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            ) : (
              <div>
                <p>Drop your CSV file here or click to browse</p>
                <p className="upload-hint">Max size: 100MB</p>
              </div>
            )}
          </div>
          <input
            ref={csvInputRef}
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0], 'csv')}
            style={{ display: 'none' }}
          />
        </div>

        {/* Rules File Upload */}
        <div className="upload-section">
          <h3>Rules Metadata File</h3>
          <div
            className={`upload-area ${dragOver === 'rules' ? 'drag-over' : ''}`}
            onDragOver={(e) => handleDragOver(e, 'rules')}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(e, 'rules')}
            onClick={() => rulesInputRef.current?.click()}
          >
            <Upload size={48} />
            {rulesFile ? (
              <div>
                <File size={24} />
                <p><strong>{rulesFile.name}</strong></p>
                <p>{(rulesFile.size / 1024).toFixed(2)} KB</p>
              </div>
            ) : (
              <div>
                <p>Drop your JSON rules file here or click to browse</p>
                <p className="upload-hint">Max size: 5MB</p>
              </div>
            )}
          </div>
          <input
            ref={rulesInputRef}
            type="file"
            accept=".json"
            onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0], 'rules')}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {errors.length > 0 && (
        <div className="error-message">
          <AlertCircle size={20} />
          <div>
            {errors.map((error, index) => (
              <p key={index}>{error}</p>
            ))}
          </div>
        </div>
      )}

      <div className="upload-actions">
        <button
          className="button"
          onClick={handleSubmit}
          disabled={!canSubmit}
        >
          Start Processing
        </button>
      </div>

      <style jsx>{`
        .file-upload-container {
          max-width: 800px;
          margin: 0 auto;
        }

        .upload-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 30px;
          margin: 30px 0;
        }

        .upload-section h3 {
          margin-bottom: 15px;
          color: #333;
        }

        .upload-area {
          min-height: 200px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 15px;
          color: #666;
        }

        .upload-hint {
          font-size: 14px;
          color: #888;
          margin: 0;
        }

        .upload-actions {
          text-align: center;
          margin-top: 30px;
        }

        .error-message {
          display: flex;
          align-items: flex-start;
          gap: 10px;
        }

        .error-message p {
          margin: 0;
        }

        @media (max-width: 768px) {
          .upload-grid {
            grid-template-columns: 1fr;
            gap: 20px;
          }
        }
      `}</style>
    </div>
  )
}
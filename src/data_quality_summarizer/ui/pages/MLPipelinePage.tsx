import React, { useState, useEffect } from 'react'
import { 
  Brain, ArrowLeft, Play, Download, BarChart3, Target, 
  TrendingUp, AlertCircle, CheckCircle, Loader, Settings
} from 'lucide-react'
import { SummaryRow, MLModelResult, PredictionRequest, PredictionResult } from '../types/common'

interface MLPipelinePageProps {
  onBackToResults: () => void
  csvData?: SummaryRow[]
}

interface ModelTrainingStatus {
  status: 'idle' | 'training' | 'completed' | 'error'
  progress: number
  message: string
}

export const MLPipelinePage: React.FC<MLPipelinePageProps> = ({ onBackToResults, csvData }) => {
  const [activeTab, setActiveTab] = useState<'train' | 'predict' | 'batch'>('train')
  const [trainingStatus, setTrainingStatus] = useState<ModelTrainingStatus>({
    status: 'idle',
    progress: 0,
    message: ''
  })
  const [modelResult, setModelResult] = useState<MLModelResult | null>(null)
  const [predictionRequest, setPredictionRequest] = useState<PredictionRequest>({
    dataset_uuid: '',
    rule_code: '',
    business_date: new Date().toISOString().split('T')[0]
  })
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [batchFile, setBatchFile] = useState<File | null>(null)
  const [batchResults, setBatchResults] = useState<any[] | null>(null)

  const handleTrainModel = async () => {
    if (!csvData || csvData.length === 0) {
      alert('No data available for training. Please process data first.')
      return
    }

    setTrainingStatus({
      status: 'training',
      progress: 0,
      message: 'Preparing training data...'
    })

    try {
      // Simulate training progress
      const progressSteps = [
        { progress: 20, message: 'Feature engineering...' },
        { progress: 40, message: 'Splitting data...' },
        { progress: 60, message: 'Training LightGBM model...' },
        { progress: 80, message: 'Validating model performance...' },
        { progress: 100, message: 'Training completed!' }
      ]

      for (const step of progressSteps) {
        await new Promise(resolve => setTimeout(resolve, 1000))
        setTrainingStatus(prev => ({
          ...prev,
          progress: step.progress,
          message: step.message
        }))
      }

      // Call API to train model
      const formData = new FormData()
      formData.append('csv_data', JSON.stringify(csvData))

      const response = await fetch('/api/ml/train', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`)
      }

      const result = await response.json()
      setModelResult(result)
      setTrainingStatus({
        status: 'completed',
        progress: 100,
        message: 'Model trained successfully!'
      })

    } catch (error) {
      setTrainingStatus({
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'Training failed'
      })
    }
  }

  const handlePredict = async () => {
    if (!predictionRequest.dataset_uuid || !predictionRequest.rule_code) {
      alert('Please fill in all required fields')
      return
    }

    try {
      const response = await fetch('/api/ml/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionRequest)
      })

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`)
      }

      const result = await response.json()
      setPredictionResult(result)
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Prediction failed')
    }
  }

  const handleBatchPredict = async () => {
    if (!batchFile) {
      alert('Please select a CSV file for batch prediction')
      return
    }

    try {
      const formData = new FormData()
      formData.append('batch_file', batchFile)

      const response = await fetch('/api/ml/batch-predict', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Batch prediction failed: ${response.statusText}`)
      }

      const results = await response.json()
      setBatchResults(results)
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Batch prediction failed')
    }
  }

  const downloadBatchResults = () => {
    if (!batchResults) return

    const headers = ['dataset_uuid', 'rule_code', 'business_date', 'prediction', 'probability', 'risk_score']
    const csvContent = [
      headers.join(','),
      ...batchResults.map(row => headers.map(h => row[h]).join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'batch_predictions.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="ml-pipeline-page">
      <div className="page-header">
        <div className="header-left">
          <button className="back-button" onClick={onBackToResults}>
            <ArrowLeft size={20} />
            Back to Results
          </button>
          <div className="header-title">
            <h2>
              <Brain size={24} />
              ML Pipeline
            </h2>
            <p>Train predictive models and make data quality predictions</p>
          </div>
        </div>
      </div>

      <div className="tabs">
        {[
          { id: 'train', label: 'Train Model', icon: Brain },
          { id: 'predict', label: 'Single Prediction', icon: Target },
          { id: 'batch', label: 'Batch Prediction', icon: BarChart3 }
        ].map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id as 'train' | 'predict' | 'batch')}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          )
        })}
      </div>

      <div className="tab-content">
        {activeTab === 'train' && (
          <div className="train-content">
            <div className="train-section">
              <h3>Model Training</h3>
              <p>Train a LightGBM model to predict data quality failures based on historical patterns.</p>

              {trainingStatus.status === 'idle' && (
                <div className="training-setup">
                  <div className="setup-info">
                    <h4>Training Configuration</h4>
                    <ul>
                      <li><strong>Algorithm:</strong> LightGBM (Gradient Boosting)</li>
                      <li><strong>Features:</strong> Failure rates, trends, execution patterns</li>
                      <li><strong>Target:</strong> Future failure prediction</li>
                      <li><strong>Validation:</strong> Time-series cross-validation</li>
                      <li><strong>Data Available:</strong> {csvData?.length || 0} records</li>
                    </ul>
                  </div>
                  <button 
                    className="button"
                    onClick={handleTrainModel}
                    disabled={!csvData || csvData.length === 0}
                  >
                    <Play size={16} />
                    Start Training
                  </button>
                </div>
              )}

              {trainingStatus.status === 'training' && (
                <div className="training-progress">
                  <div className="progress-header">
                    <Loader className="spinning" size={24} />
                    <span>{trainingStatus.message}</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${trainingStatus.progress}%` }}
                    />
                  </div>
                  <div className="progress-text">
                    {trainingStatus.progress}% Complete
                  </div>
                </div>
              )}

              {trainingStatus.status === 'completed' && modelResult && (
                <div className="training-results">
                  <div className="results-header">
                    <CheckCircle size={24} color="#28a745" />
                    <h4>Training Completed Successfully!</h4>
                  </div>
                  
                  <div className="results-grid">
                    <div className="result-card">
                      <h5>Model Performance</h5>
                      <div className="metrics">
                        <div className="metric">
                          <span className="metric-label">Training Score:</span>
                          <span className="metric-value">{(modelResult.training_score * 100).toFixed(2)}%</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Validation Score:</span>
                          <span className="metric-value">{(modelResult.validation_score * 100).toFixed(2)}%</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Test Score:</span>
                          <span className="metric-value">{(modelResult.test_score * 100).toFixed(2)}%</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Training Time:</span>
                          <span className="metric-value">{modelResult.training_time_seconds.toFixed(1)}s</span>
                        </div>
                      </div>
                    </div>

                    <div className="result-card">
                      <h5>Feature Importance</h5>
                      <div className="feature-list">
                        {Object.entries(modelResult.feature_importance)
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 5)
                          .map(([feature, importance]) => (
                            <div key={feature} className="feature-item">
                              <span className="feature-name">{feature}</span>
                              <div className="feature-bar">
                                <div 
                                  className="feature-fill"
                                  style={{ width: `${importance * 100}%` }}
                                />
                              </div>
                              <span className="feature-value">{(importance * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {trainingStatus.status === 'error' && (
                <div className="training-error">
                  <AlertCircle size={24} color="#dc3545" />
                  <div>
                    <h4>Training Failed</h4>
                    <p>{trainingStatus.message}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="predict-content">
            <div className="predict-section">
              <h3>Single Prediction</h3>
              <p>Make a prediction for a specific dataset and rule combination.</p>

              <div className="prediction-form">
                <div className="form-group">
                  <label>Dataset UUID</label>
                  <input
                    type="text"
                    value={predictionRequest.dataset_uuid}
                    onChange={(e) => setPredictionRequest(prev => ({
                      ...prev,
                      dataset_uuid: e.target.value
                    }))}
                    placeholder="Enter dataset UUID"
                  />
                </div>

                <div className="form-group">
                  <label>Rule Code</label>
                  <input
                    type="text"
                    value={predictionRequest.rule_code}
                    onChange={(e) => setPredictionRequest(prev => ({
                      ...prev,
                      rule_code: e.target.value
                    }))}
                    placeholder="Enter rule code (e.g., R001)"
                  />
                </div>

                <div className="form-group">
                  <label>Business Date</label>
                  <input
                    type="date"
                    value={predictionRequest.business_date}
                    onChange={(e) => setPredictionRequest(prev => ({
                      ...prev,
                      business_date: e.target.value
                    }))}
                  />
                </div>

                <button className="button" onClick={handlePredict}>
                  <Target size={16} />
                  Make Prediction
                </button>
              </div>

              {predictionResult && (
                <div className="prediction-result">
                  <h4>Prediction Result</h4>
                  <div className="result-metrics">
                    <div className="result-metric">
                      <span className="metric-label">Failure Probability:</span>
                      <span className="metric-value failure-prob">
                        {(predictionResult.probability * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="result-metric">
                      <span className="metric-label">Confidence Interval:</span>
                      <span className="metric-value">
                        [{(predictionResult.confidence_interval[0] * 100).toFixed(2)}%, {(predictionResult.confidence_interval[1] * 100).toFixed(2)}%]
                      </span>
                    </div>
                  </div>
                  
                  <div className="feature-contributions">
                    <h5>Feature Contributions</h5>
                    {Object.entries(predictionResult.feature_contributions).map(([feature, contribution]) => (
                      <div key={feature} className="contribution-item">
                        <span className="contribution-name">{feature}</span>
                        <span className={`contribution-value ${contribution > 0 ? 'positive' : 'negative'}`}>
                          {contribution > 0 ? '+' : ''}{(contribution * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'batch' && (
          <div className="batch-content">
            <div className="batch-section">
              <h3>Batch Prediction</h3>
              <p>Upload a CSV file to make predictions for multiple dataset-rule combinations.</p>

              <div className="batch-upload">
                <div className="upload-area" onClick={() => document.getElementById('batch-file')?.click()}>
                  {batchFile ? (
                    <div>
                      <CheckCircle size={24} />
                      <p><strong>{batchFile.name}</strong></p>
                      <p>{(batchFile.size / 1024).toFixed(1)} KB</p>
                    </div>
                  ) : (
                    <div>
                      <BarChart3 size={48} />
                      <p>Click to select CSV file for batch prediction</p>
                      <p className="upload-hint">Expected columns: dataset_uuid, rule_code, business_date</p>
                    </div>
                  )}
                </div>
                <input
                  id="batch-file"
                  type="file"
                  accept=".csv"
                  onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                  style={{ display: 'none' }}
                />
              </div>

              <button 
                className="button"
                onClick={handleBatchPredict}
                disabled={!batchFile}
              >
                <Play size={16} />
                Run Batch Prediction
              </button>

              {batchResults && (
                <div className="batch-results">
                  <div className="results-header">
                    <h4>Batch Prediction Results</h4>
                    <button className="download-btn" onClick={downloadBatchResults}>
                      <Download size={16} />
                      Download Results
                    </button>
                  </div>
                  
                  <div className="results-summary">
                    <p><strong>Total Predictions:</strong> {batchResults.length}</p>
                    <p><strong>High Risk Predictions:</strong> {batchResults.filter(r => r.probability > 0.7).length}</p>
                    <p><strong>Average Risk Score:</strong> {(batchResults.reduce((sum, r) => sum + r.probability, 0) / batchResults.length * 100).toFixed(2)}%</p>
                  </div>

                  <div className="results-preview">
                    <h5>Preview (First 10 Results)</h5>
                    <div className="results-table">
                      <table>
                        <thead>
                          <tr>
                            <th>Dataset UUID</th>
                            <th>Rule Code</th>
                            <th>Date</th>
                            <th>Risk Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchResults.slice(0, 10).map((result, index) => (
                            <tr key={index}>
                              <td>{result.dataset_uuid.substring(0, 8)}...</td>
                              <td>{result.rule_code}</td>
                              <td>{result.business_date}</td>
                              <td>
                                <span className={`risk-score ${result.probability > 0.7 ? 'high' : result.probability > 0.3 ? 'medium' : 'low'}`}>
                                  {(result.probability * 100).toFixed(1)}%
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .ml-pipeline-page {
          max-width: 1000px;
          margin: 0 auto;
        }

        .page-header {
          margin-bottom: 30px;
        }

        .header-left {
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .back-button {
          display: flex;
          align-items: center;
          gap: 8px;
          background: none;
          border: 1px solid #ced4da;
          padding: 8px 12px;
          border-radius: 6px;
          cursor: pointer;
          color: #495057;
          transition: all 0.2s ease;
        }

        .back-button:hover {
          background: #f8f9fa;
          border-color: #007bff;
          color: #007bff;
        }

        .header-title {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .header-title h2 {
          display: flex;
          align-items: center;
          gap: 10px;
          margin: 0;
          color: #333;
        }

        .header-title p {
          margin: 0;
          color: #666;
        }

        .tabs {
          display: flex;
          border-bottom: 1px solid #e9ecef;
          margin-bottom: 30px;
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
        }

        .tab.active {
          color: #007bff;
          border-bottom-color: #007bff;
        }

        .tab:hover {
          color: #007bff;
        }

        /* Training Tab Styles */
        .train-section {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .train-section h3 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .train-section p {
          margin: 0 0 30px 0;
          color: #666;
        }

        .training-setup {
          display: flex;
          gap: 30px;
          align-items: flex-start;
        }

        .setup-info {
          flex: 1;
        }

        .setup-info h4 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .setup-info ul {
          margin: 0;
          padding-left: 20px;
          color: #666;
        }

        .setup-info li {
          margin-bottom: 8px;
        }

        .training-progress {
          text-align: center;
          padding: 20px;
        }

        .progress-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          margin-bottom: 20px;
          color: #333;
          font-weight: 500;
        }

        .progress-bar {
          width: 100%;
          height: 20px;
          background-color: #e9ecef;
          border-radius: 10px;
          overflow: hidden;
          margin-bottom: 10px;
        }

        .progress-fill {
          height: 100%;
          background-color: #007bff;
          transition: width 0.3s ease;
        }

        .progress-text {
          color: #666;
          font-size: 14px;
        }

        .training-results {
          padding: 20px 0;
        }

        .results-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 20px;
        }

        .results-header h4 {
          margin: 0;
          color: #28a745;
        }

        .results-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
        }

        .result-card {
          background: #f8f9fa;
          border-radius: 6px;
          padding: 20px;
          border: 1px solid #e9ecef;
        }

        .result-card h5 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .metrics {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .metric {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .metric-label {
          color: #666;
        }

        .metric-value {
          font-weight: bold;
          color: #333;
        }

        .feature-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .feature-item {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .feature-name {
          min-width: 120px;
          font-size: 13px;
          color: #666;
        }

        .feature-bar {
          flex: 1;
          height: 8px;
          background: #e9ecef;
          border-radius: 4px;
          overflow: hidden;
        }

        .feature-fill {
          height: 100%;
          background: #007bff;
        }

        .feature-value {
          min-width: 40px;
          text-align: right;
          font-size: 12px;
          color: #666;
        }

        .training-error {
          display: flex;
          align-items: center;
          gap: 15px;
          padding: 20px;
          background: #f8d7da;
          border: 1px solid #f5c6cb;
          border-radius: 6px;
          color: #721c24;
        }

        .training-error h4 {
          margin: 0 0 5px 0;
        }

        .training-error p {
          margin: 0;
        }

        /* Prediction Tab Styles */
        .predict-section {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .predict-section h3 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .predict-section p {
          margin: 0 0 30px 0;
          color: #666;
        }

        .prediction-form {
          display: flex;
          flex-direction: column;
          gap: 20px;
          max-width: 400px;
          margin-bottom: 30px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .form-group label {
          font-weight: 500;
          color: #333;
        }

        .form-group input {
          padding: 10px;
          border: 1px solid #ced4da;
          border-radius: 6px;
          font-size: 14px;
        }

        .form-group input:focus {
          outline: none;
          border-color: #007bff;
          box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
        }

        .prediction-result {
          background: #f8f9fa;
          border-radius: 6px;
          padding: 20px;
          border: 1px solid #e9ecef;
        }

        .prediction-result h4 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .result-metrics {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-bottom: 20px;
        }

        .result-metric {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .failure-prob {
          font-size: 18px;
          font-weight: bold;
          color: #dc3545;
        }

        .feature-contributions h5 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .contribution-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 5px 0;
          border-bottom: 1px solid #e9ecef;
        }

        .contribution-item:last-child {
          border-bottom: none;
        }

        .contribution-name {
          color: #666;
          font-size: 14px;
        }

        .contribution-value {
          font-weight: bold;
          font-size: 14px;
        }

        .contribution-value.positive {
          color: #dc3545;
        }

        .contribution-value.negative {
          color: #28a745;
        }

        /* Batch Tab Styles */
        .batch-section {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .batch-section h3 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .batch-section p {
          margin: 0 0 30px 0;
          color: #666;
        }

        .batch-upload {
          margin-bottom: 20px;
        }

        .upload-area {
          border: 2px dashed #ccc;
          border-radius: 8px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: border-color 0.3s ease;
          margin-bottom: 20px;
        }

        .upload-area:hover {
          border-color: #007bff;
        }

        .upload-hint {
          font-size: 14px;
          color: #888;
          margin: 5px 0 0 0;
        }

        .batch-results {
          margin-top: 30px;
          padding-top: 20px;
          border-top: 1px solid #e9ecef;
        }

        .batch-results .results-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .batch-results .results-header h4 {
          margin: 0;
          color: #333;
        }

        .download-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #28a745;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: background-color 0.3s ease;
        }

        .download-btn:hover {
          background: #218838;
        }

        .results-summary {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 20px;
        }

        .results-summary p {
          margin: 5px 0;
          color: #333;
        }

        .results-preview h5 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .results-table {
          overflow-x: auto;
        }

        .results-table table {
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }

        .results-table th,
        .results-table td {
          padding: 10px;
          text-align: left;
          border-bottom: 1px solid #e9ecef;
        }

        .results-table th {
          background: #f8f9fa;
          font-weight: 600;
          color: #495057;
        }

        .risk-score {
          padding: 2px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: bold;
        }

        .risk-score.low {
          background: #d4edda;
          color: #155724;
        }

        .risk-score.medium {
          background: #fff3cd;
          color: #856404;
        }

        .risk-score.high {
          background: #f8d7da;
          color: #721c24;
        }

        .button {
          display: flex;
          align-items: center;
          gap: 8px;
          background: #007bff;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
          transition: background-color 0.3s ease;
        }

        .button:hover {
          background: #0056b3;
        }

        .button:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        :global(.spinning) {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
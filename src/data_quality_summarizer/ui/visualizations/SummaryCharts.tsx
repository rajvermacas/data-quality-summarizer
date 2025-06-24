import React from 'react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter
} from 'recharts'
import { SummaryRow } from '../types/common'

interface SummaryChartsProps {
  data: SummaryRow[]
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D']

export const SummaryCharts: React.FC<SummaryChartsProps> = ({ data }) => {
  // Prepare data for different chart types
  const ruleCategoryData = data.reduce((acc, row) => {
    const category = row.rule_category || 'Unknown'
    if (!acc[category]) {
      acc[category] = { category, count: 0, totalFailures: 0, totalPasses: 0 }
    }
    acc[category].count++
    acc[category].totalFailures += row.total_failures
    acc[category].totalPasses += row.total_passes
    return acc
  }, {} as Record<string, any>)

  const categoryChartData = Object.values(ruleCategoryData).map((item: any) => ({
    ...item,
    failRate: item.totalFailures / (item.totalFailures + item.totalPasses) * 100
  }))

  const riskLevelData = data.reduce((acc, row) => {
    const risk = row.risk_level || 'Unknown'
    acc[risk] = (acc[risk] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const riskChartData = Object.entries(riskLevelData).map(([risk, count]) => ({
    risk,
    count,
    fill: risk === 'HIGH' ? '#FF8042' : risk === 'MEDIUM' ? '#FFBB28' : '#00C49F'
  }))

  const trendData = data.slice(0, 20).map((row, index) => ({
    dataset: row.dataset_name.substring(0, 15) + '...',
    failRate1m: row.fail_rate_1m,
    failRate3m: row.fail_rate_3m,
    failRate12m: row.fail_rate_12m,
    index
  }))

  const executionConsistencyData = data.slice(0, 15).map((row, index) => ({
    dataset: row.dataset_name.substring(0, 12) + '...',
    consistency: row.execution_consistency,
    avgDailyExec: row.avg_daily_executions,
    index
  }))

  return (
    <div className="charts-container">
      <div className="chart-grid">
        {/* Rule Category Failure Rates */}
        <div className="chart-card">
          <h3>Failure Rates by Rule Category</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="category" 
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis />
              <Tooltip 
                formatter={(value: number) => [`${value.toFixed(2)}%`, 'Failure Rate']}
              />
              <Legend />
              <Bar dataKey="failRate" fill="#FF8042" name="Failure Rate %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Level Distribution */}
        <div className="chart-card">
          <h3>Risk Level Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskChartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ risk, count, percent }) => `${risk}: ${count} (${(percent * 100).toFixed(0)}%)`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {riskChartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Failure Rate Trends */}
        <div className="chart-card full-width">
          <h3>Failure Rate Trends (Top 20 Datasets)</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="dataset"
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="failRate1m" 
                stroke="#FF8042" 
                name="1 Month"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="failRate3m" 
                stroke="#00C49F" 
                name="3 Months"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="failRate12m" 
                stroke="#0088FE" 
                name="12 Months"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Execution Consistency vs Daily Executions */}
        <div className="chart-card full-width">
          <h3>Execution Consistency vs Daily Execution Frequency</h3>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart data={executionConsistencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="avgDailyExec" 
                name="Avg Daily Executions"
                type="number"
              />
              <YAxis 
                dataKey="consistency" 
                name="Execution Consistency"
                type="number"
                domain={[0, 1]}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value: number, name: string) => [
                  name === 'consistency' ? value.toFixed(3) : value.toFixed(1),
                  name === 'consistency' ? 'Consistency' : 'Daily Executions'
                ]}
                labelFormatter={(label) => `Dataset: ${executionConsistencyData[label]?.dataset}`}
              />
              <Scatter dataKey="consistency" fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <style jsx>{`
        .charts-container {
          width: 100%;
        }

        .chart-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
          gap: 30px;
          margin-top: 20px;
        }

        .chart-card {
          background: white;
          border-radius: 8px;
          padding: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .chart-card.full-width {
          grid-column: 1 / -1;
        }

        .chart-card h3 {
          color: #333;
          margin: 0 0 20px 0;
          font-size: 18px;
          font-weight: 600;
        }

        @media (max-width: 768px) {
          .chart-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  )
}
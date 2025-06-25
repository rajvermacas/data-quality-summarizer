import '@testing-library/jest-dom';

// Mock Recharts since it doesn't work well in test environments
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => children,
  BarChart: () => 'BarChart',
  PieChart: () => 'PieChart', 
  LineChart: () => 'LineChart',
  ScatterChart: () => 'ScatterChart',
  Bar: () => null,
  Pie: () => null,
  Line: () => null,
  Scatter: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  Cell: () => null,
}));

// Mock window.URL.createObjectURL for file downloads
Object.defineProperty(window.URL, 'createObjectURL', {
  writable: true,
  value: jest.fn(() => 'mocked-url'),
});

Object.defineProperty(window.URL, 'revokeObjectURL', {
  writable: true,
  value: jest.fn(),
});
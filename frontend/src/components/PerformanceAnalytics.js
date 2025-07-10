import React, { useMemo } from 'react';
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PerformanceAnalytics = ({ metrics }) => {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
          font: { size: 11 }
        }
      },
    },
    scales: {
      x: {
        ticks: { color: '#9ca3af', font: { size: 10 } },
        grid: { color: '#374151' }
      },
      y: {
        ticks: { color: '#9ca3af', font: { size: 10 } },
        grid: { color: '#374151' }
      }
    }
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
          font: { size: 11 }
        }
      },
    },
    scales: {
      r: {
        angleLines: { color: '#374151' },
        grid: { color: '#374151' },
        ticks: { color: '#9ca3af', font: { size: 10 } },
        pointLabels: { color: '#ffffff', font: { size: 11 } }
      }
    }
  };

  // Generate performance data
  const performanceData = useMemo(() => {
    const days = 30;
    const data = [];
    let cumulativeReturn = 0;
    
    for (let i = 0; i < days; i++) {
      const dailyReturn = (Math.random() - 0.3) * 5; // Slight positive bias
      cumulativeReturn += dailyReturn;
      data.push({
        day: i + 1,
        return: dailyReturn,
        cumulative: cumulativeReturn,
        drawdown: Math.min(0, cumulativeReturn - Math.max(...data.map(d => d?.cumulative || 0))),
        volume: Math.floor(Math.random() * 1000000) + 500000
      });
    }
    return data;
  }, []);

  const equityCurveData = {
    labels: performanceData.map(d => `Day ${d.day}`),
    datasets: [
      {
        label: 'Cumulative Return (%)',
        data: performanceData.map(d => d.cumulative),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Drawdown (%)',
        data: performanceData.map(d => d.drawdown),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const dailyReturnsData = {
    labels: performanceData.map(d => `Day ${d.day}`),
    datasets: [
      {
        label: 'Daily Returns (%)',
        data: performanceData.map(d => d.return),
        backgroundColor: performanceData.map(d => 
          d.return >= 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
        ),
        borderColor: performanceData.map(d => 
          d.return >= 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'
        ),
        borderWidth: 1,
      },
    ],
  };

  const strategyComparisonData = {
    labels: ['Win Rate', 'Avg Return', 'Risk Score', 'Consistency', 'Efficiency', 'Stability'],
    datasets: [
      {
        label: 'NostalgiaForInfinity',
        data: [89, 85, 70, 92, 88, 86],
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderColor: 'rgb(34, 197, 94)',
        borderWidth: 2,
        pointBackgroundColor: 'rgb(34, 197, 94)',
      },
      {
        label: 'LSTM Neural',
        data: [92, 88, 65, 89, 91, 87],
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 2,
        pointBackgroundColor: 'rgb(59, 130, 246)',
      },
      {
        label: 'Ichimoku Cloud',
        data: [85, 82, 75, 88, 85, 89],
        backgroundColor: 'rgba(168, 85, 247, 0.2)',
        borderColor: 'rgb(168, 85, 247)',
        borderWidth: 2,
        pointBackgroundColor: 'rgb(168, 85, 247)',
      },
    ],
  };

  const riskMetricsData = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [
      {
        data: [65, 25, 10],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(249, 115, 22, 0.8)',
          'rgba(239, 68, 68, 0.8)',
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(249, 115, 22)',
          'rgb(239, 68, 68)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const performanceMetrics = {
    totalReturn: performanceData[performanceData.length - 1]?.cumulative || 0,
    winRate: 87.3,
    sharpeRatio: 2.41,
    maxDrawdown: Math.min(...performanceData.map(d => d.drawdown)),
    volatility: 12.5,
    avgWin: 2.3,
    avgLoss: -1.8,
    profitFactor: 2.1,
    totalTrades: 245,
    winningTrades: 214,
    losingTrades: 31,
  };

  return (
    <div className="space-y-6">
      {/* Key Performance Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="bg-gradient-to-br from-green-600 to-green-700 rounded-lg p-4 text-white">
          <div className="text-sm opacity-90">Total Return</div>
          <div className="text-2xl font-bold">{performanceMetrics.totalReturn.toFixed(2)}%</div>
        </div>
        <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg p-4 text-white">
          <div className="text-sm opacity-90">Win Rate</div>
          <div className="text-2xl font-bold">{performanceMetrics.winRate}%</div>
        </div>
        <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg p-4 text-white">
          <div className="text-sm opacity-90">Sharpe Ratio</div>
          <div className="text-2xl font-bold">{performanceMetrics.sharpeRatio}</div>
        </div>
        <div className="bg-gradient-to-br from-red-600 to-red-700 rounded-lg p-4 text-white">
          <div className="text-sm opacity-90">Max Drawdown</div>
          <div className="text-2xl font-bold">{performanceMetrics.maxDrawdown.toFixed(2)}%</div>
        </div>
        <div className="bg-gradient-to-br from-yellow-600 to-yellow-700 rounded-lg p-4 text-white">
          <div className="text-sm opacity-90">Volatility</div>
          <div className="text-2xl font-bold">{performanceMetrics.volatility}%</div>
        </div>
        <div className="bg-gradient-to-br from-indigo-600 to-indigo-700 rounded-lg p-4 text-white">
          <div className="text-sm opacity-90">Profit Factor</div>
          <div className="text-2xl font-bold">{performanceMetrics.profitFactor}</div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-green-400 mb-4">Equity Curve & Drawdown</h3>
          <div className="h-64">
            <Line data={equityCurveData} options={chartOptions} />
          </div>
        </div>

        {/* Daily Returns */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-blue-400 mb-4">Daily Returns Distribution</h3>
          <div className="h-64">
            <Bar data={dailyReturnsData} options={chartOptions} />
          </div>
        </div>

        {/* Strategy Comparison Radar */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-purple-400 mb-4">Strategy Comparison</h3>
          <div className="h-64">
            <Radar data={strategyComparisonData} options={radarOptions} />
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-yellow-400 mb-4">Risk Distribution</h3>
          <div className="h-64">
            <Doughnut data={riskMetricsData} options={chartOptions} />
          </div>
        </div>
      </div>

      {/* Detailed Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-green-400 mb-4">Trading Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Trades</span>
              <span className="text-white font-medium">{performanceMetrics.totalTrades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Winning Trades</span>
              <span className="text-green-400 font-medium">{performanceMetrics.winningTrades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Losing Trades</span>
              <span className="text-red-400 font-medium">{performanceMetrics.losingTrades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Win Rate</span>
              <span className="text-green-400 font-medium">{performanceMetrics.winRate}%</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-blue-400 mb-4">Risk Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Sharpe Ratio</span>
              <span className="text-white font-medium">{performanceMetrics.sharpeRatio}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Drawdown</span>
              <span className="text-red-400 font-medium">{performanceMetrics.maxDrawdown.toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility</span>
              <span className="text-yellow-400 font-medium">{performanceMetrics.volatility}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Profit Factor</span>
              <span className="text-green-400 font-medium">{performanceMetrics.profitFactor}</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-purple-400 mb-4">Return Analysis</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Return</span>
              <span className="text-green-400 font-medium">{performanceMetrics.totalReturn.toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Average Win</span>
              <span className="text-green-400 font-medium">{performanceMetrics.avgWin}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Average Loss</span>
              <span className="text-red-400 font-medium">{performanceMetrics.avgLoss}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Return/Risk</span>
              <span className="text-yellow-400 font-medium">{(performanceMetrics.totalReturn / performanceMetrics.volatility).toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Timeline */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-bold text-indigo-400 mb-4">Performance Timeline</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-600">
                <th className="text-left py-2 px-4 text-gray-400">Period</th>
                <th className="text-left py-2 px-4 text-gray-400">Return</th>
                <th className="text-left py-2 px-4 text-gray-400">Trades</th>
                <th className="text-left py-2 px-4 text-gray-400">Win Rate</th>
                <th className="text-left py-2 px-4 text-gray-400">Drawdown</th>
                <th className="text-left py-2 px-4 text-gray-400">Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {[
                { period: 'Last 7 Days', return: 3.2, trades: 45, winRate: 89.5, drawdown: -1.2, sharpe: 2.8 },
                { period: 'Last 30 Days', return: 12.7, trades: 189, winRate: 87.3, drawdown: -3.4, sharpe: 2.4 },
                { period: 'Last 90 Days', return: 28.4, trades: 567, winRate: 86.8, drawdown: -5.7, sharpe: 2.2 },
                { period: 'Last 6 Months', return: 45.2, trades: 1234, winRate: 87.1, drawdown: -8.9, sharpe: 2.3 },
                { period: 'Last Year', return: 89.6, trades: 2456, winRate: 86.9, drawdown: -12.3, sharpe: 2.1 },
              ].map((row, index) => (
                <tr key={index} className="border-b border-gray-700 hover:bg-gray-700">
                  <td className="py-2 px-4 font-medium text-white">{row.period}</td>
                  <td className="py-2 px-4 text-green-400">{row.return}%</td>
                  <td className="py-2 px-4 text-blue-400">{row.trades}</td>
                  <td className="py-2 px-4 text-green-400">{row.winRate}%</td>
                  <td className="py-2 px-4 text-red-400">{row.drawdown}%</td>
                  <td className="py-2 px-4 text-purple-400">{row.sharpe}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PerformanceAnalytics;
import React, { useState, useEffect } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const TradingDashboard = ({ data, signals, performanceMetrics }) => {
  const [selectedMetric, setSelectedMetric] = useState('price');
  const [timeRange, setTimeRange] = useState('24h');

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: 'Market Performance',
        color: '#ffffff',
        font: {
          size: 16
        }
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#9ca3af',
          font: {
            size: 10
          }
        },
        grid: {
          color: '#374151'
        }
      },
      y: {
        ticks: {
          color: '#9ca3af',
          font: {
            size: 10
          }
        },
        grid: {
          color: '#374151'
        }
      }
    }
  };

  const priceData = {
    labels: data.map(item => item.symbol),
    datasets: [
      {
        label: 'Current Price',
        data: data.map(item => item.current_price),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const signalStrengthData = {
    labels: signals.map(signal => signal.symbol),
    datasets: [
      {
        label: 'Signal Strength',
        data: signals.map(signal => signal.strength * 100),
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(249, 115, 22, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(168, 85, 247, 0.8)',
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(59, 130, 246)',
          'rgb(249, 115, 22)',
          'rgb(239, 68, 68)',
          'rgb(168, 85, 247)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const performanceData = {
    labels: ['Wins', 'Losses', 'Pending'],
    datasets: [
      {
        data: [
          performanceMetrics.wins || 87,
          performanceMetrics.losses || 13,
          performanceMetrics.pending || 5
        ],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(249, 115, 22, 0.8)',
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(239, 68, 68)',
          'rgb(249, 115, 22)',
        ],
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
      {/* Price Chart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-green-400">Price Movement</h3>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500"
          >
            <option value="1h">1H</option>
            <option value="4h">4H</option>
            <option value="1d">1D</option>
            <option value="1w">1W</option>
          </select>
        </div>
        <div className="h-64">
          <Line data={priceData} options={chartOptions} />
        </div>
      </div>

      {/* Signal Strength Chart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-bold text-blue-400 mb-4">Signal Strength</h3>
        <div className="h-64">
          <Bar data={signalStrengthData} options={chartOptions} />
        </div>
      </div>

      {/* Performance Pie Chart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-bold text-yellow-400 mb-4">Performance Overview</h3>
        <div className="h-64">
          <Doughnut data={performanceData} options={chartOptions} />
        </div>
      </div>

      {/* Strategy Performance Table */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 xl:col-span-2">
        <h3 className="text-lg font-bold text-purple-400 mb-4">Strategy Performance</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-600">
                <th className="text-left py-2 px-4 text-gray-400">Strategy</th>
                <th className="text-left py-2 px-4 text-gray-400">Win Rate</th>
                <th className="text-left py-2 px-4 text-gray-400">Avg Return</th>
                <th className="text-left py-2 px-4 text-gray-400">Risk Score</th>
                <th className="text-left py-2 px-4 text-gray-400">Status</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: 'NostalgiaForInfinity', winRate: 89.2, avgReturn: 2.4, riskScore: 3.1, status: 'Active' },
                { name: 'Ichimoku Cloud', winRate: 85.7, avgReturn: 2.1, riskScore: 2.8, status: 'Active' },
                { name: 'SuperTrend Multi', winRate: 87.3, avgReturn: 2.3, riskScore: 3.2, status: 'Active' },
                { name: 'LSTM Neural', winRate: 92.1, avgReturn: 2.8, riskScore: 3.7, status: 'Active' },
                { name: 'Binary Options', winRate: 78.4, avgReturn: 1.9, riskScore: 4.2, status: 'Active' },
              ].map((strategy, index) => (
                <tr key={index} className="border-b border-gray-700 hover:bg-gray-700">
                  <td className="py-2 px-4 font-medium text-white">{strategy.name}</td>
                  <td className="py-2 px-4 text-green-400">{strategy.winRate}%</td>
                  <td className="py-2 px-4 text-yellow-400">{strategy.avgReturn}%</td>
                  <td className="py-2 px-4">
                    <span className={`px-2 py-1 rounded text-xs ${
                      strategy.riskScore < 3 ? 'bg-green-600' : 
                      strategy.riskScore < 4 ? 'bg-yellow-600' : 'bg-red-600'
                    }`}>
                      {strategy.riskScore}
                    </span>
                  </td>
                  <td className="py-2 px-4">
                    <span className="px-2 py-1 bg-green-600 rounded text-xs">
                      {strategy.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Market Sentiment */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-bold text-red-400 mb-4">Market Sentiment</h3>
        <div className="space-y-4">
          {[
            { pair: 'EURUSD', sentiment: 'Bullish', confidence: 85 },
            { pair: 'GBPUSD', sentiment: 'Bearish', confidence: 78 },
            { pair: 'USDJPY', sentiment: 'Neutral', confidence: 65 },
            { pair: 'AUDUSD', sentiment: 'Bullish', confidence: 82 },
          ].map((item, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="font-medium text-blue-400">{item.pair}</span>
                <span className={`px-2 py-1 rounded text-xs ${
                  item.sentiment === 'Bullish' ? 'bg-green-600' :
                  item.sentiment === 'Bearish' ? 'bg-red-600' : 'bg-gray-600'
                }`}>
                  {item.sentiment}
                </span>
              </div>
              <div className="text-sm text-gray-400">
                {item.confidence}% confidence
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;
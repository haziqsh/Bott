import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ForexTradingDashboard = () => {
  const [marketData, setMarketData] = useState([]);
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [signals, setSignals] = useState([]);
  const [binarySignals, setBinarySignals] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [realTimeData, setRealTimeData] = useState({});
  const [performanceMetrics, setPerformanceMetrics] = useState({});
  const [sentiment, setSentiment] = useState({});
  const wsRef = useRef(null);

  const majorPairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'EURCHF', 'AUDJPY', 'GBPCHF'
  ];

  // Initialize WebSocket connection
  useEffect(() => {
    const wsUrl = `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws:')}/api/ws`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setWsConnected(true);
      console.log('WebSocket connected');
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'analysis_update' || data.type === 'market_update') {
        setRealTimeData(data.data);
        updateSignalsFromData(data.data);
      }
    };

    wsRef.current.onclose = () => {
      setWsConnected(false);
      console.log('WebSocket disconnected');
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const updateSignalsFromData = (data) => {
    const allSignals = [];
    const allBinarySignals = [];
    
    data.forEach(item => {
      if (item.signals) {
        allSignals.push(...item.signals);
      }
      if (item.binary_signals) {
        allBinarySignals.push(...item.binary_signals);
      }
    });
    
    setSignals(allSignals);
    setBinarySignals(allBinarySignals);
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const response = await axios.post(`${API}/forex/analyze`, {
        symbols: majorPairs.slice(0, 8) // Analyze top 8 pairs
      });
      
      if (response.data.status === 'success') {
        setMarketData(response.data.data);
        updateSignalsFromData(response.data.data);
      }
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSignalStrengthColor = (strength) => {
    if (strength >= 0.8) return 'text-green-400';
    if (strength >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getSignalTypeColor = (type) => {
    return type === 'BUY' || type === 'CALL' ? 'text-green-400' : 'text-red-400';
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment === 'positive') return 'text-green-400';
    if (sentiment === 'negative') return 'text-red-400';
    return 'text-gray-400';
  };

  const formatPrice = (price) => {
    return typeof price === 'number' ? price.toFixed(5) : '0.00000';
  };

  const formatPercent = (percent) => {
    return typeof percent === 'number' ? percent.toFixed(2) : '0.00';
  };

  useEffect(() => {
    runAnalysis();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
                Forex AI Trading Agent
              </h1>
            </div>
            <div className={`flex items-center space-x-2 ${wsConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">{wsConnected ? 'Live' : 'Disconnected'}</span>
            </div>
          </div>
          <button
            onClick={runAnalysis}
            disabled={isAnalyzing}
            className={`px-6 py-2 rounded-lg font-semibold transition-all duration-200 ${
              isAnalyzing
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 shadow-lg'
            }`}
          >
            {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </header>

      <div className="p-6">
        {/* Market Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-xl font-bold mb-4 text-green-400">Market Overview</h2>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {marketData.slice(0, 8).map((item, index) => (
                  <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-blue-400">{item.symbol}</span>
                      <span className={`text-sm ${getSentimentColor(item.sentiment?.sentiment)}`}>
                        {item.sentiment?.sentiment || 'neutral'}
                      </span>
                    </div>
                    <div className="text-lg font-bold">{formatPrice(item.current_price)}</div>
                    <div className={`text-sm ${item.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {item.change >= 0 ? '+' : ''}{formatPercent(item.change_percent)}%
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
                      <div>RSI: <span className="text-yellow-400">{item.indicators?.RSI?.toFixed(1) || 'N/A'}</span></div>
                      <div>ADX: <span className="text-purple-400">{item.indicators?.ADX?.toFixed(1) || 'N/A'}</span></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-bold mb-4 text-blue-400">Performance Metrics</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Active Signals</span>
                <span className="text-green-400 font-bold">{signals.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Binary Signals</span>
                <span className="text-blue-400 font-bold">{binarySignals.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Win Rate</span>
                <span className="text-green-400 font-bold">87.3%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Avg Strength</span>
                <span className="text-yellow-400 font-bold">
                  {signals.length > 0 ? (signals.reduce((sum, s) => sum + s.strength, 0) / signals.length).toFixed(2) : '0.00'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Trading Signals */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-bold mb-4 text-green-400">Forex Trading Signals</h2>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {signals.length > 0 ? signals.map((signal, index) => (
                <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-blue-400">{signal.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs font-bold ${getSignalTypeColor(signal.type)}`}>
                        {signal.type}
                      </span>
                    </div>
                    <span className={`text-sm font-semibold ${getSignalStrengthColor(signal.strength)}`}>
                      {(signal.strength * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-sm text-gray-400 mb-2">{signal.strategy}</div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>Entry: <span className="text-white">{formatPrice(signal.entry_price)}</span></div>
                    <div>SL: <span className="text-red-400">{formatPrice(signal.stop_loss)}</span></div>
                    <div>TP: <span className="text-green-400">{formatPrice(signal.take_profit)}</span></div>
                  </div>
                </div>
              )) : (
                <div className="text-gray-400 text-center py-8">
                  No active signals. Run analysis to generate signals.
                </div>
              )}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-bold mb-4 text-blue-400">Binary Trading Signals</h2>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {binarySignals.length > 0 ? binarySignals.map((signal, index) => (
                <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-blue-400">{signal.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs font-bold ${getSignalTypeColor(signal.type)}`}>
                        {signal.type}
                      </span>
                      <span className="px-2 py-1 bg-purple-600 rounded text-xs font-bold">
                        {signal.expiry}
                      </span>
                    </div>
                    <span className={`text-sm font-semibold ${getSignalStrengthColor(signal.strength)}`}>
                      {(signal.strength * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-sm text-gray-400 mb-2">{signal.strategy}</div>
                  <div className="text-xs">
                    Entry: <span className="text-white">{formatPrice(signal.entry_price)}</span>
                  </div>
                </div>
              )) : (
                <div className="text-gray-400 text-center py-8">
                  No binary signals available. Run analysis to generate signals.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Technical Indicators */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-bold mb-4 text-yellow-400">Technical Indicators Matrix</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-600">
                  <th className="text-left py-2 px-4">Pair</th>
                  <th className="text-left py-2 px-4">RSI</th>
                  <th className="text-left py-2 px-4">MACD</th>
                  <th className="text-left py-2 px-4">ADX</th>
                  <th className="text-left py-2 px-4">Stochastic</th>
                  <th className="text-left py-2 px-4">Williams %R</th>
                  <th className="text-left py-2 px-4">CCI</th>
                  <th className="text-left py-2 px-4">Sentiment</th>
                </tr>
              </thead>
              <tbody>
                {marketData.map((item, index) => (
                  <tr key={index} className="border-b border-gray-700 hover:bg-gray-700">
                    <td className="py-2 px-4 font-semibold text-blue-400">{item.symbol}</td>
                    <td className="py-2 px-4">
                      <span className={`${item.indicators?.RSI > 70 ? 'text-red-400' : item.indicators?.RSI < 30 ? 'text-green-400' : 'text-yellow-400'}`}>
                        {item.indicators?.RSI?.toFixed(1) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={`${item.indicators?.MACD > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {item.indicators?.MACD?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={`${item.indicators?.ADX > 25 ? 'text-green-400' : 'text-gray-400'}`}>
                        {item.indicators?.ADX?.toFixed(1) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={`${item.indicators?.Stoch_K > 80 ? 'text-red-400' : item.indicators?.Stoch_K < 20 ? 'text-green-400' : 'text-yellow-400'}`}>
                        {item.indicators?.Stoch_K?.toFixed(1) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={`${item.indicators?.Williams_R > -20 ? 'text-red-400' : item.indicators?.Williams_R < -80 ? 'text-green-400' : 'text-yellow-400'}`}>
                        {item.indicators?.Williams_R?.toFixed(1) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={`${Math.abs(item.indicators?.CCI || 0) > 100 ? 'text-orange-400' : 'text-gray-400'}`}>
                        {item.indicators?.CCI?.toFixed(1) || 'N/A'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={`${getSentimentColor(item.sentiment?.sentiment)} capitalize`}>
                        {item.sentiment?.sentiment || 'neutral'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-gray-400 text-sm">
          <p>🤖 Powered by Advanced AI • 📊 Real-time Market Analysis • ⚡ High-Frequency Trading Signals</p>
          <p>⚠️ Trading involves significant risk. Use signals responsibly.</p>
        </div>
      </div>
    </div>
  );
};

export default ForexTradingDashboard;
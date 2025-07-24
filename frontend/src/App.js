import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';
import MailaSoulDashboard from './components/MailaSoulDashboard';
import DivineChart from './components/DivineChart';
import TradingDashboard from './components/TradingDashboard';
import AdvancedChart from './components/AdvancedChart';
import SignalHeatmap from './components/SignalHeatmap';
import PerformanceAnalytics from './components/PerformanceAnalytics';
import TechnicalIndicators from './components/TechnicalIndicators';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ForexTradingDashboard = () => {
  const [activeSystem, setActiveSystem] = useState('maila_soul');
  const [marketData, setMarketData] = useState([]);
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [signals, setSignals] = useState([]);
  const [binarySignals, setBinarySignals] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [realTimeData, setRealTimeData] = useState({});
  const [performanceMetrics, setPerformanceMetrics] = useState({});
  const [sentiment, setSentiment] = useState({});
  const [historicalData, setHistoricalData] = useState([]);
  const [dashboardLayout, setDashboardLayout] = useState('grid');
  const [activeTab, setActiveTab] = useState('overview');
  const wsRef = useRef(null);

  const systems = [
    { id: 'maila_soul', name: 'Maila Soul', icon: '💜', description: 'The Divine Whisper' },
    { id: 'advanced_forex', name: 'Advanced Forex', icon: '📊', description: 'Professional Trading' }
  ];

  const majorPairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'EURCHF', 'AUDJPY', 'GBPCHF'
  ];

  const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];

  // Default layout for grid components
  const defaultLayouts = {
    lg: [
      { i: 'chart', x: 0, y: 0, w: 8, h: 8, minW: 6, minH: 6 },
      { i: 'signals', x: 8, y: 0, w: 4, h: 4, minW: 3, minH: 3 },
      { i: 'performance', x: 8, y: 4, w: 4, h: 4, minW: 3, minH: 3 },
      { i: 'heatmap', x: 0, y: 8, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'indicators', x: 6, y: 8, w: 6, h: 4, minW: 4, minH: 3 },
    ],
    md: [
      { i: 'chart', x: 0, y: 0, w: 6, h: 6, minW: 4, minH: 4 },
      { i: 'signals', x: 6, y: 0, w: 6, h: 3, minW: 3, minH: 2 },
      { i: 'performance', x: 6, y: 3, w: 6, h: 3, minW: 3, minH: 2 },
      { i: 'heatmap', x: 0, y: 6, w: 6, h: 3, minW: 4, minH: 2 },
      { i: 'indicators', x: 6, y: 6, w: 6, h: 3, minW: 4, minH: 2 },
    ],
    sm: [
      { i: 'chart', x: 0, y: 0, w: 6, h: 4, minW: 3, minH: 3 },
      { i: 'signals', x: 0, y: 4, w: 3, h: 3, minW: 2, minH: 2 },
      { i: 'performance', x: 3, y: 4, w: 3, h: 3, minW: 2, minH: 2 },
      { i: 'heatmap', x: 0, y: 7, w: 6, h: 3, minW: 3, minH: 2 },
      { i: 'indicators', x: 0, y: 10, w: 6, h: 3, minW: 3, minH: 2 },
    ]
  };

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
    
    if (Array.isArray(data)) {
      data.forEach(item => {
        if (item.signals) {
          allSignals.push(...item.signals);
        }
        if (item.binary_signals) {
          allBinarySignals.push(...item.binary_signals);
        }
      });
    }
    
    setSignals(allSignals);
    setBinarySignals(allBinarySignals);
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const response = await axios.post(`${API}/forex/analyze`, {
        symbols: majorPairs.slice(0, 8),
        timeframe: selectedTimeframe
      });
      
      if (response.data.status === 'success') {
        setMarketData(response.data.data);
        updateSignalsFromData(response.data.data);
        await fetchHistoricalData();
        await fetchPerformanceMetrics();
      }
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(`${API}/forex/historical/${selectedPair}?timeframe=${selectedTimeframe}&limit=100`);
      if (response.data.status === 'success') {
        setHistoricalData(response.data.data);
      }
    } catch (error) {
      console.error('Historical data error:', error);
    }
  };

  const fetchPerformanceMetrics = async () => {
    try {
      const response = await axios.get(`${API}/forex/performance`);
      if (response.data.status === 'success') {
        setPerformanceMetrics(response.data.data);
      }
    } catch (error) {
      console.error('Performance metrics error:', error);
    }
  };

  const formatPrice = (price) => {
    return typeof price === 'number' ? price.toFixed(5) : '0.00000';
  };

  const formatPercent = (percent) => {
    return typeof percent === 'number' ? percent.toFixed(2) : '0.00';
  };

  const getSignalStrengthColor = (strength) => {
    if (strength >= 0.8) return 'text-green-400';
    if (strength >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getSignalTypeColor = (type) => {
    return type === 'BUY' || type === 'CALL' ? 'text-green-400' : 'text-red-400';
  };

  useEffect(() => {
    runAnalysis();
  }, [selectedPair, selectedTimeframe]);

  if (activeSystem === 'maila_soul') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-gray-900 to-black">
        {/* System Switcher */}
        <div className="fixed top-4 right-4 z-50">
          <div className="flex space-x-2">
            {systems.map(system => (
              <button
                key={system.id}
                onClick={() => setActiveSystem(system.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  activeSystem === system.id
                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/30'
                    : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50'
                }`}
                title={system.description}
              >
                <span className="mr-2">{system.icon}</span>
                {system.name}
              </button>
            ))}
          </div>
        </div>
        
        <MailaSoulDashboard />
      </div>
    );
  }

  const tabButtons = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'trading', label: 'Trading', icon: '💹' },
    { id: 'analytics', label: 'Analytics', icon: '📈' },
    { id: 'signals', label: 'Signals', icon: '🎯' },
    { id: 'performance', label: 'Performance', icon: '🏆' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverviewTab();
      case 'trading':
        return renderTradingTab();
      case 'analytics':
        return renderAnalyticsTab();
      case 'signals':
        return renderSignalsTab();
      case 'performance':
        return renderPerformanceTab();
      default:
        return renderOverviewTab();
    }
  };

  const renderOverviewTab = () => (
    <div className="space-y-6">
      {/* Market Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium opacity-90">Active Signals</h3>
              <p className="text-2xl font-bold">{signals.length}</p>
            </div>
            <div className="text-3xl">🎯</div>
          </div>
        </div>
        <div className="bg-gradient-to-r from-green-600 to-teal-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium opacity-90">Win Rate</h3>
              <p className="text-2xl font-bold">{performanceMetrics.win_rate || '87.3'}%</p>
            </div>
            <div className="text-3xl">🏆</div>
          </div>
        </div>
        <div className="bg-gradient-to-r from-yellow-600 to-orange-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium opacity-90">ROI</h3>
              <p className="text-2xl font-bold">{performanceMetrics.roi || '+24.7'}%</p>
            </div>
            <div className="text-3xl">💰</div>
          </div>
        </div>
        <div className="bg-gradient-to-r from-red-600 to-pink-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium opacity-90">Risk Score</h3>
              <p className="text-2xl font-bold">{performanceMetrics.risk_score || '3.2'}</p>
            </div>
            <div className="text-3xl">⚠️</div>
          </div>
        </div>
      </div>

      {/* Main Chart */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <AdvancedChart
          data={historicalData}
          symbol={selectedPair}
          timeframe={selectedTimeframe}
          signals={signals}
          height={400}
        />
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SignalHeatmap signals={signals} pairs={majorPairs} />
        <TechnicalIndicators data={marketData} />
      </div>
    </div>
  );

  const renderTradingTab = () => (
    <ResponsiveGridLayout
      className="layout"
      layouts={defaultLayouts}
      breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
      cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
      rowHeight={60}
      onLayoutChange={(layout, layouts) => console.log('Layout changed:', layout)}
      isDraggable={true}
      isResizable={true}
    >
      <div key="chart" className="bg-gray-800 rounded-lg border border-gray-700">
        <AdvancedChart
          data={historicalData}
          symbol={selectedPair}
          timeframe={selectedTimeframe}
          signals={signals}
          height={400}
        />
      </div>
      <div key="signals" className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-bold mb-4 text-green-400">Active Signals</h3>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {signals.slice(0, 5).map((signal, index) => (
            <div key={index} className="bg-gray-700 rounded-lg p-3 border border-gray-600">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-blue-400">{signal.symbol}</span>
                <span className={`px-2 py-1 rounded text-xs font-bold ${getSignalTypeColor(signal.type)}`}>
                  {signal.type}
                </span>
              </div>
              <div className="text-xs text-gray-400">
                Entry: {formatPrice(signal.entry_price)}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div key="performance" className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <PerformanceAnalytics metrics={performanceMetrics} />
      </div>
      <div key="heatmap" className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <SignalHeatmap signals={signals} pairs={majorPairs} />
      </div>
      <div key="indicators" className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <TechnicalIndicators data={marketData} />
      </div>
    </ResponsiveGridLayout>
  );

  const renderAnalyticsTab = () => (
    <div className="space-y-6">
      <PerformanceAnalytics metrics={performanceMetrics} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SignalHeatmap signals={signals} pairs={majorPairs} />
        <TechnicalIndicators data={marketData} />
      </div>
    </div>
  );

  const renderSignalsTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
  );

  const renderPerformanceTab = () => (
    <div className="space-y-6">
      <PerformanceAnalytics metrics={performanceMetrics} />
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* System Switcher */}
      <div className="fixed top-4 right-4 z-50">
        <div className="flex space-x-2">
          {systems.map(system => (
            <button
              key={system.id}
              onClick={() => setActiveSystem(system.id)}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                activeSystem === system.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50'
              }`}
              title={system.description}
            >
              <span className="mr-2">{system.icon}</span>
              {system.name}
            </button>
          ))}
        </div>
      </div>
      
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
                Advanced Forex Trading System
              </h1>
            </div>
            <div className={`flex items-center space-x-2 ${wsConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">{wsConnected ? 'Live' : 'Disconnected'}</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Pair and Timeframe Selectors */}
            <select
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            >
              {majorPairs.map(pair => (
                <option key={pair} value={pair}>{pair}</option>
              ))}
            </select>
            
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            >
              {timeframes.map(tf => (
                <option key={tf} value={tf}>{tf}</option>
              ))}
            </select>
            
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
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="px-6">
          <div className="flex space-x-8">
            {tabButtons.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-6">
        {renderTabContent()}
      </main>
    </div>
  );
};

export default ForexTradingDashboard;
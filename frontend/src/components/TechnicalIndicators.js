import React, { useState } from 'react';

const TechnicalIndicators = ({ data }) => {
  const [selectedIndicator, setSelectedIndicator] = useState('overview');

  const indicators = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'momentum', name: 'Momentum', icon: '📈' },
    { id: 'trend', name: 'Trend', icon: '📉' },
    { id: 'volatility', name: 'Volatility', icon: '⚡' },
    { id: 'volume', name: 'Volume', icon: '📦' },
  ];

  const getIndicatorColor = (value, type) => {
    switch (type) {
      case 'rsi':
        if (value > 70) return 'text-red-400';
        if (value < 30) return 'text-green-400';
        return 'text-yellow-400';
      case 'macd':
        return value > 0 ? 'text-green-400' : 'text-red-400';
      case 'adx':
        if (value > 25) return 'text-green-400';
        if (value > 20) return 'text-yellow-400';
        return 'text-red-400';
      case 'stoch':
        if (value > 80) return 'text-red-400';
        if (value < 20) return 'text-green-400';
        return 'text-yellow-400';
      case 'williams':
        if (value > -20) return 'text-red-400';
        if (value < -80) return 'text-green-400';
        return 'text-yellow-400';
      case 'cci':
        if (value > 100) return 'text-red-400';
        if (value < -100) return 'text-green-400';
        return 'text-yellow-400';
      default:
        return 'text-white';
    }
  };

  const getSignalStrength = (indicators) => {
    if (!indicators) return 'Neutral';
    
    let bullishCount = 0;
    let bearishCount = 0;
    
    // RSI analysis
    if (indicators.RSI > 70) bearishCount++;
    else if (indicators.RSI < 30) bullishCount++;
    
    // MACD analysis
    if (indicators.MACD > 0) bullishCount++;
    else bearishCount++;
    
    // ADX analysis
    if (indicators.ADX > 25) {
      if (indicators.RSI > 50) bullishCount++;
      else bearishCount++;
    }
    
    // Stochastic analysis
    if (indicators.Stochastic > 80) bearishCount++;
    else if (indicators.Stochastic < 20) bullishCount++;
    
    if (bullishCount > bearishCount) return 'Bullish';
    if (bearishCount > bullishCount) return 'Bearish';
    return 'Neutral';
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'Bullish': return 'text-green-400';
      case 'Bearish': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const renderOverview = () => (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-600">
            <th className="text-left py-2 px-4 text-gray-400">Pair</th>
            <th className="text-left py-2 px-4 text-gray-400">RSI</th>
            <th className="text-left py-2 px-4 text-gray-400">MACD</th>
            <th className="text-left py-2 px-4 text-gray-400">ADX</th>
            <th className="text-left py-2 px-4 text-gray-400">Signal</th>
            <th className="text-left py-2 px-4 text-gray-400">Strength</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, index) => {
            const signal = getSignalStrength(item.indicators);
            return (
              <tr key={index} className="border-b border-gray-700 hover:bg-gray-700">
                <td className="py-2 px-4 font-semibold text-blue-400">{item.symbol}</td>
                <td className="py-2 px-4">
                  <span className={getIndicatorColor(item.indicators?.RSI, 'rsi')}>
                    {item.indicators?.RSI?.toFixed(1) || 'N/A'}
                  </span>
                </td>
                <td className="py-2 px-4">
                  <span className={getIndicatorColor(item.indicators?.MACD, 'macd')}>
                    {item.indicators?.MACD?.toFixed(4) || 'N/A'}
                  </span>
                </td>
                <td className="py-2 px-4">
                  <span className={getIndicatorColor(item.indicators?.ADX, 'adx')}>
                    {item.indicators?.ADX?.toFixed(1) || 'N/A'}
                  </span>
                </td>
                <td className="py-2 px-4">
                  <span className={`px-2 py-1 rounded text-xs font-bold ${getSignalColor(signal)}`}>
                    {signal}
                  </span>
                </td>
                <td className="py-2 px-4">
                  <div className="flex items-center">
                    <div className="w-16 bg-gray-600 rounded-full h-2 mr-2">
                      <div 
                        className={`h-2 rounded-full ${
                          signal === 'Bullish' ? 'bg-green-400' : 
                          signal === 'Bearish' ? 'bg-red-400' : 'bg-yellow-400'
                        }`}
                        style={{ width: `${Math.abs(item.indicators?.RSI - 50) * 2}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-gray-400">
                      {Math.abs(item.indicators?.RSI - 50) * 2 || 0}%
                    </span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  const renderMomentum = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {data.map((item, index) => (
        <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <h4 className="font-semibold text-blue-400 mb-3">{item.symbol}</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">RSI (14)</span>
              <span className={getIndicatorColor(item.indicators?.RSI, 'rsi')}>
                {item.indicators?.RSI?.toFixed(1) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Stochastic</span>
              <span className={getIndicatorColor(item.indicators?.Stochastic, 'stoch')}>
                {item.indicators?.Stochastic?.toFixed(1) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Williams %R</span>
              <span className={getIndicatorColor(item.indicators?.Williams_R, 'williams')}>
                {item.indicators?.Williams_R?.toFixed(1) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">CCI</span>
              <span className={getIndicatorColor(item.indicators?.CCI, 'cci')}>
                {item.indicators?.CCI?.toFixed(1) || 'N/A'}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderTrend = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {data.map((item, index) => (
        <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <h4 className="font-semibold text-blue-400 mb-3">{item.symbol}</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">MACD</span>
              <span className={getIndicatorColor(item.indicators?.MACD, 'macd')}>
                {item.indicators?.MACD?.toFixed(4) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">ADX</span>
              <span className={getIndicatorColor(item.indicators?.ADX, 'adx')}>
                {item.indicators?.ADX?.toFixed(1) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">EMA 20</span>
              <span className="text-purple-400">
                {item.indicators?.EMA_20?.toFixed(5) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">SMA 50</span>
              <span className="text-orange-400">
                {item.indicators?.SMA_50?.toFixed(5) || 'N/A'}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderVolatility = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {data.map((item, index) => (
        <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <h4 className="font-semibold text-blue-400 mb-3">{item.symbol}</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">ATR</span>
              <span className="text-yellow-400">
                {item.indicators?.ATR?.toFixed(5) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Bollinger Upper</span>
              <span className="text-green-400">
                {item.indicators?.BB_upper?.toFixed(5) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Bollinger Lower</span>
              <span className="text-red-400">
                {item.indicators?.BB_lower?.toFixed(5) || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility</span>
              <span className="text-purple-400">
                {((item.indicators?.ATR || 0) / (item.current_price || 1) * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderVolume = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {data.map((item, index) => (
        <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <h4 className="font-semibold text-blue-400 mb-3">{item.symbol}</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Volume</span>
              <span className="text-blue-400">
                {item.volume ? (item.volume / 1000000).toFixed(2) + 'M' : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volume SMA</span>
              <span className="text-green-400">
                {item.volume_sma ? (item.volume_sma / 1000000).toFixed(2) + 'M' : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volume Ratio</span>
              <span className="text-yellow-400">
                {item.volume && item.volume_sma 
                  ? (item.volume / item.volume_sma).toFixed(2) + 'x' 
                  : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">OBV</span>
              <span className="text-purple-400">
                {item.obv ? (item.obv / 1000000).toFixed(2) + 'M' : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderContent = () => {
    switch (selectedIndicator) {
      case 'overview': return renderOverview();
      case 'momentum': return renderMomentum();
      case 'trend': return renderTrend();
      case 'volatility': return renderVolatility();
      case 'volume': return renderVolume();
      default: return renderOverview();
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-yellow-400">Technical Indicators</h3>
        <div className="flex items-center space-x-2">
          {indicators.map(indicator => (
            <button
              key={indicator.id}
              onClick={() => setSelectedIndicator(indicator.id)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                selectedIndicator === indicator.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <span className="mr-1">{indicator.icon}</span>
              {indicator.name}
            </button>
          ))}
        </div>
      </div>

      {data.length > 0 ? renderContent() : (
        <div className="text-center py-8">
          <div className="text-gray-400">No data available</div>
          <div className="text-sm text-gray-500 mt-2">
            Run analysis to load technical indicators
          </div>
        </div>
      )}
    </div>
  );
};

export default TechnicalIndicators;
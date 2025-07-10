import React, { useMemo } from 'react';

const SignalHeatmap = ({ signals, pairs }) => {
  const heatmapData = useMemo(() => {
    const data = {};
    pairs.forEach(pair => {
      data[pair] = { buy: 0, sell: 0, strength: 0 };
    });

    signals.forEach(signal => {
      if (data[signal.symbol]) {
        if (signal.type === 'BUY') {
          data[signal.symbol].buy += 1;
        } else {
          data[signal.symbol].sell += 1;
        }
        data[signal.symbol].strength += signal.strength;
      }
    });

    // Calculate average strength
    Object.keys(data).forEach(pair => {
      const total = data[pair].buy + data[pair].sell;
      if (total > 0) {
        data[pair].strength = data[pair].strength / total;
      }
    });

    return data;
  }, [signals, pairs]);

  const getHeatmapColor = (strength, signalCount) => {
    if (signalCount === 0) return 'bg-gray-700';
    
    const intensity = Math.min(strength, 1);
    if (intensity >= 0.8) return 'bg-green-500';
    if (intensity >= 0.6) return 'bg-green-400';
    if (intensity >= 0.4) return 'bg-yellow-400';
    if (intensity >= 0.2) return 'bg-orange-400';
    return 'bg-red-400';
  };

  const getTextColor = (strength, signalCount) => {
    if (signalCount === 0) return 'text-gray-400';
    return 'text-white';
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-bold text-green-400 mb-4">Signal Heatmap</h3>
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
        {pairs.map(pair => {
          const data = heatmapData[pair];
          const totalSignals = data.buy + data.sell;
          const strength = data.strength;
          
          return (
            <div
              key={pair}
              className={`p-3 rounded-lg transition-all duration-200 hover:scale-105 cursor-pointer ${getHeatmapColor(strength, totalSignals)}`}
            >
              <div className={`text-center ${getTextColor(strength, totalSignals)}`}>
                <div className="font-bold text-sm">{pair}</div>
                <div className="text-xs mt-1">
                  {totalSignals > 0 ? (
                    <>
                      <div>Signals: {totalSignals}</div>
                      <div>Strength: {(strength * 100).toFixed(0)}%</div>
                      <div className="flex justify-center space-x-2 mt-1">
                        {data.buy > 0 && (
                          <span className="text-green-200 text-xs">↑{data.buy}</span>
                        )}
                        {data.sell > 0 && (
                          <span className="text-red-200 text-xs">↓{data.sell}</span>
                        )}
                      </div>
                    </>
                  ) : (
                    <div>No signals</div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-between text-sm">
        <span className="text-gray-400">Signal Strength:</span>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-gray-700 rounded"></div>
            <span className="text-gray-400">None</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-red-400 rounded"></div>
            <span className="text-gray-400">Weak</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-yellow-400 rounded"></div>
            <span className="text-gray-400">Medium</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-400 rounded"></div>
            <span className="text-gray-400">Strong</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-gray-400">Very Strong</span>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div className="bg-gray-700 rounded p-3">
          <div className="text-2xl font-bold text-green-400">
            {Object.values(heatmapData).reduce((sum, data) => sum + data.buy, 0)}
          </div>
          <div className="text-sm text-gray-400">Buy Signals</div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-2xl font-bold text-red-400">
            {Object.values(heatmapData).reduce((sum, data) => sum + data.sell, 0)}
          </div>
          <div className="text-sm text-gray-400">Sell Signals</div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-2xl font-bold text-yellow-400">
            {pairs.filter(pair => heatmapData[pair].buy + heatmapData[pair].sell > 0).length}
          </div>
          <div className="text-sm text-gray-400">Active Pairs</div>
        </div>
      </div>
    </div>
  );
};

export default SignalHeatmap;
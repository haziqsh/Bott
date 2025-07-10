import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

const AdvancedChart = ({ data, symbol, timeframe, signals, height = 400 }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const [chartType, setChartType] = useState('candlestick');
  const [indicators, setIndicators] = useState({
    sma: false,
    ema: false,
    rsi: false,
    macd: false,
    bollinger: false,
    volume: true
  });

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1f2937' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
    });

    chartRef.current = chart;

    // Create main series (candlestick or line)
    let mainSeries;
    if (chartType === 'candlestick') {
      mainSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
      });
    } else {
      mainSeries = chart.addLineSeries({
        color: '#3b82f6',
        lineWidth: 2,
      });
    }

    // Add sample data if no data provided
    const sampleData = data.length > 0 ? data : generateSampleData();
    
    if (chartType === 'candlestick') {
      mainSeries.setData(sampleData.map(item => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      })));
    } else {
      mainSeries.setData(sampleData.map(item => ({
        time: item.time,
        value: item.close,
      })));
    }

    // Add volume series
    if (indicators.volume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#6b7280',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });

      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });

      volumeSeries.setData(sampleData.map(item => ({
        time: item.time,
        value: item.volume || Math.random() * 1000000,
        color: item.close > item.open ? '#10b981' : '#ef4444',
      })));
    }

    // Add indicators
    if (indicators.sma) {
      const smaSeries = chart.addLineSeries({
        color: '#f59e0b',
        lineWidth: 2,
      });
      smaSeries.setData(calculateSMA(sampleData, 20));
    }

    if (indicators.ema) {
      const emaSeries = chart.addLineSeries({
        color: '#8b5cf6',
        lineWidth: 2,
      });
      emaSeries.setData(calculateEMA(sampleData, 20));
    }

    if (indicators.bollinger) {
      const bb = calculateBollingerBands(sampleData, 20, 2);
      const upperBandSeries = chart.addLineSeries({
        color: '#6b7280',
        lineWidth: 1,
        lineStyle: 2,
      });
      const lowerBandSeries = chart.addLineSeries({
        color: '#6b7280',
        lineWidth: 1,
        lineStyle: 2,
      });
      upperBandSeries.setData(bb.upper);
      lowerBandSeries.setData(bb.lower);
    }

    // Add signal markers
    if (signals && signals.length > 0) {
      const markers = signals.map(signal => ({
        time: signal.timestamp || Date.now() / 1000,
        position: signal.type === 'BUY' ? 'belowBar' : 'aboveBar',
        color: signal.type === 'BUY' ? '#10b981' : '#ef4444',
        shape: signal.type === 'BUY' ? 'arrowUp' : 'arrowDown',
        text: `${signal.type} ${signal.symbol}`,
        size: 1,
      }));
      mainSeries.setMarkers(markers);
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, symbol, timeframe, chartType, indicators, signals, height]);

  // Helper functions for technical indicators
  const calculateSMA = (data, period) => {
    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, item) => acc + item.close, 0);
      sma.push({
        time: data[i].time,
        value: sum / period,
      });
    }
    return sma;
  };

  const calculateEMA = (data, period) => {
    const ema = [];
    const multiplier = 2 / (period + 1);
    let previousEMA = data[0].close;
    
    ema.push({
      time: data[0].time,
      value: previousEMA,
    });

    for (let i = 1; i < data.length; i++) {
      const currentEMA = (data[i].close - previousEMA) * multiplier + previousEMA;
      ema.push({
        time: data[i].time,
        value: currentEMA,
      });
      previousEMA = currentEMA;
    }
    return ema;
  };

  const calculateBollingerBands = (data, period, stdDev) => {
    const sma = calculateSMA(data, period);
    const upper = [];
    const lower = [];

    for (let i = 0; i < sma.length; i++) {
      const dataIndex = i + period - 1;
      const slice = data.slice(dataIndex - period + 1, dataIndex + 1);
      const variance = slice.reduce((acc, item) => acc + Math.pow(item.close - sma[i].value, 2), 0) / period;
      const standardDeviation = Math.sqrt(variance);

      upper.push({
        time: sma[i].time,
        value: sma[i].value + (standardDeviation * stdDev),
      });
      lower.push({
        time: sma[i].time,
        value: sma[i].value - (standardDeviation * stdDev),
      });
    }

    return { upper, lower };
  };

  const generateSampleData = () => {
    const data = [];
    let basePrice = 1.1000;
    const now = Date.now() / 1000;

    for (let i = 0; i < 100; i++) {
      const time = now - (100 - i) * 60; // 1-minute intervals
      const change = (Math.random() - 0.5) * 0.001;
      const open = basePrice;
      const close = basePrice + change;
      const high = Math.max(open, close) + Math.random() * 0.0005;
      const low = Math.min(open, close) - Math.random() * 0.0005;
      
      data.push({
        time: time,
        open: parseFloat(open.toFixed(5)),
        high: parseFloat(high.toFixed(5)),
        low: parseFloat(low.toFixed(5)),
        close: parseFloat(close.toFixed(5)),
        volume: Math.floor(Math.random() * 1000000),
      });
      
      basePrice = close;
    }
    return data;
  };

  const toggleIndicator = (indicator) => {
    setIndicators(prev => ({
      ...prev,
      [indicator]: !prev[indicator]
    }));
  };

  return (
    <div className="w-full">
      {/* Chart Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-bold text-white">
            {symbol} - {timeframe}
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setChartType('candlestick')}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                chartType === 'candlestick' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Candlestick
            </button>
            <button
              onClick={() => setChartType('line')}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                chartType === 'line' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Line
            </button>
          </div>
        </div>

        {/* Technical Indicators */}
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-400">Indicators:</span>
          {Object.entries(indicators).map(([indicator, enabled]) => (
            <button
              key={indicator}
              onClick={() => toggleIndicator(indicator)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                enabled 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {indicator.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Container */}
      <div 
        ref={chartContainerRef} 
        className="w-full bg-gray-800 rounded-lg border border-gray-700"
        style={{ height: `${height}px` }}
      />

      {/* Chart Info */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Current Price</div>
          <div className="text-white font-bold">1.10250</div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">24h Change</div>
          <div className="text-green-400 font-bold">+0.35%</div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Volume</div>
          <div className="text-white font-bold">2.4M</div>
        </div>
        <div className="bg-gray-700 rounded p-3">
          <div className="text-gray-400">Volatility</div>
          <div className="text-yellow-400 font-bold">Medium</div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedChart;
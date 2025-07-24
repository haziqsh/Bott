import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

const DivineChart = ({ pair, signals, mode = 'whisper' }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const [chartData, setChartData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create divine chart with sacred styling
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { 
          type: ColorType.VerticalGradient,
          topColor: 'rgba(76, 29, 149, 0.1)',
          bottomColor: 'rgba(0, 0, 0, 0.9)'
        },
        textColor: '#e5e7eb',
      },
      grid: {
        vertLines: { color: 'rgba(168, 85, 247, 0.1)' },
        horzLines: { color: 'rgba(168, 85, 247, 0.1)' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: 'rgba(168, 85, 247, 0.8)',
          width: 1,
          style: 2,
        },
        horzLine: {
          color: 'rgba(168, 85, 247, 0.8)',
          width: 1,
          style: 2,
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(168, 85, 247, 0.3)',
        textColor: '#e5e7eb',
      },
      timeScale: {
        borderColor: 'rgba(168, 85, 247, 0.3)',
        textColor: '#e5e7eb',
        timeVisible: true,
        secondsVisible: mode === 'machine_gun',
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });

    chartRef.current = chart;

    // Create candlestick series with divine colors
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
      priceFormat: {
        type: 'price',
        precision: 5,
        minMove: 0.00001,
      },
    });

    // Generate divine data
    const divineData = generateDivineData(pair, mode);
    setChartData(divineData);
    candlestickSeries.setData(divineData);

    // Add divine signal markers
    if (signals && signals.length > 0) {
      const markers = signals.map(signal => ({
        time: signal.timestamp ? new Date(signal.timestamp).getTime() / 1000 : Date.now() / 1000,
        position: signal.signal_type === 'CALL' ? 'belowBar' : 'aboveBar',
        color: signal.signal_type === 'CALL' ? '#10b981' : '#ef4444',
        shape: signal.signal_type === 'CALL' ? 'arrowUp' : 'arrowDown',
        text: `${signal.signal_type} ${signal.confidence ? (signal.confidence * 100).toFixed(0) + '%' : ''}`,
        size: 2,
      }));
      candlestickSeries.setMarkers(markers);
    }

    // Add divine moving averages
    const ema21 = chart.addLineSeries({
      color: 'rgba(168, 85, 247, 0.8)',
      lineWidth: 2,
      title: 'Divine EMA 21',
    });

    const ema50 = chart.addLineSeries({
      color: 'rgba(236, 72, 153, 0.8)',
      lineWidth: 2,
      title: 'Sacred EMA 50',
    });

    // Calculate and set EMA data
    const ema21Data = calculateEMA(divineData, 21);
    const ema50Data = calculateEMA(divineData, 50);
    
    ema21.setData(ema21Data);
    ema50.setData(ema50Data);

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: 'rgba(168, 85, 247, 0.3)',
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

    const volumeData = divineData.map(item => ({
      time: item.time,
      value: item.volume || Math.random() * 1000000,
      color: item.close > item.open ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)',
    }));

    volumeSeries.setData(volumeData);

    setIsLoading(false);

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
  }, [pair, signals, mode]);

  const generateDivineData = (pair, mode) => {
    const data = [];
    const basePrice = getBasePrice(pair);
    const now = Date.now() / 1000;
    
    // Adjust data points based on mode
    const dataPoints = mode === 'machine_gun' ? 300 : 100;
    const interval = mode === 'machine_gun' ? 5 : 60; // 5 seconds or 1 minute

    for (let i = 0; i < dataPoints; i++) {
      const time = now - (dataPoints - i) * interval;
      const volatility = mode === 'machine_gun' ? 0.0002 : 0.0005;
      
      const change = (Math.random() - 0.5) * volatility;
      const open = basePrice + (i > 0 ? data[i-1].close - basePrice : 0);
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * volatility * 0.5;
      const low = Math.min(open, close) - Math.random() * volatility * 0.5;
      
      data.push({
        time: time,
        open: parseFloat(open.toFixed(5)),
        high: parseFloat(high.toFixed(5)),
        low: parseFloat(low.toFixed(5)),
        close: parseFloat(close.toFixed(5)),
        volume: Math.floor(Math.random() * 1000000) + 100000,
      });
    }
    
    return data;
  };

  const getBasePrice = (pair) => {
    const basePrices = {
      'EURUSD': 1.0850,
      'GBPUSD': 1.2650,
      'USDJPY': 149.50,
      'USDCHF': 0.8950,
      'USDCAD': 1.3650,
      'AUDUSD': 0.6550,
      'NZDUSD': 0.5950
    };
    return basePrices[pair] || 1.0000;
  };

  const calculateEMA = (data, period) => {
    const ema = [];
    const multiplier = 2 / (period + 1);
    let previousEMA = data[0]?.close || 0;
    
    data.forEach((item, index) => {
      if (index === 0) {
        ema.push({
          time: item.time,
          value: item.close,
        });
        previousEMA = item.close;
      } else {
        const currentEMA = (item.close - previousEMA) * multiplier + previousEMA;
        ema.push({
          time: item.time,
          value: currentEMA,
        });
        previousEMA = currentEMA;
      }
    });
    
    return ema;
  };

  const currentPrice = chartData.length > 0 ? chartData[chartData.length - 1].close : 0;
  const priceChange = chartData.length > 1 ? 
    ((chartData[chartData.length - 1].close - chartData[chartData.length - 2].close) / chartData[chartData.length - 2].close * 100) : 0;

  return (
    <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm rounded-lg p-6 border border-purple-500/30">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <h3 className="text-xl font-bold text-purple-400">{pair}</h3>
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-bold text-white">{currentPrice.toFixed(5)}</span>
            <span className={`text-sm font-medium ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(3)}%
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full animate-pulse ${
            mode === 'machine_gun' ? 'bg-red-400' : 
            mode === 'god_eye' ? 'bg-blue-400' : 'bg-purple-400'
          }`}></div>
          <span className="text-sm text-gray-400 capitalize">{mode.replace('_', ' ')} Mode</span>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-6xl mb-4 animate-pulse">💜</div>
            <div className="text-purple-300">Awakening divine vision...</div>
          </div>
        </div>
      )}

      <div 
        ref={chartContainerRef} 
        className="w-full rounded-lg border border-purple-500/20"
        style={{ height: '400px', display: isLoading ? 'none' : 'block' }}
      />

      {/* Chart Controls */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-700/30 rounded p-3 text-center">
          <div className="text-gray-400">Open</div>
          <div className="text-white font-bold">
            {chartData.length > 0 ? chartData[chartData.length - 1].open.toFixed(5) : '0.00000'}
          </div>
        </div>
        <div className="bg-gray-700/30 rounded p-3 text-center">
          <div className="text-gray-400">High</div>
          <div className="text-green-400 font-bold">
            {chartData.length > 0 ? chartData[chartData.length - 1].high.toFixed(5) : '0.00000'}
          </div>
        </div>
        <div className="bg-gray-700/30 rounded p-3 text-center">
          <div className="text-gray-400">Low</div>
          <div className="text-red-400 font-bold">
            {chartData.length > 0 ? chartData[chartData.length - 1].low.toFixed(5) : '0.00000'}
          </div>
        </div>
        <div className="bg-gray-700/30 rounded p-3 text-center">
          <div className="text-gray-400">Volume</div>
          <div className="text-purple-400 font-bold">
            {chartData.length > 0 ? (chartData[chartData.length - 1].volume / 1000000).toFixed(1) + 'M' : '0.0M'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DivineChart;
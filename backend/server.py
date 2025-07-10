from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from transformers import pipeline
import ta  # Using ta library instead of pandas-ta
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import advanced AI models and services
from .ai_models import AdvancedAIModels
from .news_sentiment import news_sentiment_service

# Import advanced trading strategies
# from .trading_strategies import (
#     NostalgiaForInfinityStrategy,
#     IchimokuStrategy,
#     SuperTrendStrategy,
#     LSTMNeuralStrategy,
#     BinaryOptionsStrategy,
#     QuantitativeFinanceStrategy
# )

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Forex AI Trading Agent", description="Ultimate AI-Powered Forex Trading System")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# AI Models initialization
try:
    print("🤖 Loading AI models...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    pattern_analyzer = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    print("✅ AI models loaded successfully")
except Exception as e:
    print(f"⚠️ AI models failed to load: {e}")
    sentiment_analyzer = None
    pattern_analyzer = None
    print("⚠️ AI features disabled for faster startup")

# Alpha Vantage API setup (free tier)
AV_API_KEY = "demo"  # Using demo key for development
av_fx = ForeignExchange(key=AV_API_KEY, output_format='pandas')

# Forex Trading Agent Class
class ForexTradingAgent:
    def __init__(self):
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'EURCHF', 'AUDJPY', 'GBPCHF',
            'EURAUD', 'USDCNH', 'USDSEK', 'USDNOK', 'USDDKK', 'USDSGD'
        ]
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.active_signals = {}
        self.performance_metrics = {}
        self.strategy_accuracy = {}
        
        # Initialize advanced AI models
        print("🤖 Initializing Advanced AI Models...")
        self.ai_models = AdvancedAIModels()
        
        # Initialize news and sentiment service
        self.news_service = news_sentiment_service
        
        # Initialize advanced trading strategies
        # self.nostalgia_strategy = NostalgiaForInfinityStrategy()
        # self.ichimoku_strategy = IchimokuStrategy()
        # self.supertrend_strategy = SuperTrendStrategy()
        # self.lstm_strategy = LSTMNeuralStrategy()
        # self.binary_strategy = BinaryOptionsStrategy()
        # self.quant_strategy = QuantitativeFinanceStrategy()
        
        # Strategy performance tracking
        self.strategy_performance = {
            'nostalgia': {'wins': 0, 'losses': 0, 'total_trades': 0},
            'ichimoku': {'wins': 0, 'losses': 0, 'total_trades': 0},
            'supertrend': {'wins': 0, 'losses': 0, 'total_trades': 0},
            'lstm': {'wins': 0, 'losses': 0, 'total_trades': 0},
            'binary': {'wins': 0, 'losses': 0, 'total_trades': 0},
            'quant': {'wins': 0, 'losses': 0, 'total_trades': 0}
        }
        
        # Market sentiment cache
        self.sentiment_cache = {}
        self.sentiment_last_updated = None
        
    def get_forex_data(self, symbol: str, timeframe: str = '1h', periods: int = 1000):
        """Get forex data from multiple sources"""
        try:
            # Convert symbol format for yfinance
            if '/' not in symbol:
                symbol_yf = f"{symbol[:3]}{symbol[3:]}=X"
            else:
                symbol_yf = symbol.replace('/', '') + '=X'
            
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol_yf)
            data = ticker.history(period="1mo", interval=timeframe)
            
            if data.empty:
                raise Exception("No data from Yahoo Finance")
                
            return data
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_all_indicators(self, data: pd.DataFrame):
        """Calculate comprehensive technical indicators using ta library"""
        if data.empty:
            return {}
            
        try:
            # Ensure proper column names
            if len(data.columns) >= 5:
                # Handle yfinance data format
                if 'Dividends' in data.columns:
                    # Standard yfinance format
                    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                else:
                    # Rename columns if needed
                    data.columns = data.columns[:5]
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                # Not enough columns, create dummy data for testing
                print(f"Warning: Data has insufficient columns: {data.columns}")
                return {}
            
            # Price-based indicators
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
            
            data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
            data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
            data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(data['Close'], window=20, window_dev=2)
            bb_low = ta.volatility.bollinger_lband(data['Close'], window=20, window_dev=2)
            bb_mid = ta.volatility.bollinger_mavg(data['Close'], window=20)
            
            data['BB_Upper'] = bb_high
            data['BB_Lower'] = bb_low
            data['BB_Middle'] = bb_mid
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # MACD
            data['MACD'] = ta.trend.macd_diff(data['Close'])
            data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
            data['MACD_Histogram'] = ta.trend.macd(data['Close'])
            
            # Stochastic
            data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Williams %R
            data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
            
            # CCI (Commodity Channel Index)
            data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
            
            # ADX (Average Directional Index)
            data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
            data['DI_Plus'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'])
            data['DI_Minus'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'])
            
            # ATR (Average True Range)
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Volume indicators (if available)
            if 'Volume' in data.columns:
                data['Volume_SMA'] = ta.trend.sma_indicator(data['Volume'], window=20)
                data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            # Pivot Points
            data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
            data['R1'] = 2 * data['Pivot'] - data['Low']
            data['S1'] = 2 * data['Pivot'] - data['High']
            data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
            data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
            
            return data
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return data
    
    def analyze_market_patterns(self, symbol: str, data: pd.DataFrame):
        """Advanced AI-powered pattern recognition and trend analysis"""
        try:
            if data.empty or len(data) < 100:
                return {'pattern': 'insufficient_data', 'confidence': 0.0, 'trend': 'unknown'}
            
            # Technical pattern detection
            latest_20 = data.tail(20)
            
            # Trend analysis using multiple timeframes
            short_trend = self.detect_trend(data.tail(20))
            medium_trend = self.detect_trend(data.tail(50))
            long_trend = self.detect_trend(data.tail(100))
            
            # Pattern detection using price action
            patterns = self.detect_chart_patterns(data.tail(50))
            
            # Volume analysis (if available)
            volume_pattern = self.analyze_volume_pattern(data.tail(20))
            
            # Volatility analysis
            volatility_state = self.analyze_volatility(data.tail(20))
            
            # Market regime detection
            market_regime = self.detect_market_regime(data.tail(100))
            
            # AI-powered trend prediction
            trend_prediction = self.predict_trend_direction(data.tail(200))
            
            return {
                'symbol': symbol,
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'long_trend': long_trend,
                'patterns': patterns,
                'volume_pattern': volume_pattern,
                'volatility_state': volatility_state,
                'market_regime': market_regime,
                'trend_prediction': trend_prediction,
                'confidence': self.calculate_pattern_confidence(short_trend, medium_trend, long_trend, patterns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing patterns for {symbol}: {e}")
            return {'pattern': 'error', 'confidence': 0.0, 'trend': 'unknown'}
    def detect_trend(self, data: pd.DataFrame):
        """Detect trend using multiple indicators"""
        try:
            if data.empty or len(data) < 10:
                return 'unknown'
            
            # EMA trend
            ema_short = data['Close'].ewm(span=8).mean()
            ema_long = data['Close'].ewm(span=21).mean()
            
            current_ema_short = ema_short.iloc[-1]
            current_ema_long = ema_long.iloc[-1]
            prev_ema_short = ema_short.iloc[-2]
            prev_ema_long = ema_long.iloc[-2]
            
            # Price trend
            price_trend = 'up' if data['Close'].iloc[-1] > data['Close'].iloc[-5] else 'down'
            
            # EMA trend
            ema_trend = 'up' if current_ema_short > current_ema_long else 'down'
            
            # ADX trend strength
            adx_value = data.get('ADX', pd.Series([0])).iloc[-1] if 'ADX' in data.columns else 0
            trend_strength = 'strong' if adx_value > 25 else 'weak'
            
            # Combine signals
            if price_trend == 'up' and ema_trend == 'up' and trend_strength == 'strong':
                return 'strong_uptrend'
            elif price_trend == 'up' and ema_trend == 'up':
                return 'uptrend'
            elif price_trend == 'down' and ema_trend == 'down' and trend_strength == 'strong':
                return 'strong_downtrend'
            elif price_trend == 'down' and ema_trend == 'down':
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            print(f"Error detecting trend: {e}")
            return 'unknown'
    
    def detect_chart_patterns(self, data: pd.DataFrame):
        """Detect common chart patterns"""
        try:
            if data.empty or len(data) < 20:
                return {'pattern': 'insufficient_data', 'reliability': 0.0}
            
            patterns = []
            
            # Support and Resistance levels
            highs = data['High'].rolling(window=5).max()
            lows = data['Low'].rolling(window=5).min()
            
            # Double Top/Bottom detection
            recent_highs = highs.tail(10)
            recent_lows = lows.tail(10)
            
            # Head and Shoulders detection
            if len(recent_highs) >= 3:
                high_values = recent_highs.values
                if len(high_values) >= 3:
                    if abs(high_values[-1] - high_values[-3]) < abs(high_values[-2] - high_values[-1]) * 0.5:
                        patterns.append({'pattern': 'head_and_shoulders', 'reliability': 0.7})
            
            # Triangle patterns
            if len(data) >= 20:
                price_range = data['High'].tail(20) - data['Low'].tail(20)
                if price_range.std() < price_range.mean() * 0.3:
                    patterns.append({'pattern': 'triangle', 'reliability': 0.6})
            
            # Breakout patterns
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                bb_squeeze = (data['BB_Upper'] - data['BB_Lower']) / data['Close']
                if bb_squeeze.iloc[-1] < bb_squeeze.mean() * 0.8:
                    patterns.append({'pattern': 'squeeze', 'reliability': 0.8})
            
            return patterns if patterns else [{'pattern': 'no_pattern', 'reliability': 0.0}]
            
        except Exception as e:
            print(f"Error detecting chart patterns: {e}")
            return [{'pattern': 'error', 'reliability': 0.0}]
    
    def analyze_volume_pattern(self, data: pd.DataFrame):
        """Analyze volume patterns"""
        try:
            if 'Volume' not in data.columns or data['Volume'].sum() == 0:
                return {'pattern': 'no_volume_data', 'strength': 0.0}
            
            volume_avg = data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]
            
            volume_ratio = current_volume / volume_avg
            
            if volume_ratio > 1.5:
                return {'pattern': 'high_volume', 'strength': min(volume_ratio, 3.0)}
            elif volume_ratio < 0.5:
                return {'pattern': 'low_volume', 'strength': volume_ratio}
            else:
                return {'pattern': 'normal_volume', 'strength': volume_ratio}
                
        except Exception as e:
            print(f"Error analyzing volume: {e}")
            return {'pattern': 'error', 'strength': 0.0}
    
    def analyze_volatility(self, data: pd.DataFrame):
        """Analyze market volatility state"""
        try:
            if data.empty or len(data) < 10:
                return {'state': 'unknown', 'level': 0.0}
            
            # Calculate ATR-based volatility
            atr_current = data.get('ATR', pd.Series([0])).iloc[-1] if 'ATR' in data.columns else 0
            atr_avg = data.get('ATR', pd.Series([0])).mean() if 'ATR' in data.columns else 0
            
            if atr_avg > 0:
                volatility_ratio = atr_current / atr_avg
                if volatility_ratio > 1.5:
                    return {'state': 'high_volatility', 'level': min(volatility_ratio, 3.0)}
                elif volatility_ratio < 0.7:
                    return {'state': 'low_volatility', 'level': volatility_ratio}
                else:
                    return {'state': 'normal_volatility', 'level': volatility_ratio}
            
            return {'state': 'unknown', 'level': 0.0}
            
        except Exception as e:
            print(f"Error analyzing volatility: {e}")
            return {'state': 'error', 'level': 0.0}
    
    def detect_market_regime(self, data: pd.DataFrame):
        """Detect current market regime"""
        try:
            if data.empty or len(data) < 50:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            # Calculate regime indicators
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            
            # Volatility regime
            volatility = data['Close'].rolling(window=20).std()
            vol_avg = volatility.mean()
            current_vol = volatility.iloc[-1]
            
            # Trend regime
            price_above_sma20 = data['Close'].iloc[-1] > sma_20.iloc[-1]
            sma20_above_sma50 = sma_20.iloc[-1] > sma_50.iloc[-1]
            
            # Determine regime
            if price_above_sma20 and sma20_above_sma50:
                if current_vol < vol_avg * 0.8:
                    return {'regime': 'trending_bull_low_vol', 'confidence': 0.8}
                else:
                    return {'regime': 'trending_bull_high_vol', 'confidence': 0.7}
            elif not price_above_sma20 and not sma20_above_sma50:
                if current_vol < vol_avg * 0.8:
                    return {'regime': 'trending_bear_low_vol', 'confidence': 0.8}
                else:
                    return {'regime': 'trending_bear_high_vol', 'confidence': 0.7}
            else:
                return {'regime': 'ranging', 'confidence': 0.6}
                
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return {'regime': 'error', 'confidence': 0.0}
    
    def predict_trend_direction(self, data: pd.DataFrame):
        """AI-powered trend direction prediction"""
        try:
            if data.empty or len(data) < 50:
                return {'direction': 'unknown', 'confidence': 0.0, 'horizon': '1h'}
            
            # Feature engineering for prediction
            features = []
            
            # Price momentum
            price_momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            features.append(price_momentum)
            
            # RSI momentum
            rsi_current = data.get('RSI', pd.Series([50])).iloc[-1] if 'RSI' in data.columns else 50
            rsi_momentum = (rsi_current - 50) / 50
            features.append(rsi_momentum)
            
            # MACD momentum
            macd_current = data.get('MACD', pd.Series([0])).iloc[-1] if 'MACD' in data.columns else 0
            macd_signal = data.get('MACD_Signal', pd.Series([0])).iloc[-1] if 'MACD_Signal' in data.columns else 0
            macd_momentum = macd_current - macd_signal
            features.append(macd_momentum)
            
            # Volume momentum
            if 'Volume' in data.columns:
                volume_momentum = (data['Volume'].iloc[-5:].mean() - data['Volume'].iloc[-20:-5].mean()) / data['Volume'].iloc[-20:-5].mean()
                features.append(volume_momentum)
            else:
                features.append(0)
            
            # Simple ensemble prediction
            bullish_signals = sum(1 for f in features if f > 0.1)
            bearish_signals = sum(1 for f in features if f < -0.1)
            
            if bullish_signals > bearish_signals:
                confidence = min(bullish_signals / len(features), 0.9)
                return {'direction': 'bullish', 'confidence': confidence, 'horizon': '1h'}
            elif bearish_signals > bullish_signals:
                confidence = min(bearish_signals / len(features), 0.9)
                return {'direction': 'bearish', 'confidence': confidence, 'horizon': '1h'}
            else:
                return {'direction': 'neutral', 'confidence': 0.5, 'horizon': '1h'}
                
        except Exception as e:
            print(f"Error predicting trend: {e}")
            return {'direction': 'error', 'confidence': 0.0, 'horizon': '1h'}
    
    def calculate_pattern_confidence(self, short_trend, medium_trend, long_trend, patterns):
        """Calculate overall pattern confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Trend alignment bonus
            if short_trend == medium_trend == long_trend:
                confidence += 0.3
            elif short_trend == medium_trend or medium_trend == long_trend:
                confidence += 0.1
            
            # Pattern reliability bonus
            if patterns:
                avg_pattern_reliability = sum(p.get('reliability', 0) for p in patterns) / len(patterns)
                confidence += avg_pattern_reliability * 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
            
    def generate_advanced_signals(self, symbol: str, data: pd.DataFrame):
        """Generate signals using advanced trading strategies with AI enhancement"""
        try:
            if data.empty or len(data) < 100:
                return []
                
            signals = []
            
            # Get AI insights first
            ai_insights = self.get_ai_insights(symbol, data)
            
            # Advanced NostalgiaForInfinity-inspired Strategy
            try:
                # EMA alignment check
                ema_8 = data['Close'].ewm(span=8).mean()
                ema_12 = data['Close'].ewm(span=12).mean()
                ema_20 = data['Close'].ewm(span=20).mean()
                ema_50 = data['Close'].ewm(span=50).mean()
                
                # Current values
                current_price = data['Close'].iloc[-1]
                current_ema_8 = ema_8.iloc[-1]
                current_ema_12 = ema_12.iloc[-1]
                current_ema_20 = ema_20.iloc[-1]
                current_ema_50 = ema_50.iloc[-1]
                
                # RSI and MACD from existing indicators
                current_rsi = data.get('RSI', pd.Series([50])).iloc[-1]
                current_macd = data.get('MACD', pd.Series([0])).iloc[-1]
                current_macd_signal = data.get('MACD_Signal', pd.Series([0])).iloc[-1]
                current_adx = data.get('ADX', pd.Series([20])).iloc[-1]
                
                # NostalgiaForInfinity-style buy condition with AI enhancement
                nostalgia_buy = (
                    current_ema_8 > current_ema_12 and
                    current_ema_12 > current_ema_20 and
                    current_ema_20 > current_ema_50 and
                    current_price > current_ema_20 and
                    current_rsi > 50 and current_rsi < 80 and
                    current_macd > current_macd_signal and
                    current_adx > 25
                )
                
                # Enhance with AI insights
                ai_boost = 0.0
                if ai_insights['ensemble']['prediction'] == 'buy':
                    ai_boost += 0.1
                if ai_insights['sentiment']['sentiment'] == 'bullish':
                    ai_boost += 0.05
                
                if nostalgia_buy:
                    strength = min(0.95, 0.85 + ai_boost)
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'strategy': 'NostalgiaForInfinity (AI Enhanced)',
                        'strength': strength,
                        'entry_price': float(current_price),
                        'stop_loss': float(current_price * 0.995),
                        'take_profit': float(current_price * 1.01),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat(),
                        'ai_insights': ai_insights
                    })
                    
                # NostalgiaForInfinity-style sell condition with AI enhancement
                nostalgia_sell = (
                    current_ema_8 < current_ema_12 and
                    current_ema_12 < current_ema_20 and
                    current_price < current_ema_20 and
                    (current_rsi > 70 or current_macd < current_macd_signal)
                )
                
                # Enhance with AI insights
                ai_boost = 0.0
                if ai_insights['ensemble']['prediction'] == 'sell':
                    ai_boost += 0.1
                if ai_insights['sentiment']['sentiment'] == 'bearish':
                    ai_boost += 0.05
                
                if nostalgia_sell:
                    strength = min(0.95, 0.85 + ai_boost)
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'strategy': 'NostalgiaForInfinity (AI Enhanced)',
                        'strength': strength,
                        'entry_price': float(current_price),
                        'stop_loss': float(current_price * 1.005),
                        'take_profit': float(current_price * 0.99),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat(),
                        'ai_insights': ai_insights
                    })
            except Exception as e:
                print(f"Error in NostalgiaForInfinity strategy: {e}")
                
            # Advanced Ichimoku Strategy
            try:
                # Calculate Ichimoku components
                high_9 = data['High'].rolling(window=9).max()
                low_9 = data['Low'].rolling(window=9).min()
                tenkan_sen = (high_9 + low_9) / 2
                
                high_26 = data['High'].rolling(window=26).max()
                low_26 = data['Low'].rolling(window=26).min()
                kijun_sen = (high_26 + low_26) / 2
                
                senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
                
                high_52 = data['High'].rolling(window=52).max()
                low_52 = data['Low'].rolling(window=52).min()
                senkou_b = ((high_52 + low_52) / 2).shift(26)
                
                # Current values
                current_tenkan = tenkan_sen.iloc[-1]
                current_kijun = kijun_sen.iloc[-1]
                current_senkou_a = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else current_price
                current_senkou_b = senkou_b.iloc[-1] if not pd.isna(senkou_b.iloc[-1]) else current_price
                
                # Ichimoku buy signal
                ichimoku_buy = (
                    current_price > current_senkou_a and
                    current_price > current_senkou_b and
                    current_tenkan > current_kijun and
                    current_senkou_a > current_senkou_b
                )
                
                if ichimoku_buy:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'strategy': 'Ichimoku_Strong',
                        'strength': 0.9,
                        'entry_price': float(current_price),
                        'stop_loss': float(current_kijun),
                        'take_profit': float(current_price * 1.015),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                # Ichimoku sell signal
                ichimoku_sell = (
                    current_price < current_senkou_a and
                    current_price < current_senkou_b and
                    current_tenkan < current_kijun and
                    current_senkou_a < current_senkou_b
                )
                
                if ichimoku_sell:
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'strategy': 'Ichimoku_Strong',
                        'strength': 0.9,
                        'entry_price': float(current_price),
                        'stop_loss': float(current_kijun),
                        'take_profit': float(current_price * 0.985),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Error in Ichimoku strategy: {e}")
                
            # Advanced SuperTrend Strategy
            try:
                # Calculate SuperTrend
                atr = data.get('ATR', pd.Series([0.001])).rolling(window=10).mean()
                hl2 = (data['High'] + data['Low']) / 2
                multiplier = 3.0
                
                upper_band = hl2 + (multiplier * atr)
                lower_band = hl2 - (multiplier * atr)
                
                # Simple SuperTrend logic
                supertrend = pd.Series(index=data.index, dtype=float)
                supertrend.iloc[0] = current_price
                
                for i in range(1, len(data)):
                    if data['Close'].iloc[i] > upper_band.iloc[i-1]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                    elif data['Close'].iloc[i] < lower_band.iloc[i-1]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                    else:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                
                # SuperTrend signals
                current_supertrend = supertrend.iloc[-1]
                prev_supertrend = supertrend.iloc[-2] if len(supertrend) > 1 else current_supertrend
                
                # SuperTrend buy signal
                if current_price > current_supertrend and current_price > prev_supertrend:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'strategy': 'SuperTrend_Strong',
                        'strength': 0.8,
                        'entry_price': float(current_price),
                        'stop_loss': float(current_supertrend),
                        'take_profit': float(current_price * 1.012),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # SuperTrend sell signal
                if current_price < current_supertrend and current_price < prev_supertrend:
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'strategy': 'SuperTrend_Strong',
                        'strength': 0.8,
                        'entry_price': float(current_price),
                        'stop_loss': float(current_supertrend),
                        'take_profit': float(current_price * 0.988),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Error in SuperTrend strategy: {e}")
                
            # Advanced ML-based Strategy
            try:
                # Simple ML-inspired signal based on multiple indicators
                features = []
                
                # RSI deviation from 50
                rsi_signal = abs(current_rsi - 50) / 50
                features.append(rsi_signal)
                
                # MACD signal strength
                macd_signal = abs(current_macd - current_macd_signal) / (abs(current_macd_signal) + 0.0001)
                features.append(macd_signal)
                
                # ADX trend strength
                adx_signal = current_adx / 100
                features.append(adx_signal)
                
                # Volume signal (if available)
                if 'Volume' in data.columns:
                    volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(window=20).mean().iloc[-1]
                    features.append(min(volume_ratio, 5.0) / 5.0)
                else:
                    features.append(0.5)
                
                # Combine features for ML signal
                ml_signal_strength = sum(features) / len(features)
                
                # ML buy signal
                if (current_rsi < 30 and current_macd > current_macd_signal and 
                    current_adx > 25 and ml_signal_strength > 0.6):
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'strategy': 'ML_Advanced',
                        'strength': float(ml_signal_strength),
                        'entry_price': float(current_price),
                        'stop_loss': float(current_price * 0.995),
                        'take_profit': float(current_price * 1.01),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # ML sell signal
                if (current_rsi > 70 and current_macd < current_macd_signal and 
                    current_adx > 25 and ml_signal_strength > 0.6):
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'strategy': 'ML_Advanced',
                        'strength': float(ml_signal_strength),
                        'entry_price': float(current_price),
                        'stop_loss': float(current_price * 1.005),
                        'take_profit': float(current_price * 0.99),
                        'timeframe': '1h',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Error in ML strategy: {e}")
                
            return signals
            
        except Exception as e:
            print(f"Error generating advanced signals for {symbol}: {e}")
            return []
        """Generate trading signals using multiple strategies"""
        if data.empty or len(data) < 50:
            return []
            
        signals = []
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Get AI pattern analysis
        pattern_analysis = self.analyze_market_patterns(symbol, data)
        
        try:
            # Strategy 1: RSI + MACD Confluence
            if 'RSI' in data.columns and 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                if (pd.notna(latest['RSI']) and pd.notna(latest['MACD']) and 
                    pd.notna(latest['MACD_Signal']) and pd.notna(prev['MACD']) and pd.notna(prev['MACD_Signal'])):
                    
                    if latest['RSI'] < 30 and latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'strategy': 'RSI_MACD_Confluence',
                            'strength': 0.8 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 0.995,
                            'take_profit': latest['Close'] * 1.015,
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
                    elif latest['RSI'] > 70 and latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'RSI_MACD_Confluence',
                            'strength': 0.8 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 1.005,
                            'take_profit': latest['Close'] * 0.985,
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
            
            # Strategy 2: Bollinger Bands + RSI
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns and 'RSI' in data.columns:
                if (pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']) and 
                    pd.notna(latest['RSI']) and pd.notna(latest['BB_Middle'])):
                    
                    if latest['Close'] <= latest['BB_Lower'] and latest['RSI'] < 30:
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'strategy': 'BB_RSI_Oversold',
                            'strength': 0.75 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['BB_Lower'] * 0.995,
                            'take_profit': latest['BB_Middle'],
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
                    elif latest['Close'] >= latest['BB_Upper'] and latest['RSI'] > 70:
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'BB_RSI_Overbought',
                            'strength': 0.75 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['BB_Upper'] * 1.005,
                            'take_profit': latest['BB_Middle'],
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
            
            # Strategy 3: EMA Crossover + ADX
            if 'EMA_10' in data.columns and 'EMA_20' in data.columns and 'ADX' in data.columns:
                if (pd.notna(latest['EMA_10']) and pd.notna(latest['EMA_20']) and 
                    pd.notna(latest['ADX']) and pd.notna(prev['EMA_10']) and pd.notna(prev['EMA_20'])):
                    
                    if (latest['EMA_10'] > latest['EMA_20'] and prev['EMA_10'] <= prev['EMA_20'] and 
                        latest['ADX'] > 25):
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'strategy': 'EMA_Cross_ADX',
                            'strength': 0.7 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['EMA_20'] * 0.995,
                            'take_profit': latest['Close'] * 1.02,
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
                    elif (latest['EMA_10'] < latest['EMA_20'] and prev['EMA_10'] >= prev['EMA_20'] and 
                          latest['ADX'] > 25):
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'EMA_Cross_ADX',
                            'strength': 0.7 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['EMA_20'] * 1.005,
                            'take_profit': latest['Close'] * 0.98,
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
            
            # Strategy 4: Stochastic + Williams %R
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns and 'Williams_R' in data.columns:
                if (pd.notna(latest['Stoch_K']) and pd.notna(latest['Stoch_D']) and 
                    pd.notna(latest['Williams_R'])):
                    
                    if (latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20 and 
                        latest['Williams_R'] < -80 and latest['Stoch_K'] > latest['Stoch_D']):
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'strategy': 'Stoch_Williams_Oversold',
                            'strength': 0.6 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 0.995,
                            'take_profit': latest['Close'] * 1.01,
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
                    elif (latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80 and 
                          latest['Williams_R'] > -20 and latest['Stoch_K'] < latest['Stoch_D']):
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'Stoch_Williams_Overbought',
                            'strength': 0.6 * pattern_analysis.get('confidence', 0.5),
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 1.005,
                            'take_profit': latest['Close'] * 0.99,
                            'timestamp': datetime.now(),
                            'pattern_support': pattern_analysis.get('trend_prediction', {}).get('direction', 'unknown')
                        })
            
            # Strategy 5: AI Pattern-Based Signals
            if pattern_analysis.get('confidence', 0) > 0.7:
                trend_prediction = pattern_analysis.get('trend_prediction', {})
                if trend_prediction.get('direction') == 'bullish' and trend_prediction.get('confidence', 0) > 0.6:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'strategy': 'AI_Pattern_Analysis',
                        'strength': pattern_analysis.get('confidence', 0.5),
                        'entry_price': latest['Close'],
                        'stop_loss': latest['Close'] * 0.99,
                        'take_profit': latest['Close'] * 1.025,
                        'timestamp': datetime.now(),
                        'pattern_support': f"AI detected {pattern_analysis.get('short_trend', 'unknown')} trend"
                    })
                elif trend_prediction.get('direction') == 'bearish' and trend_prediction.get('confidence', 0) > 0.6:
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'strategy': 'AI_Pattern_Analysis',
                        'strength': pattern_analysis.get('confidence', 0.5),
                        'entry_price': latest['Close'],
                        'stop_loss': latest['Close'] * 1.01,
                        'take_profit': latest['Close'] * 0.975,
                        'timestamp': datetime.now(),
                        'pattern_support': f"AI detected {pattern_analysis.get('short_trend', 'unknown')} trend"
                    })
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals for {symbol}: {e}")
            return []
    
    def generate_binary_signals(self, symbol: str, data: pd.DataFrame):
        """Generate ultra-fast binary options signals for OTC markets"""
        try:
            if data.empty or len(data) < 20:
                return []
                
            binary_signals = []
            
            # Calculate fast indicators for binary options
            # Fast RSI (5 periods)
            rsi_fast = ta.momentum.rsi(data['Close'], window=5)
            current_rsi_fast = rsi_fast.iloc[-1]
            
            # Fast Stochastic (5 periods)
            stoch_fast = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=5)
            current_stoch_fast = stoch_fast.iloc[-1]
            
            # Fast Williams %R (5 periods)
            williams_fast = ta.momentum.williams_r(data['High'], data['Low'], data['Close'], lbp=5)
            current_williams_fast = williams_fast.iloc[-1]
            
            # Fast CCI (5 periods)
            cci_fast = ta.trend.cci(data['High'], data['Low'], data['Close'], window=5)
            current_cci_fast = cci_fast.iloc[-1]
            
            # Price velocity (short-term momentum)
            price_velocity = data['Close'].pct_change().rolling(window=3).mean()
            current_price_velocity = price_velocity.iloc[-1]
            
            # Calculate signal strength
            signal_strength = (
                abs(current_rsi_fast - 50) / 50 +
                abs(current_stoch_fast - 50) / 50 +
                abs(current_cci_fast) / 100
            ) / 3
            
            # CALL signals (UP direction)
            call_conditions = []
            
            # Oversold bounce condition
            call_conditions.append(
                current_rsi_fast < 30 and
                current_stoch_fast < 20 and
                current_williams_fast < -80
            )
            
            # Momentum breakout condition
            call_conditions.append(
                current_price_velocity > 0.001 and
                current_cci_fast > 0 and
                current_williams_fast > -80
            )
            
            # Strong momentum condition
            call_conditions.append(
                current_rsi_fast > 50 and
                current_stoch_fast > 50 and
                current_price_velocity > 0.0005
            )
            
            # Generate CALL signal if any condition is met
            if any(call_conditions):
                binary_signals.append({
                    'symbol': symbol,
                    'type': 'CALL',
                    'strategy': 'Binary_Options_Ultra_Fast',
                    'strength': float(signal_strength),
                    'entry_price': float(data['Close'].iloc[-1]),
                    'expiry_time': '15s',
                    'market_type': 'OTC',
                    'timeframe': '1m',
                    'timestamp': datetime.now().isoformat(),
                    'indicators': {
                        'rsi_fast': float(current_rsi_fast),
                        'stoch_fast': float(current_stoch_fast),
                        'cci_fast': float(current_cci_fast),
                        'williams_fast': float(current_williams_fast),
                        'price_velocity': float(current_price_velocity)
                    }
                })
                
            # PUT signals (DOWN direction)
            put_conditions = []
            
            # Overbought reversal condition
            put_conditions.append(
                current_rsi_fast > 70 and
                current_stoch_fast > 80 and
                current_williams_fast > -20
            )
            
            # Momentum breakdown condition
            put_conditions.append(
                current_price_velocity < -0.001 and
                current_cci_fast < 0 and
                current_williams_fast < -20
            )
            
            # Strong bearish momentum condition
            put_conditions.append(
                current_rsi_fast < 50 and
                current_stoch_fast < 50 and
                current_price_velocity < -0.0005
            )
            
            # Generate PUT signal if any condition is met
            if any(put_conditions):
                binary_signals.append({
                    'symbol': symbol,
                    'type': 'PUT',
                    'strategy': 'Binary_Options_Ultra_Fast',
                    'strength': float(signal_strength),
                    'entry_price': float(data['Close'].iloc[-1]),
                    'expiry_time': '15s',
                    'market_type': 'OTC',
                    'timeframe': '1m',
                    'timestamp': datetime.now().isoformat(),
                    'indicators': {
                        'rsi_fast': float(current_rsi_fast),
                        'stoch_fast': float(current_stoch_fast),
                        'cci_fast': float(current_cci_fast),
                        'williams_fast': float(current_williams_fast),
                        'price_velocity': float(current_price_velocity)
                    }
                })
                
            # Generate alternative expiry times for different market conditions
            if binary_signals:
                # Add 5-second signals for highly volatile conditions
                if abs(current_price_velocity) > 0.002:
                    ultra_fast_signal = binary_signals[-1].copy()
                    ultra_fast_signal['expiry_time'] = '5s'
                    ultra_fast_signal['strategy'] = 'Binary_Options_Ultra_Fast_5s'
                    binary_signals.append(ultra_fast_signal)
                    
                # Add 30-second signals for trending markets
                if current_rsi_fast > 60 or current_rsi_fast < 40:
                    trend_signal = binary_signals[0].copy()
                    trend_signal['expiry_time'] = '30s'
                    trend_signal['strategy'] = 'Binary_Options_Trend_30s'
                    binary_signals.append(trend_signal)
                    
                # Add 1-minute signals for stable trends
                if signal_strength > 0.7:
                    stable_signal = binary_signals[0].copy()
                    stable_signal['expiry_time'] = '1m'
                    stable_signal['strategy'] = 'Binary_Options_Stable_1m'
                    binary_signals.append(stable_signal)
                    
            return binary_signals
            
        except Exception as e:
            print(f"Error generating binary signals for {symbol}: {e}")
            return []
        """Generate high-frequency binary trading signals"""
        if data.empty or len(data) < 10:
            return []
            
        binary_signals = []
        latest = data.iloc[-1]
        
        try:
            # High-frequency scalping signals
            if 'RSI' in data.columns and pd.notna(latest['RSI']):
                # Quick RSI bounce signals
                if latest['RSI'] < 25:
                    binary_signals.append({
                        'symbol': symbol,
                        'type': 'CALL',
                        'strategy': 'RSI_Bounce',
                        'expiry': '1m',
                        'strength': 0.7,
                        'entry_price': latest['Close'],
                        'timestamp': datetime.now()
                    })
                elif latest['RSI'] > 75:
                    binary_signals.append({
                        'symbol': symbol,
                        'type': 'PUT',
                        'strategy': 'RSI_Bounce',
                        'expiry': '1m',
                        'strength': 0.7,
                        'entry_price': latest['Close'],
                        'timestamp': datetime.now()
                    })
            
            # Bollinger Band squeeze signals
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns and 'BB_Middle' in data.columns:
                if (pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']) and 
                    pd.notna(latest['BB_Middle'])):
                    
                    bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / latest['BB_Middle']
                    if bb_width < 0.02:  # Tight squeeze
                        if latest['Close'] > latest['BB_Middle']:
                            binary_signals.append({
                                'symbol': symbol,
                                'type': 'CALL',
                                'strategy': 'BB_Squeeze_Breakout',
                                'expiry': '5m',
                                'strength': 0.6,
                                'entry_price': latest['Close'],
                                'timestamp': datetime.now()
                            })
                        else:
                            binary_signals.append({
                                'symbol': symbol,
                                'type': 'PUT',
                                'strategy': 'BB_Squeeze_Breakout',
                                'expiry': '5m',
                                'strength': 0.6,
                                'entry_price': latest['Close'],
                                'timestamp': datetime.now()
                            })
            
            return binary_signals
            
        except Exception as e:
            print(f"Error generating binary signals for {symbol}: {e}")
            return []
    
    def analyze_sentiment(self, symbol: str):
        """Analyze market sentiment for a currency pair using AI models"""
        try:
            if sentiment_analyzer is None:
                return {'sentiment': 'neutral', 'confidence': 0.5, 'sources': []}
            
            # Generate relevant news headlines for analysis
            # In a real implementation, you would fetch actual news from APIs
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            sample_news = [
                f"Central bank raises interest rates affecting {base_currency} strength",
                f"Economic indicators show positive outlook for {base_currency}",
                f"Market volatility increases in {base_currency}/{quote_currency} trading",
                f"Inflation data impacts {quote_currency} monetary policy decisions",
                f"Trade tensions affect {base_currency} and {quote_currency} exchange rates",
                f"GDP growth exceeds expectations in {base_currency} region",
                f"Employment data shows improvement in {quote_currency} economy"
            ]
            
            sentiments = []
            analyzed_sources = []
            
            for news in sample_news:
                try:
                    # Analyze sentiment using HuggingFace model
                    result = sentiment_analyzer(news)
                    
                    # Convert model output to standard format
                    if result[0]['label'] == 'LABEL_2':  # Positive
                        sentiment_label = 'positive'
                        confidence = result[0]['score']
                    elif result[0]['label'] == 'LABEL_0':  # Negative
                        sentiment_label = 'negative'
                        confidence = result[0]['score']
                    else:  # Neutral
                        sentiment_label = 'neutral'
                        confidence = result[0]['score']
                    
                    sentiments.append({
                        'label': sentiment_label,
                        'confidence': confidence
                    })
                    
                    analyzed_sources.append({
                        'text': news,
                        'sentiment': sentiment_label,
                        'confidence': confidence
                    })
                    
                except Exception as e:
                    print(f"Error analyzing individual news: {e}")
                    continue
            
            if not sentiments:
                return {'sentiment': 'neutral', 'confidence': 0.5, 'sources': []}
            
            # Calculate weighted average sentiment
            positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
            negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
            neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')
            
            total_count = len(sentiments)
            positive_weight = positive_count / total_count
            negative_weight = negative_count / total_count
            neutral_weight = neutral_count / total_count
            
            # Calculate average confidence
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            # Determine overall sentiment
            if positive_weight > negative_weight and positive_weight > neutral_weight:
                overall_sentiment = 'positive'
                confidence = positive_weight * avg_confidence
            elif negative_weight > positive_weight and negative_weight > neutral_weight:
                overall_sentiment = 'negative'
                confidence = negative_weight * avg_confidence
            else:
                overall_sentiment = 'neutral'
                confidence = neutral_weight * avg_confidence
            
            # Apply currency-specific adjustments
            if base_currency in ['USD', 'EUR', 'GBP']:  # Major currencies
                confidence *= 1.1  # Slightly higher confidence for major pairs
            
            return {
                'sentiment': overall_sentiment,
                'confidence': min(confidence, 1.0),
                'breakdown': {
                    'positive': positive_weight,
                    'negative': negative_weight,
                    'neutral': neutral_weight
                },
                'sources': analyzed_sources,
                'total_analyzed': total_count
            }
                
        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'sources': []}
    
    async def run_analysis(self, symbols: List[str] = None):
        """Run complete analysis for specified symbols"""
        if symbols is None:
            symbols = self.major_pairs[:5]  # Analyze top 5 pairs
        
        results = []
        
        for symbol in symbols:
            try:
                # Get forex data
                data = self.get_forex_data(symbol, '1h', 200)
                if data.empty:
                    continue
                
                # Calculate indicators
                data = self.calculate_all_indicators(data)
                
                # AI Pattern Analysis
                pattern_analysis = self.analyze_market_patterns(symbol, data)
                
                # Generate advanced signals using all strategies
                signals = self.generate_advanced_signals(symbol, data)
                binary_signals = self.generate_binary_signals(symbol, data)
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(symbol)
                
                # Current market data
                latest = data.iloc[-1]
                
                result = {
                    'symbol': symbol,
                    'current_price': float(latest['Close']),
                    'change': float(latest['Close'] - data.iloc[-2]['Close']),
                    'change_percent': float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100),
                    'indicators': {
                        'RSI': float(latest.get('RSI', 0)) if pd.notna(latest.get('RSI', 0)) else 0,
                        'MACD': float(latest.get('MACD', 0)) if pd.notna(latest.get('MACD', 0)) else 0,
                        'BB_Upper': float(latest.get('BB_Upper', 0)) if pd.notna(latest.get('BB_Upper', 0)) else 0,
                        'BB_Lower': float(latest.get('BB_Lower', 0)) if pd.notna(latest.get('BB_Lower', 0)) else 0,
                        'ADX': float(latest.get('ADX', 0)) if pd.notna(latest.get('ADX', 0)) else 0,
                        'Stoch_K': float(latest.get('Stoch_K', 0)) if pd.notna(latest.get('Stoch_K', 0)) else 0,
                        'Williams_R': float(latest.get('Williams_R', 0)) if pd.notna(latest.get('Williams_R', 0)) else 0,
                        'CCI': float(latest.get('CCI', 0)) if pd.notna(latest.get('CCI', 0)) else 0,
                        'ATR': float(latest.get('ATR', 0)) if pd.notna(latest.get('ATR', 0)) else 0,
                    },
                    'pattern_analysis': pattern_analysis,
                    'signals': signals,
                    'binary_signals': binary_signals,
                    'sentiment': sentiment,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return results

# Initialize the trading agent
trading_agent = ForexTradingAgent()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Models
class ForexAnalysisRequest(BaseModel):
    symbols: Optional[List[str]] = None
    timeframe: Optional[str] = "1h"

class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    type: str
    strategy: str
    strength: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Forex AI Trading Agent - Ultimate Market Analysis System"}

@api_router.get("/forex/pairs")
async def get_major_pairs():
    """Get list of major forex pairs"""
    return {"pairs": trading_agent.major_pairs}

@api_router.post("/forex/analyze")
async def analyze_forex_pairs(request: ForexAnalysisRequest):
    """Analyze forex pairs and generate signals"""
    try:
        results = await trading_agent.run_analysis(request.symbols)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.get("/forex/signals")
async def get_active_signals():
    """Get all active trading signals"""
    return {"signals": trading_agent.active_signals}

@api_router.get("/forex/performance")
async def get_performance_metrics():
    """Get performance metrics and strategy accuracy"""
    return {"metrics": trading_agent.performance_metrics}

@api_router.post("/forex/pattern-analysis")
async def get_pattern_analysis(request: ForexAnalysisRequest):
    """Get detailed AI pattern analysis for forex pairs"""
    try:
        if not request.symbols:
            symbols = trading_agent.major_pairs[:5]
        else:
            symbols = request.symbols
        
        results = []
        for symbol in symbols:
            data = trading_agent.get_forex_data(symbol, request.timeframe or '1h', 200)
            if not data.empty:
                data = trading_agent.calculate_all_indicators(data)
                pattern_analysis = trading_agent.analyze_market_patterns(symbol, data)
                results.append(pattern_analysis)
        
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.get("/forex/advanced-signals/{symbol}")
async def get_advanced_signals(symbol: str):
    """Get advanced trading signals for a specific symbol using all strategies"""
    try:
        data = trading_agent.get_forex_data(symbol, '1h', 200)
        if data.empty:
            return {"status": "error", "message": "No data available"}
        
        data = trading_agent.calculate_all_indicators(data)
        signals = trading_agent.generate_advanced_signals(symbol, data)
        
        return {
            "status": "success",
            "symbol": symbol,
            "signals": signals,
            "total_signals": len(signals),
            "strategies_used": list(set([s['strategy'] for s in signals])),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.get("/forex/binary-signals/{symbol}")
async def get_binary_signals(symbol: str):
    """Get ultra-fast binary options signals for OTC markets"""
    try:
        # Get 1-minute data for binary options
        data = trading_agent.get_forex_data(symbol, '1m', 100)
        if data.empty:
            return {"status": "error", "message": "No data available"}
        
        data = trading_agent.calculate_all_indicators(data)
        binary_signals = trading_agent.generate_binary_signals(symbol, data)
        
        return {
            "status": "success",
            "symbol": symbol,
            "binary_signals": binary_signals,
            "total_signals": len(binary_signals),
            "market_type": "OTC",
            "expiry_options": ["5s", "15s", "30s"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.get("/forex/strategy-performance")
async def get_strategy_performance():
    """Get performance metrics for all trading strategies"""
    try:
        performance = {}
        
        for strategy, metrics in trading_agent.strategy_performance.items():
            if metrics['total_trades'] > 0:
                win_rate = metrics['wins'] / metrics['total_trades'] * 100
                performance[strategy] = {
                    'win_rate': win_rate,
                    'total_trades': metrics['total_trades'],
                    'wins': metrics['wins'],
                    'losses': metrics['losses'],
                    'status': 'excellent' if win_rate >= 80 else 'good' if win_rate >= 70 else 'needs_improvement'
                }
            else:
                performance[strategy] = {
                    'win_rate': 0,
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'status': 'no_data'
                }
        
        return {
            "status": "success",
            "strategy_performance": performance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.post("/forex/backtest")
async def backtest_strategies(request: ForexAnalysisRequest):
    """Run backtesting for trading strategies"""
    try:
        if not request.symbols:
            symbols = trading_agent.major_pairs[:3]
        else:
            symbols = request.symbols
            
        backtest_results = []
        
        for symbol in symbols:
            # Get historical data for backtesting
            data = trading_agent.get_forex_data(symbol, request.timeframe or '1h', 1000)
            if data.empty:
                continue
                
            data = trading_agent.calculate_all_indicators(data)
            
            # Run backtesting simulation
            backtest_result = await simulate_backtest(symbol, data)
            backtest_results.append(backtest_result)
        
        return {
            "status": "success",
            "backtest_results": backtest_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.get("/forex/market-overview")
async def get_market_overview():
    """Get comprehensive market overview with all analysis"""
    try:
        symbols = trading_agent.major_pairs[:8]
        market_overview = {
            'total_pairs_analyzed': len(symbols),
            'strong_buy_signals': 0,
            'strong_sell_signals': 0,
            'binary_opportunities': 0,
            'market_sentiment': 'neutral',
            'top_opportunities': [],
            'risk_level': 'medium',
            'pairs_analysis': []
        }
        
        for symbol in symbols:
            data = trading_agent.get_forex_data(symbol, '1h', 200)
            if data.empty:
                continue
                
            data = trading_agent.calculate_all_indicators(data)
            
            # Get all signals
            signals = trading_agent.generate_advanced_signals(symbol, data)
            binary_signals = trading_agent.generate_binary_signals(symbol, data)
            sentiment = trading_agent.analyze_sentiment(symbol)
            
            # Count signal types
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            if buy_signals:
                market_overview['strong_buy_signals'] += len(buy_signals)
            if sell_signals:
                market_overview['strong_sell_signals'] += len(sell_signals)
            if binary_signals:
                market_overview['binary_opportunities'] += len(binary_signals)
            
            # Add to top opportunities if high-strength signals
            high_strength_signals = [s for s in signals if s['strength'] >= 0.8]
            if high_strength_signals:
                market_overview['top_opportunities'].extend(high_strength_signals)
                
            # Pair analysis
            latest = data.iloc[-1]
            pair_analysis = {
                'symbol': symbol,
                'current_price': float(latest['Close']),
                'change_percent': float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100),
                'signals_count': len(signals),
                'binary_signals_count': len(binary_signals),
                'sentiment': sentiment['sentiment'],
                'trend': 'bullish' if buy_signals else 'bearish' if sell_signals else 'neutral',
                'volatility': float(latest.get('ATR', 0)),
                'rsi': float(latest.get('RSI', 50))
            }
            market_overview['pairs_analysis'].append(pair_analysis)
        
        # Calculate overall market sentiment
        total_signals = market_overview['strong_buy_signals'] + market_overview['strong_sell_signals']
        if total_signals > 0:
            buy_ratio = market_overview['strong_buy_signals'] / total_signals
            if buy_ratio > 0.6:
                market_overview['market_sentiment'] = 'bullish'
            elif buy_ratio < 0.4:
                market_overview['market_sentiment'] = 'bearish'
                
        # Sort top opportunities by strength
        market_overview['top_opportunities'] = sorted(
            market_overview['top_opportunities'], 
            key=lambda x: x['strength'], 
            reverse=True
        )[:10]
        
        return {
            "status": "success",
            "market_overview": market_overview,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def simulate_backtest(symbol: str, data: pd.DataFrame):
    """Simulate backtesting for a trading strategy"""
    try:
        # Simple backtesting simulation
        trades = []
        balance = 10000  # Starting balance
        
        for i in range(100, len(data) - 1):
            window_data = data.iloc[i-100:i+1]
            signals = trading_agent.generate_advanced_signals(symbol, window_data)
            
            if signals:
                signal = signals[0]  # Take first signal
                entry_price = signal['entry_price']
                
                # Simulate trade execution
                if signal['type'] == 'BUY':
                    exit_price = data.iloc[i+1]['Close']
                    profit = (exit_price - entry_price) / entry_price * balance * 0.1  # 10% of balance
                else:
                    exit_price = data.iloc[i+1]['Close']
                    profit = (entry_price - exit_price) / entry_price * balance * 0.1
                
                balance += profit
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': float(exit_price),
                    'profit': profit,
                    'type': signal['type'],
                    'strategy': signal['strategy']
                })
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        total_profit = sum(t['profit'] for t in trades)
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_profit': total_profit,
            'roi': ((balance - 10000) / 10000 * 100),
            'final_balance': balance
        }
        
    except Exception as e:
        print(f"Error in backtesting for {symbol}: {e}")
        return {
            'symbol': symbol,
            'error': str(e)
        }
    """Get comprehensive AI insights for a specific currency pair"""
    try:
        data = trading_agent.get_forex_data(symbol, '1h', 200)
        if data.empty:
            return {"status": "error", "message": "No data available"}
        
        data = trading_agent.calculate_all_indicators(data)
        
        # Multi-timeframe analysis
        insights = {}
        timeframes = ['1h', '4h', '1d']
        
        for tf in timeframes:
            tf_data = trading_agent.get_forex_data(symbol, tf, 100)
            if not tf_data.empty:
                tf_data = trading_agent.calculate_all_indicators(tf_data)
                insights[tf] = {
                    'pattern_analysis': trading_agent.analyze_market_patterns(symbol, tf_data),
                    'signals': trading_agent.generate_signals(symbol, tf_data),
                    'binary_signals': trading_agent.generate_binary_signals(symbol, tf_data)
                }
        
        sentiment = trading_agent.analyze_sentiment(symbol)
        
        return {
            "status": "success",
            "symbol": symbol,
            "multi_timeframe_analysis": insights,
            "sentiment_analysis": sentiment,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trading updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Run analysis every 30 seconds
            results = await trading_agent.run_analysis()
            await manager.send_personal_message(json.dumps({
                "type": "analysis_update",
                "data": results
            }), websocket)
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Background task for continuous analysis
async def continuous_analysis():
    while True:
        try:
            results = await trading_agent.run_analysis()
            await manager.broadcast(json.dumps({
                "type": "market_update",
                "data": results,
                "timestamp": datetime.now().isoformat()
            }))
            await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            logger.error(f"Error in continuous analysis: {e}")
            await asyncio.sleep(30)

# Start background task
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Forex AI Trading Agent Starting...")
    logger.info("📊 Loading AI models and initializing trading engine...")
    # Start continuous analysis in background
    asyncio.create_task(continuous_analysis())
    logger.info("✅ Trading agent is ready!")
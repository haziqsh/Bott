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
import warnings
warnings.filterwarnings('ignore')

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
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
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
        """Generate trading signals using multiple strategies"""
        if data.empty or len(data) < 50:
            return []
            
        signals = []
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
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
                            'strength': 0.8,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 0.995,
                            'take_profit': latest['Close'] * 1.015,
                            'timestamp': datetime.now()
                        })
                    elif latest['RSI'] > 70 and latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'RSI_MACD_Confluence',
                            'strength': 0.8,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 1.005,
                            'take_profit': latest['Close'] * 0.985,
                            'timestamp': datetime.now()
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
                            'strength': 0.75,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['BB_Lower'] * 0.995,
                            'take_profit': latest['BB_Middle'],
                            'timestamp': datetime.now()
                        })
                    elif latest['Close'] >= latest['BB_Upper'] and latest['RSI'] > 70:
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'BB_RSI_Overbought',
                            'strength': 0.75,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['BB_Upper'] * 1.005,
                            'take_profit': latest['BB_Middle'],
                            'timestamp': datetime.now()
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
                            'strength': 0.7,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['EMA_20'] * 0.995,
                            'take_profit': latest['Close'] * 1.02,
                            'timestamp': datetime.now()
                        })
                    elif (latest['EMA_10'] < latest['EMA_20'] and prev['EMA_10'] >= prev['EMA_20'] and 
                          latest['ADX'] > 25):
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'EMA_Cross_ADX',
                            'strength': 0.7,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['EMA_20'] * 1.005,
                            'take_profit': latest['Close'] * 0.98,
                            'timestamp': datetime.now()
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
                            'strength': 0.6,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 0.995,
                            'take_profit': latest['Close'] * 1.01,
                            'timestamp': datetime.now()
                        })
                    elif (latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80 and 
                          latest['Williams_R'] > -20 and latest['Stoch_K'] < latest['Stoch_D']):
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'strategy': 'Stoch_Williams_Overbought',
                            'strength': 0.6,
                            'entry_price': latest['Close'],
                            'stop_loss': latest['Close'] * 1.005,
                            'take_profit': latest['Close'] * 0.99,
                            'timestamp': datetime.now()
                        })
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals for {symbol}: {e}")
            return []
    
    def generate_binary_signals(self, symbol: str, data: pd.DataFrame):
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
        """Analyze market sentiment for a currency pair"""
        try:
            if sentiment_analyzer is None:
                return {'sentiment': 'neutral', 'confidence': 0.5}
            
            # Simulate news sentiment analysis
            # In a real implementation, you would fetch news related to the currency pair
            sample_news = [
                f"Economic indicators show positive outlook for {symbol[:3]}",
                f"Central bank policy supports {symbol[3:]} strength",
                f"Market volatility affects {symbol} trading sentiment"
            ]
            
            sentiments = []
            for news in sample_news:
                result = sentiment_analyzer(news)
                sentiments.append(result[0])
            
            # Calculate average sentiment
            positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
            negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
            
            if positive_count > negative_count:
                return {'sentiment': 'positive', 'confidence': 0.7}
            elif negative_count > positive_count:
                return {'sentiment': 'negative', 'confidence': 0.7}
            else:
                return {'sentiment': 'neutral', 'confidence': 0.5}
                
        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
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
                
                # Generate signals
                signals = self.generate_signals(symbol, data)
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
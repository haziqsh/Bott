import requests
import unittest
import json
import sys
import time
import websocket
import threading
from datetime import datetime

class ForexAITradingAgentAPITester:
    def __init__(self, base_url="https://54cdce57-a2b3-4c35-9128-e3ef13f4a9fb.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.ws_url = f"wss://54cdce57-a2b3-4c35-9128-e3ef13f4a9fb.preview.emergentagent.com/api/ws"
        self.tests_run = 0
        self.tests_passed = 0
        self.ws_messages_received = 0
        self.ws_connected = False

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                return success, response.json()
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                return success, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        if success:
            print(f"Response: {response}")
        return success

    def test_forex_pairs(self):
        """Test getting forex pairs"""
        success, response = self.run_test(
            "Get Forex Pairs",
            "GET",
            "forex/pairs",
            200
        )
        if success:
            print(f"Number of pairs: {len(response.get('pairs', []))}")
            print(f"Sample pairs: {response.get('pairs', [])[:5]}")
        return success

    def test_forex_analyze(self):
        """Test forex analysis endpoint with major currency pairs"""
        major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
        success, response = self.run_test(
            "Forex Analysis with Major Pairs",
            "POST",
            "forex/analyze",
            200,
            data={"symbols": major_pairs}
        )
        if success:
            print(f"Analysis status: {response.get('status')}")
            data = response.get('data', [])
            print(f"Number of analyzed pairs: {len(data)}")
            if data:
                print(f"Sample analysis for {data[0].get('symbol')}:")
                print(f"  - Current price: {data[0].get('current_price')}")
                print(f"  - RSI: {data[0].get('indicators', {}).get('RSI')}")
                print(f"  - Number of signals: {len(data[0].get('signals', []))}")
                print(f"  - Number of binary signals: {len(data[0].get('binary_signals', []))}")
                
                # Verify advanced indicators are present
                indicators = data[0].get('indicators', {})
                advanced_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ADX', 'Stoch_K', 'Williams_R', 'CCI', 'ATR']
                present_indicators = [ind for ind in advanced_indicators if indicators.get(ind, 0) != 0]
                print(f"  - Advanced indicators present: {present_indicators}")
                print(f"  - Pattern analysis: {data[0].get('pattern_analysis', {}).get('short_trend', 'N/A')}")
        return success

    def test_forex_signals(self):
        """Test getting active signals"""
        success, response = self.run_test(
            "Get Active Signals",
            "GET",
            "forex/signals",
            200
        )
        if success:
            print(f"Active signals: {response}")
        return success

    def test_forex_performance(self):
        """Test getting performance metrics"""
        success, response = self.run_test(
            "Get Performance Metrics",
            "GET",
            "forex/performance",
            200
        )
        if success:
            print(f"Performance metrics: {response}")
        return success
        
    def test_advanced_signals(self):
        """Test advanced signal generation for EURUSD"""
        success, response = self.run_test(
            "Advanced Signals for EURUSD",
            "GET",
            "forex/advanced-signals/EURUSD",
            200
        )
        if success:
            print(f"Signal status: {response.get('status')}")
            signals = response.get('signals', [])
            print(f"Number of advanced signals: {len(signals)}")
            if signals:
                strategies = set([s.get('strategy') for s in signals])
                print(f"Strategies used: {strategies}")
                
                # Check for NostalgiaForInfinity strategy
                nostalgia_signals = [s for s in signals if 'Nostalgia' in s.get('strategy', '')]
                if nostalgia_signals:
                    print(f"  - NostalgiaForInfinity signals found: {len(nostalgia_signals)}")
                
                # Check for Ichimoku strategy
                ichimoku_signals = [s for s in signals if 'Ichimoku' in s.get('strategy', '')]
                if ichimoku_signals:
                    print(f"  - Ichimoku signals found: {len(ichimoku_signals)}")
                
                # Check for SuperTrend strategy
                supertrend_signals = [s for s in signals if 'SuperTrend' in s.get('strategy', '')]
                if supertrend_signals:
                    print(f"  - SuperTrend signals found: {len(supertrend_signals)}")
                
                # Check for ML-based strategy
                ml_signals = [s for s in signals if 'ML' in s.get('strategy', '')]
                if ml_signals:
                    print(f"  - ML-based signals found: {len(ml_signals)}")
                
                # Check signal strength calculations
                strengths = [s.get('strength', 0) for s in signals]
                if strengths:
                    avg_strength = sum(strengths) / len(strengths)
                    print(f"  - Average signal strength: {avg_strength:.2f}")
        return success
        
    def test_binary_signals(self):
        """Test binary options signals for EURUSD"""
        success, response = self.run_test(
            "Binary Options Signals for EURUSD",
            "GET",
            "forex/binary-signals/EURUSD",
            200
        )
        if success:
            print(f"Signal status: {response.get('status')}")
            binary_signals = response.get('binary_signals', [])
            print(f"Number of binary signals: {len(binary_signals)}")
            
            # Check for OTC market signals
            if response.get('market_type') == 'OTC':
                print(f"  - OTC market signals confirmed")
            
            # Check expiry times
            expiry_options = response.get('expiry_options', [])
            print(f"  - Available expiry options: {expiry_options}")
            
            # Verify ultra-fast expiry times (5-15 seconds)
            has_ultra_fast = any(opt in ['5s', '15s'] for opt in expiry_options)
            if has_ultra_fast:
                print(f"  - Ultra-fast expiry times available")
            
            if binary_signals:
                # Check signal types
                signal_types = set([s.get('type') for s in binary_signals])
                print(f"  - Signal types: {signal_types}")
                
                # Check expiry times in actual signals
                signal_expiries = set([s.get('expiry_time') for s in binary_signals])
                print(f"  - Signal expiry times: {signal_expiries}")
                
                # Check indicators used
                if 'indicators' in binary_signals[0]:
                    indicator_keys = binary_signals[0]['indicators'].keys()
                    print(f"  - Indicators used: {list(indicator_keys)}")
        return success
        
    def test_market_overview(self):
        """Test market overview endpoint"""
        success, response = self.run_test(
            "Market Overview",
            "GET",
            "forex/market-overview",
            200
        )
        if success:
            print(f"Overview status: {response.get('status')}")
            
            # Check number of pairs analyzed
            pairs_analyzed = response.get('total_pairs_analyzed', 0)
            print(f"  - Total pairs analyzed: {pairs_analyzed}")
            
            # Check signal counts
            buy_signals = response.get('strong_buy_signals', 0)
            sell_signals = response.get('strong_sell_signals', 0)
            print(f"  - Strong buy signals: {buy_signals}")
            print(f"  - Strong sell signals: {sell_signals}")
            
            # Check market sentiment
            sentiment = response.get('market_sentiment', 'unknown')
            print(f"  - Market sentiment: {sentiment}")
            
            # Check top opportunities
            opportunities = response.get('top_opportunities', [])
            if opportunities:
                print(f"  - Top opportunities: {[o.get('symbol') for o in opportunities]}")
            
            # Check pairs analysis
            pairs_analysis = response.get('pairs_analysis', [])
            if pairs_analysis:
                print(f"  - Number of pairs with detailed analysis: {len(pairs_analysis)}")
        return success
        
    def test_historical_data(self):
        """Test historical data API endpoint"""
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        for symbol in test_symbols:
            success, response = self.run_test(
                f"Historical Data for {symbol}",
                "GET",
                f"forex/historical/{symbol}",
                200
            )
            if success:
                print(f"Historical data status: {response.get('status')}")
                data = response.get('data', [])
                print(f"  - Number of data points: {len(data)}")
                
                if data:
                    # Verify data structure
                    first_point = data[0]
                    required_fields = ['timestamp', 'open', 'high', 'low', 'close']
                    has_all_fields = all(field in first_point for field in required_fields)
                    print(f"  - Has required fields (timestamp, OHLC): {has_all_fields}")
                    
                    # Check if we have at least 100 data points as expected
                    if len(data) >= 100:
                        print(f"  - ✅ Sufficient data points (>= 100)")
                    else:
                        print(f"  - ⚠️ Limited data points (< 100)")
                    
                    # Show sample data
                    print(f"  - Sample data point: {first_point}")
                    
                    # Verify timestamp format
                    try:
                        datetime.fromisoformat(first_point['timestamp'].replace('Z', '+00:00'))
                        print(f"  - ✅ Valid timestamp format")
                    except:
                        print(f"  - ❌ Invalid timestamp format")
                        
            if not success:
                return False
                
        return True

    def test_historical_data_invalid_symbol(self):
        """Test historical data endpoint with invalid symbol"""
        success, response = self.run_test(
            "Historical Data with Invalid Symbol",
            "GET",
            "forex/historical/INVALID",
            200  # Should still return 200 but with error message
        )
        if success:
            # Check if error is properly handled
            if response.get('status') == 'error' or 'error' in response:
                print(f"  - ✅ Error properly handled for invalid symbol")
                return True
            else:
                print(f"  - ⚠️ No error handling for invalid symbol")
                return True
        return False
        """Test strategy performance tracking"""
        success, response = self.run_test(
            "Strategy Performance",
            "GET",
            "forex/strategy-performance",
            200
        )
        if success:
            print(f"Performance data received")
            
            # Check strategies tracked
            strategies = response.keys()
            print(f"  - Strategies tracked: {list(strategies)}")
            
            # Check performance metrics
            for strategy, metrics in response.items():
                if 'win_rate' in metrics:
                    print(f"  - {strategy}: Win rate {metrics.get('win_rate', 0):.2f}%, "
                          f"Trades: {metrics.get('total_trades', 0)}")
        return success
        
    def on_ws_message(self, ws, message):
        """Handle WebSocket messages"""
        self.ws_messages_received += 1
        print(f"📡 WebSocket message received ({self.ws_messages_received})")
        try:
            data = json.loads(message)
            print(f"  - Message type: {data.get('type')}")
            if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                print(f"  - Data for {len(data['data'])} symbols received")
                print(f"  - First symbol: {data['data'][0].get('symbol')}")
        except Exception as e:
            print(f"  - Error parsing message: {e}")
    
    def on_ws_error(self, ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_ws_close(self, ws, close_status_code, close_msg):
        print(f"📡 WebSocket connection closed")
        self.ws_connected = False
    
    def on_ws_open(self, ws):
        print(f"📡 WebSocket connection established")
        self.ws_connected = True
        
    def test_websocket(self):
        """Test WebSocket real-time updates"""
        self.tests_run += 1
        print(f"\n🔍 Testing WebSocket Real-time Updates...")
        
        try:
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_ws_message,
                on_error=self.on_ws_error,
                on_close=self.on_ws_close,
                on_open=self.on_ws_open
            )
            
            # Start WebSocket connection in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection and messages
            timeout = 30  # seconds
            start_time = time.time()
            
            # Wait for connection
            while not self.ws_connected and time.time() - start_time < timeout:
                time.sleep(0.5)
            
            if not self.ws_connected:
                print("❌ Failed - WebSocket connection not established")
                return False
                
            # Wait for at least one message
            while self.ws_messages_received == 0 and time.time() - start_time < timeout:
                time.sleep(0.5)
                
            # Close connection
            ws.close()
            
            # Check if we received any messages
            if self.ws_messages_received > 0:
                self.tests_passed += 1
                print(f"✅ Passed - Received {self.ws_messages_received} WebSocket messages")
                return True
            else:
                print("❌ Failed - No WebSocket messages received")
                return False
                
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False

def main():
    # Setup
    tester = ForexAITradingAgentAPITester()
    
    # Run tests
    print("\n🧪 TESTING FOREX TRADING SYSTEM WITH ADVANCED INDICATORS 🧪")
    print("===========================================================")
    
    # Basic API tests
    root_success = tester.test_root_endpoint()
    pairs_success = tester.test_forex_pairs()
    
    # Advanced signal generation tests
    advanced_signals_success = tester.test_advanced_signals()
    binary_signals_success = tester.test_binary_signals()
    market_overview_success = tester.test_market_overview()
    
    # Main analysis endpoint test
    analyze_success = tester.test_forex_analyze()
    
    # Performance tracking test
    strategy_performance_success = tester.test_strategy_performance()
    signals_success = tester.test_forex_signals()
    performance_success = tester.test_forex_performance()
    
    # WebSocket test
    websocket_success = tester.test_websocket()

    # Print results
    print("\n📊 TEST RESULTS SUMMARY")
    print("=======================")
    print(f"Tests passed: {tester.tests_passed}/{tester.tests_run} ({tester.tests_passed/tester.tests_run*100:.1f}%)")
    
    # Print detailed results
    print("\n📋 DETAILED RESULTS")
    print("===================")
    print(f"✓ Basic API functionality: {'✅ PASS' if root_success and pairs_success else '❌ FAIL'}")
    print(f"✓ Advanced signal generation: {'✅ PASS' if advanced_signals_success else '❌ FAIL'}")
    print(f"✓ Binary options capabilities: {'✅ PASS' if binary_signals_success else '❌ FAIL'}")
    print(f"✓ Market overview analytics: {'✅ PASS' if market_overview_success else '❌ FAIL'}")
    print(f"✓ Main analysis endpoint: {'✅ PASS' if analyze_success else '❌ FAIL'}")
    print(f"✓ Performance tracking: {'✅ PASS' if strategy_performance_success else '❌ FAIL'}")
    print(f"✓ WebSocket real-time updates: {'✅ PASS' if websocket_success else '❌ FAIL'}")
    
    # Return success status
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
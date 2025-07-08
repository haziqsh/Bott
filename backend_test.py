import requests
import unittest
import json
import sys
from datetime import datetime

class ForexAITradingAgentAPITester:
    def __init__(self, base_url="https://d3fc7e76-e1ee-4cb1-96f0-f0edda3b2227.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

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
        """Test forex analysis endpoint"""
        success, response = self.run_test(
            "Forex Analysis",
            "POST",
            "forex/analyze",
            200,
            data={"symbols": ["EURUSD", "GBPUSD", "USDJPY"]}
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

def main():
    # Setup
    tester = ForexAITradingAgentAPITester()
    
    # Run tests
    root_success = tester.test_root_endpoint()
    pairs_success = tester.test_forex_pairs()
    analyze_success = tester.test_forex_analyze()
    signals_success = tester.test_forex_signals()
    performance_success = tester.test_forex_performance()

    # Print results
    print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
    
    # Return success status
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
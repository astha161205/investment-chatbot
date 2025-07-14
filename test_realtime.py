#!/usr/bin/env python3
"""
Test script for real-time stock price functionality
This script tests the new real-time endpoints and features.
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_realtime_endpoint():
    """Test the real-time stock endpoint"""
    print("🧪 Testing Real-Time Stock Endpoint")
    print("=" * 50)
    
    test_stocks = ["AAPL", "MSFT", "TSLA", "RELIANCE.NS", "TCS.NS"]
    
    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")
        try:
            response = requests.get(f"{BASE_URL}/stock/realtime?ticker={ticker}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {ticker}: ${data['current_price']:.2f}")
                print(f"   Timestamp: {data['timestamp']}")
                print(f"   Data Age: {data['data_age_seconds']:.1f} seconds")
            elif response.status_code == 429:
                print(f"⚠️  {ticker}: Rate limited")
            else:
                print(f"❌ {ticker}: Error {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ {ticker}: Exception - {str(e)}")
        
        time.sleep(1)  # Small delay between requests

def test_batch_endpoint():
    """Test the batch stock endpoint"""
    print("\n🧪 Testing Batch Stock Endpoint")
    print("=" * 50)
    
    test_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    try:
        response = requests.post(f"{BASE_URL}/stocks/batch", 
                               json={'tickers': test_stocks, 'real_time': True})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch request successful")
            print(f"   Results: {len(data['results'])} stocks")
            print(f"   Errors: {len(data['errors'])} errors")
            
            for ticker, info in data['results'].items():
                print(f"   {ticker}: ${info['price']:.2f} ({info['data_age_seconds']:.1f}s old)")
            
            if data['errors']:
                print(f"   Errors: {data['errors']}")
        else:
            print(f"❌ Batch request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch request exception: {str(e)}")

def test_price_alerts():
    """Test price alert functionality"""
    print("\n🧪 Testing Price Alert Functionality")
    print("=" * 50)
    
    # Set a price alert
    alert_data = {
        'ticker': 'AAPL',
        'threshold': 150.0,
        'user_id': 'test_user'
    }
    
    try:
        response = requests.post(f"{BASE_URL}/alerts/set", json=alert_data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Alert set successfully")
            print(f"   Alert ID: {data['alert_id']}")
            print(f"   Message: {data['message']}")
        else:
            print(f"❌ Failed to set alert: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Alert setting exception: {str(e)}")
    
    # List alerts
    try:
        response = requests.get(f"{BASE_URL}/alerts/list?user_id=test_user")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Alerts retrieved successfully")
            print(f"   Count: {data['count']} alerts")
            
            for alert in data['alerts']:
                print(f"   {alert['ticker']}: ${alert['threshold']} (current: {alert['current_price']})")
        else:
            print(f"❌ Failed to get alerts: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Alert listing exception: {str(e)}")

def test_cache_comparison():
    """Compare cached vs real-time data"""
    print("\n🧪 Testing Cache vs Real-Time Comparison")
    print("=" * 50)
    
    ticker = "AAPL"
    
    # Get cached data
    print(f"Getting cached data for {ticker}...")
    try:
        cached_response = requests.get(f"{BASE_URL}/stock?ticker={ticker}")
        if cached_response.status_code == 200:
            cached_data = cached_response.json()
            print(f"✅ Cached data retrieved")
        else:
            print(f"❌ Cached data failed: {cached_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cached data exception: {str(e)}")
        return
    
    # Get real-time data
    print(f"Getting real-time data for {ticker}...")
    try:
        realtime_response = requests.get(f"{BASE_URL}/stock/realtime?ticker={ticker}")
        if realtime_response.status_code == 200:
            realtime_data = realtime_response.json()
            print(f"✅ Real-time data retrieved")
        else:
            print(f"❌ Real-time data failed: {realtime_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Real-time data exception: {str(e)}")
        return
    
    # Compare data ages
    if 'data_age_seconds' in realtime_data:
        print(f"   Real-time data age: {realtime_data['data_age_seconds']:.1f} seconds")
    
    # Check if data is actually different
    if 'current_price' in realtime_data:
        print(f"   Current price: ${realtime_data['current_price']:.2f}")

def test_rate_limit_status():
    """Test rate limit status endpoint"""
    print("\n🧪 Testing Rate Limit Status")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/rate-limit-status")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Rate limit status retrieved")
            print(f"   Current Delay: {data['rate_limiter']['current_delay']}s")
            print(f"   Requests in Window: {data['rate_limiter']['requests_in_window']}/{data['rate_limiter']['max_requests_per_window']}")
            print(f"   Total Cached: {data['cache_stats']['total_cached']}")
            print(f"   With Errors: {data['cache_stats']['cached_with_errors']}")
            print(f"   In Backoff: {data['cache_stats']['in_backoff']}")
        else:
            print(f"❌ Rate limit status failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Rate limit status exception: {str(e)}")

def main():
    """Main test function"""
    print("🧪 Real-Time Stock Price Test Suite")
    print("Make sure your Flask app is running on http://localhost:5000")
    print("=" * 60)
    
    try:
        # Test basic connectivity
        print("Testing basic connectivity...")
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Flask app is running")
        else:
            print(f"❌ Flask app not responding: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Flask app: {str(e)}")
        return
    
    # Run tests
    test_realtime_endpoint()
    test_batch_endpoint()
    test_price_alerts()
    test_cache_comparison()
    test_rate_limit_status()
    
    print("\n" + "="*60)
    print("🎉 Real-time test suite completed!")
    print("Visit http://localhost:5000/dashboard for the real-time dashboard")
    print("Visit http://localhost:5000/monitor for rate limiter monitoring")

if __name__ == "__main__":
    main() 
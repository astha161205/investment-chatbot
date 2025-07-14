#!/usr/bin/env python3
"""
Test script for the rate limiting system
This script will help you test and verify that the rate limiting is working correctly.
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_single_request(ticker):
    """Test a single stock request"""
    print(f"Testing request for {ticker}...")
    try:
        response = requests.get(f"{BASE_URL}/stock?ticker={ticker}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {ticker}")
            return True
        elif response.status_code == 429:
            print(f"⚠️  Rate limited: {ticker}")
            return False
        else:
            print(f"❌ Error {response.status_code}: {ticker}")
            return False
    except Exception as e:
        print(f"❌ Exception: {ticker} - {str(e)}")
        return False

def test_rate_limit_status():
    """Test the rate limit status endpoint"""
    print("\n📊 Checking rate limit status...")
    try:
        response = requests.get(f"{BASE_URL}/rate-limit-status")
        if response.status_code == 200:
            data = response.json()
            print(f"Rate Limiter Status:")
            print(f"  - Current Delay: {data['rate_limiter']['current_delay']}s")
            print(f"  - Requests in Window: {data['rate_limiter']['requests_in_window']}/{data['rate_limiter']['max_requests_per_window']}")
            print(f"  - Window Remaining: {data['rate_limiter']['window_remaining']:.1f}s")
            print(f"Cache Status:")
            print(f"  - Total Cached: {data['cache_stats']['total_cached']}")
            print(f"  - With Errors: {data['cache_stats']['cached_with_errors']}")
            print(f"  - In Backoff: {data['cache_stats']['in_backoff']}")
            return data
        else:
            print(f"❌ Error getting status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Exception getting status: {str(e)}")
        return None

def test_multiple_requests(tickers, delay=0):
    """Test multiple requests with optional delay"""
    print(f"\n🚀 Testing {len(tickers)} requests with {delay}s delay...")
    results = []
    
    for i, ticker in enumerate(tickers):
        print(f"\nRequest {i+1}/{len(tickers)}: {ticker}")
        success = test_single_request(ticker)
        results.append((ticker, success))
        
        if delay > 0 and i < len(tickers) - 1:
            print(f"Waiting {delay}s...")
            time.sleep(delay)
    
    success_count = sum(1 for _, success in results if success)
    print(f"\n📈 Results: {success_count}/{len(tickers)} successful")
    return results

def test_rate_limiting():
    """Test the rate limiting behavior"""
    print("🧪 Testing Rate Limiting System")
    print("=" * 50)
    
    # Test popular stocks
    test_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"
    ]
    
    # Initial status
    test_rate_limit_status()
    
    # Test without delay (should trigger rate limiting)
    print("\n" + "="*50)
    print("TEST 1: Multiple requests without delay (should trigger rate limiting)")
    print("="*50)
    results1 = test_multiple_requests(test_stocks[:5], delay=0)
    
    # Check status after rapid requests
    time.sleep(2)
    test_rate_limit_status()
    
    # Test with delay (should work better)
    print("\n" + "="*50)
    print("TEST 2: Multiple requests with 2s delay (should work better)")
    print("="*50)
    results2 = test_multiple_requests(test_stocks[5:], delay=2)
    
    # Final status
    time.sleep(2)
    test_rate_limit_status()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Test 1 (no delay): {sum(1 for _, success in results1 if success)}/{len(results1)} successful")
    print(f"Test 2 (with delay): {sum(1 for _, success in results2 if success)}/{len(results2)} successful")

def test_reset_functionality():
    """Test the reset functionality"""
    print("\n" + "="*50)
    print("TESTING RESET FUNCTIONALITY")
    print("="*50)
    
    print("Resetting rate limiter and cache...")
    try:
        response = requests.post(f"{BASE_URL}/reset-rate-limiter")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Reset successful!")
                test_rate_limit_status()
            else:
                print(f"❌ Reset failed: {data.get('error')}")
        else:
            print(f"❌ Reset failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Exception during reset: {str(e)}")

def main():
    """Main test function"""
    print("🧪 Yahoo Finance Rate Limiting Test Suite")
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
    test_rate_limiting()
    test_reset_functionality()
    
    print("\n" + "="*60)
    print("🎉 Test suite completed!")
    print("Check http://localhost:5000/monitor for real-time monitoring")

if __name__ == "__main__":
    main() 
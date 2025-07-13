import yfinance as yf
import requests
import json
from flask import Flask, request, jsonify, send_from_directory
import os
import re
import logging
from datetime import datetime
import google.generativeai as genai  # Import Google Generative AI library
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Configure the Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')  # Use the Gemini 1.5 Pro model

# Track API availability
api_available = True  # Set to True to indicate API is available

# Simple in-memory cache for stock data
stock_cache = {}  # {symbol: (data, timestamp)}

# Function to fetch stock data with caching and error handling
def get_stock_data(ticker):
    now = time.time()
    cache_duration = 180  # Cache for 3 minutes
    if ticker in stock_cache:
        data, timestamp = stock_cache[ticker]
        if now - timestamp < cache_duration:
            return data
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        stock_cache[ticker] = (data, now)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        # Optionally, return last cached data if available
        if ticker in stock_cache:
            return stock_cache[ticker][0]
        return None

# Serve the homepage
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Serve the chat page
@app.route('/chat.html')
def chat_page():
    return send_from_directory('.', 'chat.html')

# Serve CSS files
@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

# Endpoint to get stock data
@app.route('/stock', methods=['GET'])
def stock():
    ticker = request.args.get('ticker')
    data = get_stock_data(ticker)
    if data is None or data.empty:
        return jsonify({'error': 'Data temporarily unavailable due to rate limits. Please try again later.'}), 503
    # Convert the data to a format that can be JSON serialized
    result = {}
    for column in data.columns:
        result[column] = {}
        for timestamp, value in data[column].items():
            result[column][str(timestamp)] = value
    return jsonify(result)


# Function to get information for a specific stock symbol
def get_specific_stock_info(symbol, include_recommendation=True, include_news=False):
    try:
        app.logger.info(f"Fetching data for stock symbol: {symbol}")
        
        # Handle the specific case for AIRTELPP.NS
        if symbol == 'AIRTELPP.NS':
            app.logger.info("Processing special case for AIRTELPP.NS")
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period="1d")
                
                if data.empty:
                    app.logger.warning(f"Empty data returned for {symbol}, trying alternative approach")
                    # Try with regular Airtel symbol
                    stock = yf.Ticker("BHARTIARTL.NS")
                    data = stock.history(period="1d")
                    
                    if data.empty:
                        app.logger.warning("Could not get data for BHARTIARTL.NS either")
                        return "I couldn't retrieve data for Bharti Airtel Partly Paid shares (AIRTELPP.NS) at this time. This may be because the partly paid shares have been converted to fully paid shares. Please try querying BHARTIARTL.NS for regular Bharti Airtel shares."
                    
                    # Return information for regular shares with a note
                    result = get_specific_stock_info("BHARTIARTL.NS", include_recommendation, False)
                    if result:
                        return "Note: AIRTELPP.NS (Bharti Airtel Partly Paid shares) may not be available or have been converted to fully paid shares. Here's information for regular Bharti Airtel shares instead:\n\n" + result
                    else:
                        return "I couldn't retrieve specific data for Bharti Airtel at this time. Please try again later."
                
                # If we have data for AIRTELPP.NS, continue normally
            except Exception as e:
                app.logger.error(f"Error processing AIRTELPP.NS: {str(e)}")
                return "I encountered an error while fetching data for Bharti Airtel Partly Paid shares (AIRTELPP.NS). This may be because the partly paid shares have been converted to fully paid shares. Please try querying BHARTIARTL.NS for regular Bharti Airtel shares."
        
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        
        if data.empty:
            app.logger.info(f"No data found for {symbol}")
            return None
            
        # Get basic info
        info = {}
        try:
            info = stock.info
            app.logger.info(f"Successfully retrieved info dict for {symbol}")
        except Exception as e:
            app.logger.warning(f"Could not get additional info for {symbol}: {str(e)}")
            
        # Format the stock name
        stock_name = symbol
        if info and 'longName' in info and info['longName']:
            stock_name = info['longName']
            app.logger.info(f"Using longName: {stock_name}")
        elif info and 'shortName' in info and info['shortName']:
            stock_name = info['shortName']
            app.logger.info(f"Using shortName: {stock_name}")
        else:
            app.logger.info(f"No name found, using symbol: {stock_name}")
            
        # Get the current price
        try:
            last_price = data['Close'].iloc[-1]
            app.logger.info(f"Latest price for {symbol}: {last_price}")
        except Exception as e:
            app.logger.error(f"Error getting price for {symbol}: {str(e)}")
            return f"I found data for {stock_name} ({symbol}) but couldn't retrieve the current price. Please try again later."
            
        currency_symbol = "$"
        
        # Determine currency symbol if available
        if info and 'currency' in info:
            if info['currency'] == 'INR':
                currency_symbol = "₹"
            elif info['currency'] == 'EUR':
                currency_symbol = "€"
            elif info['currency'] == 'GBP':
                currency_symbol = "£"
            elif info['currency'] == 'JPY':
                currency_symbol = "¥"
                
        # Create response
        result = f"{stock_name} ({symbol}) is currently trading at {currency_symbol}{last_price:.2f}. "
        
        # Add price change if available
        try:
            if 'Open' in data and len(data['Open']) > 0:
                change = last_price - data['Open'].iloc[-1]
                percent_change = (change / data['Open'].iloc[-1]) * 100
                
                if change >= 0:
                    result += f"Up {currency_symbol}{abs(change):.2f} (+{abs(percent_change):.2f}%) today. "
                else:
                    result += f"Down {currency_symbol}{abs(change):.2f} (-{abs(percent_change):.2f}%) today. "
        except Exception as e:
            app.logger.warning(f"Couldn't calculate price change: {str(e)}")
        
        # Add additional info
        if info:
            if 'marketCap' in info and info['marketCap']:
                if info['marketCap'] >= 1_000_000_000:
                    market_cap_billions = info['marketCap'] / 1_000_000_000
                    result += f"Market Cap: {currency_symbol}{market_cap_billions:.2f}B. "
                else:
                    market_cap_millions = info['marketCap'] / 1_000_000
                    result += f"Market Cap: {currency_symbol}{market_cap_millions:.2f}M. "
                    
            if 'sector' in info and info['sector']:
                result += f"Sector: {info['sector']}. "
                
            if 'volume' in data and len(data['Volume']) > 0:
                volume = data['Volume'].iloc[-1]
                if volume >= 1_000_000:
                    volume_millions = volume / 1_000_000
                    result += f"Volume: {volume_millions:.2f}M shares. "
                else:
                    result += f"Volume: {volume:,.0f} shares. "
                    
            if 'exchange' in info and info['exchange']:
                result += f"Exchange: {info['exchange']}. "
                
        # Add recommendation if requested
        if include_recommendation:
            try:
                recommendation = get_stock_recommendation(symbol, stock, info, data)
                if recommendation:
                    result += f"\n\n{recommendation}"
            except Exception as rec_error:
                app.logger.error(f"Error generating recommendation for {symbol}: {str(rec_error)}")
                
        # News functionality has been removed
        
        app.logger.info(f"Successfully retrieved info for {symbol}")
        return result
    except Exception as e:
        app.logger.error(f"Error getting specific stock info for {symbol}: {str(e)}", exc_info=True)
        return None

# Function to analyze a stock and provide a recommendation
def get_stock_recommendation(symbol, stock, info, current_data):
    try:
        # Check if we have the necessary data
        if not info or not isinstance(info, dict):
            return None
            
        # Initialize score and reasons
        fundamental_score = 0
        technical_score = 0
        pros = []
        cons = []
        
        # 1. FUNDAMENTAL ANALYSIS
        # ---------------------
        
        # P/E Ratio (Price to Earnings)
        if 'trailingPE' in info and info['trailingPE'] is not None:
            pe_ratio = info['trailingPE']
            if pe_ratio < 15:
                fundamental_score += 2
                pros.append(f"Low P/E ratio ({pe_ratio:.2f}) suggests the stock may be undervalued")
            elif pe_ratio > 30:
                fundamental_score -= 1
                cons.append(f"High P/E ratio ({pe_ratio:.2f}) may indicate overvaluation")
                
        # Debt to Equity
        if 'debtToEquity' in info and info['debtToEquity'] is not None:
            debt_to_equity = info['debtToEquity']
            if debt_to_equity < 0.5:
                fundamental_score += 1
                pros.append(f"Low debt-to-equity ratio ({debt_to_equity:.2f}) indicates financial stability")
            elif debt_to_equity > 1.5:
                fundamental_score -= 1
                cons.append(f"High debt-to-equity ratio ({debt_to_equity:.2f}) may be a concern")
                
        # Profit Margins
        if 'profitMargins' in info and info['profitMargins'] is not None:
            profit_margin = info['profitMargins'] * 100  # Convert to percentage
            if profit_margin > 15:
                fundamental_score += 1
                pros.append(f"Strong profit margin ({profit_margin:.2f}%)")
            elif profit_margin < 5:
                fundamental_score -= 1
                cons.append(f"Low profit margin ({profit_margin:.2f}%)")
                
        # Dividend Yield
        if 'dividendYield' in info and info['dividendYield'] is not None:
            dividend_yield = info['dividendYield'] * 100  # Convert to percentage
            if dividend_yield > 3:
                fundamental_score += 1
                pros.append(f"Good dividend yield ({dividend_yield:.2f}%)")
        
        # 2. TECHNICAL ANALYSIS
        # ---------------------
        
        # Get historical data for technical analysis
        historical_data = stock.history(period="200d")
        
        if not historical_data.empty and len(historical_data) > 50:
            # Moving Averages
            current_price = current_data['Close'].iloc[-1]
            ma50 = historical_data['Close'].tail(50).mean()
            ma200 = historical_data['Close'].tail(200).mean()
            
            # Golden Cross / Death Cross
            if ma50 > ma200:
                technical_score += 1
                pros.append(f"50-day moving average above 200-day moving average (bullish)")
            else:
                technical_score -= 1
                cons.append(f"50-day moving average below 200-day moving average (bearish)")
                
            # Current Price vs Moving Averages
            if current_price > ma50:
                technical_score += 1
                pros.append(f"Price above 50-day moving average")
            else:
                technical_score -= 1
                cons.append(f"Price below 50-day moving average")
                
            # Relative Strength Index (RSI)
            try:
                # Calculate 14-day RSI
                delta = historical_data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                if rsi < 30:
                    technical_score += 1
                    pros.append(f"RSI is {rsi:.2f} - stock may be oversold (potential buy signal)")
                elif rsi > 70:
                    technical_score -= 1
                    cons.append(f"RSI is {rsi:.2f} - stock may be overbought (potential sell signal)")
            except Exception as e:
                app.logger.warning(f"Couldn't calculate RSI: {str(e)}")
                
        # 3. GENERATE RECOMMENDATION
        # -------------------------
        total_score = fundamental_score + technical_score
        
        recommendation = "Investment Analysis:\n"
        
        # Strengths
        if pros:
            recommendation += "\nStrengths:\n"
            for pro in pros[:3]:  # Limit to top 3 pros
                recommendation += f"• {pro}\n"
                
        # Weaknesses
        if cons:
            recommendation += "\nCautions:\n"
            for con in cons[:3]:  # Limit to top 3 cons
                recommendation += f"• {con}\n"
        
        # Overall Recommendation
        recommendation += "\nRecommendation: "
        
        if total_score >= 3:
            recommendation += "⭐⭐⭐⭐⭐ STRONG BUY - This stock shows multiple positive indicators."
        elif total_score == 2:
            recommendation += "⭐⭐⭐⭐ BUY - This stock appears favorable for investment."
        elif total_score == 1:
            recommendation += "⭐⭐⭐ CONSIDER - This stock may be worth considering with further research."
        elif total_score == 0:
            recommendation += "⭐⭐ NEUTRAL - This stock shows mixed signals."
        elif total_score == -1:
            recommendation += "⭐ CAUTION - Some negative indicators suggest careful consideration."
        else:
            recommendation += "NOT RECOMMENDED - Multiple negative indicators suggest looking elsewhere."
            
        recommendation += "\n\nNote: This is an automated analysis and not financial advice. Always do your own research before investing."
        
        return recommendation
    except Exception as e:
        app.logger.error(f"Error generating recommendation: {str(e)}")
        return None

# Define a function to extract and check for stock symbols in user queries
def check_for_stock_info(query, for_chat=False):
    """
    Extract stock symbol from user query and return appropriate values.
    If for_chat=True: Returns a tuple of (stock_symbol, stock_text) 
    If for_chat=False: Returns stock info string or None
    """
    app.logger.info(f"Checking for stock info in: {query}")
    
    # Clean up the query for better matching
    cleaned_query = query.strip().upper()
    
    # Remove trailing words like "STOCK", "PRICE", etc.
    cleaned_query = re.sub(r'\s+(STOCK|PRICE|OF|SHARES?)$', '', cleaned_query)
    
    # Define patterns to identify stock price queries
    patterns = [
        # Direct symbol queries
        r'(?:^|[^\w])([A-Z]{1,5})(?:\.[A-Z]{2})?(?:[^\w]|$)',  # Simple ticker like AAPL or TCS.NS
        
        # Natural language queries
        r'(?:PRICE|VALUE|QUOTE|INFO|DETAILS|INFORMATION|DATA)\s+(?:OF|FOR|ON|ABOUT)?\s+([A-Z0-9\.\-_]+)',
        r'(?:HOW\s+(?:MUCH|IS))\s+([A-Z0-9\.\-_]+)(?:\s+WORTH)?',
        r'(?:WHAT\'?S|WHAT\s+IS)\s+(?:THE\s+)?(?:PRICE|VALUE|QUOTE)\s+(?:OF|FOR)?\s+([A-Z0-9\.\-_]+)',
        r'(?:TELL\s+(?:ME\s+)?ABOUT)\s+([A-Z0-9\.\-\s_]+)',
        r'(?:INFO|INFORMATION|DETAILS|DATA)\s+(?:FOR|ON|ABOUT)?\s+([A-Z0-9\.\-\s_]+)',
    ]
    
    # Check for known company/stock mappings to handle company names
    company_to_symbol = {
        # Indian stocks
        'RELIANCE': 'RELIANCE.NS',
        'RELIANCE INDUSTRIES': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'TATA CONSULTANCY SERVICES': 'TCS.NS',
        'HDFC': 'HDFCBANK.NS',
        'HDFC BANK': 'HDFCBANK.NS',
        'INFOSYS': 'INFY.NS',
        'SBI': 'SBIN.NS',
        'STATE BANK OF INDIA': 'SBIN.NS',
        'ICICI': 'ICICIBANK.NS',
        'ICICI BANK': 'ICICIBANK.NS',
        'AIRTEL': 'BHARTIARTL.NS',
        'BHARTI AIRTEL': 'BHARTIARTL.NS',
        'ADANI PORTS': 'ADANIPORTS.NS',
        'TATA MOTORS': 'TATAMOTORS.NS',
        
        # US stocks
        'APPLE': 'AAPL',
        'MICROSOFT': 'MSFT',
        'AMAZON': 'AMZN',
        'GOOGLE': 'GOOGL',
        'ALPHABET': 'GOOGL',
        'TESLA': 'TSLA',
        'META': 'META',
        'FACEBOOK': 'META',
        'NETFLIX': 'NFLX',
        'JPMORGAN': 'JPM',
        'JP MORGAN': 'JPM',
        'VISA': 'V',
        'DISNEY': 'DIS',
        'NVIDIA': 'NVDA',
        
        # Major indices
        'NIFTY': '^NSEI',
        'NIFTY 50': '^NSEI',
        'SENSEX': '^BSESN',
        'DOW': '^DJI',
        'DOW JONES': '^DJI',
        'NASDAQ': '^IXIC',
        'S&P': '^GSPC',
        'S&P 500': '^GSPC',
    }
    
    # First, check for direct company name matches
    for company, symbol in company_to_symbol.items():
        if company in cleaned_query:
            app.logger.info(f"Found direct company match: {company} -> {symbol}")
            
            # For generate_local_response
            stock_info = get_specific_stock_info(symbol)
            if stock_info:
                if for_chat:
                    return symbol, company
                else:
                    return stock_info
            
    # Try matching patterns for stock symbols
    for pattern in patterns:
        match = re.search(pattern, cleaned_query)
        if match:
            potential_symbol = match.group(1).strip()
            
            # Clean up potential multi-word symbols for better matching
            potential_symbol = re.sub(r'\s+', '', potential_symbol)
            
            app.logger.info(f"Found pattern match: {potential_symbol}")
            
            # Handle NIFTY/SENSEX as special cases
            if 'NIFTY' in potential_symbol:
                symbol = '^NSEI'
                stock_info = get_specific_stock_info(symbol)
                if stock_info:
                    if for_chat:
                        return symbol, potential_symbol
                    else:
                        return stock_info
            elif 'SENSEX' in potential_symbol:
                symbol = '^BSESN'
                stock_info = get_specific_stock_info(symbol)
                if stock_info:
                    if for_chat:
                        return symbol, potential_symbol
                    else:
                        return stock_info
            
            # Try with the potential symbol directly
            symbol = potential_symbol
            stock_info = get_specific_stock_info(symbol)
            
            if stock_info:
                if for_chat:
                    return symbol, potential_symbol
                else:
                    return stock_info
            
            # For Indian companies, try adding .NS suffix if missing
            if not '.' in symbol and len(symbol) >= 2:
                symbol_with_ns = f"{symbol}.NS"
                app.logger.info(f"Trying with .NS suffix: {symbol_with_ns}")
                stock_info = get_specific_stock_info(symbol_with_ns)
                
                if stock_info:
                    if for_chat:
                        return symbol_with_ns, potential_symbol
                    else:
                        return stock_info
    
    app.logger.info("No stock info found in query")
    # Different return value based on context
    if for_chat:
        return None, None
    else:
        return None

# Check for specific stock recommendation requests
def is_recommendation_request(query):
    query = query.lower()
    recommendation_phrases = [
        'should i buy', 'worth buying', 'good investment', 'invest in', 
        'recommend', 'good stock', 'buy or sell', 'good time to buy',
        'analysis', 'evaluate', 'assessment', 'outlook'
    ]
    
    for phrase in recommendation_phrases:
        if phrase in query:
            return True
            
    return False

# Extract stock symbol from recommendation request
def extract_symbol_from_recommendation(query):
    query = query.lower()
    
    patterns = [
        r'should i buy ([a-zA-Z0-9\.\-_]+)',
        r'is ([a-zA-Z0-9\.\-_]+) (?:a good|worth) (?:investment|buying)',
        r'(?:recommend|thoughts on|analysis of|evaluate) ([a-zA-Z0-9\.\-_]+)',
        r'buy or sell ([a-zA-Z0-9\.\-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).upper()
            
    # If no direct match, try to find any stock-like symbols in the text
    words = query.split()
    for word in words:
        # Look for typical stock formats (.NS suffix, all caps, etc)
        if ('.' in word or word.isupper()) and len(word) >= 2 and len(word) <= 15:
            return word.upper()
    
    return None

# Function to generate a local response when OpenAI API is unavailable
def generate_local_response(query):
    # Common stock symbols with their company names
    known_stocks = {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'SBIN.NS': 'State Bank of India',
        'ICICIBANK.NS': 'ICICI Bank',
        'AIRTELPP.NS': 'Bharti Airtel Partly Paid',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'ADANIPORTS.NS': 'Adani Ports',
        'TATAMOTORS.NS': 'Tata Motors',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'AMZN': 'Amazon',
        'GOOGL': 'Alphabet (Google)',
        'TSLA': 'Tesla',
        'META': 'Meta Platforms (Facebook)',
        'NFLX': 'Netflix',
        'JPM': 'JPMorgan Chase',
        'V': 'Visa',
        'DIS': 'Disney'
    }
    
    # Check if the user is asking about a stock
    stock_info = check_for_stock_info(query, for_chat=False)
    if stock_info:
        return stock_info
    
    # Check for known stock ticker directly in the query
    for ticker, name in known_stocks.items():
        if ticker.lower() in query.lower():
            info = get_specific_stock_info(ticker)
            if info:
                return info
            else:
                return f"I couldn't find information for {ticker} ({name}). Please check if the ticker is correct or try a different stock."
    
    # For greeting messages
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    if any(greeting in query.lower() for greeting in greetings):
        return "Hello! I'm your AI Investment Assistant. I can help you with stock information, particularly for Indian and global markets. What stock would you like to know about?"
    
    # For thank you messages
    thanks = ['thank', 'thanks', 'appreciate', 'grateful']
    if any(thank in query.lower() for thank in thanks):
        return "You're welcome! Is there any specific stock you'd like to know about?"
    
    # Default fallback response
    return "I'm currently operating in stock information mode. You can ask me about specific stocks like 'What's the price of RELIANCE.NS?' or 'Tell me about Apple stock'."

# Function to detect if a query is a natural language market analysis request
def is_natural_language_market_analysis(query):
    query = query.lower()
    keywords = [
        "explain the effect",
        "summarize key drivers",
        "market trend",
        "economic indicators",
        "policy shifts",
        "tariffs",
        "trade war",
        "inflation",
        "interest rates",
        "market analysis",
        "natural language",
        "detailed insights",
        "financial trends",
        "stock market trends",
        "market sentiment",
        "macroeconomic",
        "geopolitical",
        "market impact",
        "trade policy",
        "economic policy",
        "market forecast",
        "market outlook",
        "market prediction",
        "market direction",
        "fed decision",
        "central bank",
        "market report",
        "global markets",
        "sector performance",
        "industry outlook",
        "analyze",
        "analyse",
        "performance",
        "america",
        "us market",
        "global economy",
        "market condition",
        "economy",
        "bull market",
        "bear market",
        "volatile",
        "volatility",
        "india",
        "indian economy",
        "nifty",
        "sensex",
        "rbi"
    ]
    return any(keyword in query for keyword in keywords)

# Function to fetch economic indicators (mocked for now)
def get_economic_indicators(country=None):
    # Default to US data
    us_indicators = """
    - U.S. unemployment rate: 3.7%
    - U.S. inflation rate (CPI): 4.2%
    - Federal Reserve interest rate: 5.25%
    - GDP growth rate (Q1 2024): 2.1%
    - Consumer confidence index: 98.5
    """
    
    # India-specific indicators
    india_indicators = """
    - India unemployment rate: 7.4%
    - India inflation rate (CPI): 5.1%
    - RBI repo rate: 6.5%
    - GDP growth rate (FY 2023-24): 7.2%
    - India manufacturing PMI: 57.3
    """

    # Pakistan-specific indicators
    pakistan_indicators = """
    - Pakistan unemployment rate: 5.8%
    - Pakistan inflation rate (CPI): 12.3%
    - SBP policy rate: 13.75%
    - GDP growth rate (FY 2023-24): 3.5%
    - Pakistan manufacturing PMI: 50.2
    """

    # China-specific indicators
    china_indicators = """
    - China unemployment rate: 5.5%
    - China inflation rate (CPI): 2.3%
    - People's Bank of China benchmark lending rate: 3.65%
    - GDP growth rate (Q1 2024): 5.0%
    - China manufacturing PMI: 51.2
    """

    # EU-specific indicators
    eu_indicators = """
    - EU unemployment rate: 6.2%
    - EU inflation rate (CPI): 3.1%
    - European Central Bank interest rate: 4.0%
    - GDP growth rate (Q1 2024): 1.8%
    - EU manufacturing PMI: 49.8
    """

    # UK-specific indicators
    uk_indicators = """
    - UK unemployment rate: 4.1%
    - UK inflation rate (CPI): 5.0%
    - Bank of England base rate: 4.25%
    - GDP growth rate (Q1 2024): 1.5%
    - UK manufacturing PMI: 50.5
    """
    
    if country:
        country_lower = country.lower()
        if country_lower == 'india':
            return india_indicators
        elif country_lower == 'pakistan':
            return pakistan_indicators
        elif country_lower == 'china':
            return china_indicators
        elif country_lower == 'eu':
            return eu_indicators
        elif country_lower == 'uk':
            return uk_indicators
        else:
            return us_indicators
    else:
        return us_indicators

# Function to fetch recent policy shifts (mocked for now)
def get_policy_shifts(country=None):
    # Default to US policy data
    us_policy = """
    - Recent U.S. tariffs on imported steel and aluminum increased by 10%
    - New trade agreements signed with several Asian countries
    - Proposed changes to corporate tax rates under review
    - Environmental regulations tightened for manufacturing sector
    """
    
    # India-specific policy data
    india_policy = """
    - RBI maintained repo rate at 6.5% in latest monetary policy
    - New foreign direct investment regulations implemented for e-commerce
    - Production-Linked Incentive (PLI) schemes expanded to more sectors
    - GST Council considering rate rationalization for several product categories
    """

    # Pakistan-specific policy data
    pakistan_policy = """
    - SBP maintained policy rate at 13.75% amid inflation concerns
    - New trade agreements with China and Middle Eastern countries
    - Tax reforms aimed at increasing revenue collection
    - Energy sector subsidies under review
    """

    # China-specific policy data
    china_policy = """
    - People's Bank of China kept benchmark lending rate steady at 3.65%
    - New trade agreements with ASEAN countries
    - Environmental regulations tightened on heavy industries
    - Expansion of digital currency pilot programs
    """

    # EU-specific policy data
    eu_policy = """
    - European Central Bank maintained interest rates at 4.0%
    - New carbon emission regulations for manufacturing
    - Trade negotiations ongoing with UK and US
    - Expansion of green energy subsidies
    """

    # UK-specific policy data
    uk_policy = """
    - Bank of England kept base rate at 4.25%
    - New tax incentives for technology startups
    - Trade agreements with Commonwealth countries
    - Energy price caps reviewed amid inflation concerns
    """
    
    if country:
        country_lower = country.lower()
        if country_lower == 'india':
            return india_policy
        elif country_lower == 'pakistan':
            return pakistan_policy
        elif country_lower == 'china':
            return china_policy
        elif country_lower == 'eu':
            return eu_policy
        elif country_lower == 'uk':
            return uk_policy
        else:
            return us_policy
    else:
        return us_policy

# Function to gather market data for enhanced analysis
def get_market_data_for_analysis(country=None):
    market_data = {}
    
    # Get data for key indices
    if country:
        country_lower = country.lower()
        if country_lower == 'india':
            # India-focused indices
            indices = {
                "Nifty 50": "^NSEI",
                "Sensex": "^BSESN",
                "Nifty Bank": "^NSEBANK",
                "Nifty IT": "^CNXIT",
                "Nifty Auto": "^CNXAUTO",
                "India VIX": "^INDIAVIX"
            }
        elif country_lower == 'pakistan':
            # Pakistan-focused indices (mock symbols, replace with actual if available)
            indices = {
                "KSE 100": "^KSE",
                "KSE 30": "^KSE30",
                "Pakistan VIX": "^PAKVIX"
            }
        elif country_lower == 'china':
            # China-focused indices (mock symbols, replace with actual if available)
            indices = {
                "Shanghai Composite": "000001.SS",
                "CSI 300": "000300.SS",
                "China VIX": "^CHIVIX"
            }
        elif country_lower == 'eu':
            # EU-focused indices (mock symbols, replace with actual if available)
            indices = {
                "Euro Stoxx 50": "^STOXX50E",
                "DAX": "^GDAXI",
                "FTSE 100": "^FTSE"
            }
        elif country_lower == 'uk':
            # UK-focused indices
            indices = {
                "FTSE 100": "^FTSE",
                "FTSE 250": "^FTMC",
                "UK VIX": "^UKVIX"
            }
        else:
            # US/Global indices
            indices = {
                "S&P 500": "^GSPC",
                "Dow Jones": "^DJI",
                "NASDAQ": "^IXIC",
                "Russell 2000": "^RUT",
                "VIX": "^VIX"
            }
    else:
        # Default to US/Global indices
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
    
    for name, symbol in indices.items():
        try:
            data = get_stock_data(symbol)
            if not data.empty:
                last_price = data['Close'].iloc[-1]
                open_price = data['Open'].iloc[-1]
                change = last_price - open_price
                percent_change = (change / open_price) * 100
                
                market_data[name] = {
                    "price": last_price,
                    "change": change,
                    "percent_change": percent_change,
                    "direction": "up" if change >= 0 else "down"
                }
        except Exception as e:
            app.logger.warning(f"Could not get data for {name} ({symbol}): {str(e)}")
    
    return market_data

# Get current sector performance
def get_sector_performance(country=None):
    if country:
        country_lower = country.lower()
        if country_lower == 'india':
            # India sectors with representative stocks
            sectors = {
                "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
                "Banking": ["HDFCBANK.NS", "SBIN.NS", "ICICIBANK.NS"],
                "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
                "Auto": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS"],
                "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS"],
                "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"]
            }
        elif country_lower == 'pakistan':
            # Pakistan sectors with representative stocks (mock tickers, replace with actual if available)
            sectors = {
                "Banking": ["UBL.KA", "HBL.KA", "MCB.KA"],
                "Energy": ["PSO.KA", "KAPCO.KA", "SNGP.KA"],
                "Telecom": ["PTCL.KA", "CMPAK.KA", "WATEEN.KA"],
                "Cement": ["LUCK.KA", "DGKC.KA", "FECTC.KA"],
                "Fertilizer": ["FATIMA.KA", "FFBL.KA", "ENGRO.KA"]
            }
        elif country_lower == 'china':
            # China sectors with representative stocks (mock tickers)
            sectors = {
                "Technology": ["BIDU", "TCEHY", "JD"],
                "Finance": ["ICBC", "CMB", "ABC"],
                "Energy": ["SNP", "PTR", "HNP"],
                "Consumer": ["MCD", "KFC", "YUMC"],
                "Industrial": ["BAIC", "CRRC", "CSIC"]
            }
        elif country_lower == 'eu':
            # EU sectors with representative stocks (mock tickers)
            sectors = {
                "Technology": ["SAP.DE", "ASML.AS", "STM.PA"],
                "Finance": ["HSBA.L", "BNP.PA", "DBK.DE"],
                "Energy": ["RDSA.AS", "ENEL.MI", "IBE.MC"],
                "Consumer": ["NESN.SW", "LVMH.PA", "AD.AS"],
                "Industrial": ["SIEMENS.DE", "AIR.PA", "VOW3.DE"]
            }
        elif country_lower == 'uk':
            # UK sectors with representative stocks
            sectors = {
                "Technology": ["SAGE.L", "AVE.L", "CDR.L"],
                "Finance": ["HSBA.L", "BARC.L", "LLOY.L"],
                "Energy": ["BP.L", "RDSA.L", "NG.L"],
                "Consumer": ["DGE.L", "ULVR.L", "MKS.L"],
                "Industrial": ["BA.L", "GKN.L", "RR.L"]
            }
        else:
            # US sectors with representative stocks
            sectors = {
                "Technology": ["AAPL", "MSFT", "GOOGL"],
                "Finance": ["JPM", "BAC", "GS"],
                "Healthcare": ["JNJ", "PFE", "UNH"],
                "Consumer": ["AMZN", "WMT", "PG"],
                "Energy": ["XOM", "CVX", "COP"],
                "Industrial": ["GE", "BA", "CAT"]
            }
    else:
        # Default to US sectors
        sectors = {
            "Technology": ["AAPL", "MSFT", "GOOGL"],
            "Finance": ["JPM", "BAC", "GS"],
            "Healthcare": ["JNJ", "PFE", "UNH"],
            "Consumer": ["AMZN", "WMT", "PG"],
            "Energy": ["XOM", "CVX", "COP"],
            "Industrial": ["GE", "BA", "CAT"]
        }
    
    results = {}
    
    for sector, stocks in sectors.items():
        sector_changes = []
        for stock in stocks:
            try:
                data = get_stock_data(stock)
                if not data.empty:
                    last_price = data['Close'].iloc[-1]
                    open_price = data['Open'].iloc[-1]
                    percent_change = ((last_price - open_price) / open_price) * 100
                    sector_changes.append(percent_change)
            except Exception:
                continue
        
        if sector_changes:
            avg_change = sum(sector_changes) / len(sector_changes)
            results[sector] = {
                "change": avg_change,
                "direction": "up" if avg_change >= 0 else "down",
                "strength": "strong" if abs(avg_change) > 1 else "moderate" if abs(avg_change) > 0.3 else "weak"
            }
    
    return results

# Function to generate content using Gemini API
def generate_gemini_response(prompt, max_tokens=500):
    try:
        app.logger.info(f"Sending prompt to Gemini: {prompt[:50]}...")
        
        # Set generation configuration
        generation_config = {
            'max_output_tokens': max_tokens,
            'temperature': 0.3,
            'top_p': 0.8,
            'top_k': 40
        }
        
        # Generate content with proper error handling
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Check for valid response
        if response and hasattr(response, 'text'):
            app.logger.info("Successfully received response from Gemini")
            return response.text
        else:
            app.logger.warning("Empty or invalid response from Gemini API")
            return None
    except Exception as e:
        app.logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
        # Return a more detailed error message for debugging
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Detailed error trace: {error_details}")
        return None

# Modify the chat endpoint to handle investment advice queries better
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        app.logger.info(f"Received chat request: {user_input}")
        
        # Special handling for checking API key status
        if "api key" in user_input.lower() and ("working" in user_input.lower() or "status" in user_input.lower()):
            try:
                # Test the API
                test_prompt = "Respond with 'API is working' if you receive this message."
                test_response = generate_gemini_response(test_prompt)
                
                if test_response and "API is working" in test_response:
                    return jsonify("Yes, your Google Gemini API key is working correctly! You can use the full functionality of the chatbot.")
                else:
                    return jsonify("Your API key may not be working correctly. The chatbot will still work for stock information, but AI-powered analysis might be limited.")
            except Exception as e:
                app.logger.error(f"Error checking API key: {str(e)}")
                return jsonify("There was an error checking your API key status. The chatbot will still work for stock information.")
        
        # Check for investment advice questions
        investment_advice_patterns = [
            r'(good|right|best) time to (invest|buy)',
            r'should I (invest|buy|sell)',
            r'worth (investing|buying)',
            r'(invest|put money) (in|into)',
            r'good investment',
            r'investment advice',
            r'portfolio recommendation'
        ]
        
        if any(re.search(pattern, user_input.lower()) for pattern in investment_advice_patterns):
            app.logger.info("Detected investment advice query")
            response = handle_investment_advice_query(user_input)
            return jsonify(response)
        
        # Handle market analysis queries directly in the chat endpoint
        if is_natural_language_market_analysis(user_input):
            app.logger.info("Detected market analysis query in chat endpoint")
            try:
                # Determine if this is an India-specific query
                is_india_query = any(term in user_input.lower() for term in ["india", "indian", "nifty", "sensex", "rbi", "bharti", "reliance"])
                is_pakistan_query = any(term in user_input.lower() for term in ["pakistan", "kse", "karachi", "pakistani"])
                is_china_query = any(term in user_input.lower() for term in ["china", "chinese", "shanghai", "beijing", "csi 300"])
                is_eu_query = any(term in user_input.lower() for term in ["eu", "european union", "europe", "eurozone", "dax", "stoxx", "ftse"])
                is_uk_query = any(term in user_input.lower() for term in ["uk", "united kingdom", "london", "ftse 100", "ftse 250"])
                country = "us"  # default
                
                if is_india_query:
                    country = "india"
                elif is_pakistan_query:
                    country = "pakistan"
                elif is_china_query:
                    country = "china"
                elif is_eu_query:
                    country = "eu"
                elif is_uk_query:
                    country = "uk"
                
                app.logger.info(f"Market analysis query for country: {country}")
                
                # Gather comprehensive market data
                economic_data = get_economic_indicators(country)
                policy_data = get_policy_shifts(country)
                market_indices = get_market_data_for_analysis(country)
                sectors = get_sector_performance(country)

                # Debug: Log the fetched data
                app.logger.info(f"Fetched market_indices: {market_indices}")
                app.logger.info(f"Fetched sectors: {sectors}")
                
                # Format data clearly
                data = f"## Market Data Summary for {country.upper()}\n\n"
                
                # Add market indices
                data += "### Market Indices\n"
                for name, info in market_indices.items():
                    direction = "up" if info['change'] >= 0 else "down"
                    emoji = "📈" if info['change'] >= 0 else "📉"
                    data += f"{emoji} {name}: {info['price']:.2f} ({direction} {abs(info['percent_change']):.2f}%)\n"
                
                # Add sector performance
                data += "\n### Sector Performance\n"
                for sector, info in sectors.items():
                    emoji = "🟢" if info['direction'] == "up" else "🔴"
                    data += f"{emoji} {sector}: {info['change']:.2f}% ({info['strength']} {info['direction']})\n"
                
                # Add economic indicators and policy information
                data += f"\n### Economic Indicators\n{economic_data}\n"
                data += f"\n### Policy Updates\n{policy_data}\n"
                # Debug: Log the prompt
                app.logger.info(f"Prompt sent to Gemini:\n{data}\nUser question: {user_input}")
                # Try to get AI analysis with a better prompt
                try:
                    prompt = f"""You are a financial market analyst. Based on this data about {country} markets:\n{data}\n\nPlease provide a thorough analysis addressing this question: {user_input}\n\nImportant instructions:\n1. Include all relevant data points from the provided information in your analysis\n2. Format your response with clear headings, bullet points, and use emojis for readability\n3. Your analysis should be complete and standalone without needing to show the raw data\n4. Summarize key metrics from the raw data within your analysis text\n5. Focus on trends, patterns, and implications\n"""
                    app.logger.info(f"Gemini prompt:\n{prompt}")
                    ai_analysis = generate_gemini_response(prompt)
                    
                    if ai_analysis:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                        # Return just the AI analysis without the raw source data
                        return jsonify(f"{ai_analysis}")
                except Exception as ai_error:
                    app.logger.error(f"Error getting AI analysis: {str(ai_error)}")
                
                # Basic analysis as fallback
                if is_india_query:
                    india_indices = ["Nifty 50", "Sensex", "Nifty Bank"]
                    india_trend = ""
                    
                    for idx in india_indices:
                        if idx in market_indices:
                            direction = "positive" if market_indices[idx]['change'] >= 0 else "negative"
                            magnitude = abs(market_indices[idx]['percent_change'])
                            if magnitude < 0.5:
                                strength = "slightly"
                            elif magnitude < 1.0:
                                strength = "moderately"
                            else:
                                strength = "strongly"
                            
                            india_trend += f"The {idx} is trending {strength} {direction} at {magnitude:.2f}%. "
                    
                    if india_trend:
                        india_analysis = f"## India Market Analysis\n\n{india_trend}\n\nIT and Banking sectors are currently the key drivers of market movement. Recent RBI monetary policy and government initiatives in manufacturing are shaping market sentiment."
                        return jsonify(india_analysis)
                elif is_pakistan_query:
                    pakistan_indices = ["KSE 100", "KSE 30"]
                    pakistan_trend = ""
                    
                    for idx in pakistan_indices:
                        if idx in market_indices:
                            direction = "positive" if market_indices[idx]['change'] >= 0 else "negative"
                            magnitude = abs(market_indices[idx]['percent_change'])
                            if magnitude < 0.5:
                                strength = "slightly"
                            elif magnitude < 1.0:
                                strength = "moderately"
                            else:
                                strength = "strongly"
                            
                            pakistan_trend += f"The {idx} is trending {strength} {direction} at {magnitude:.2f}%. "
                    
                    if pakistan_trend:
                        pakistan_analysis = f"## Pakistan Market Analysis\n\n{pakistan_trend}\n\nBanking and Energy sectors are currently the key drivers of market movement. Recent SBP monetary policy and trade agreements are influencing market sentiment."
                        return jsonify(pakistan_analysis)
                elif is_china_query:
                    china_indices = ["Shanghai Composite", "CSI 300"]
                    china_trend = ""
                    
                    for idx in china_indices:
                        if idx in market_indices:
                            direction = "positive" if market_indices[idx]['change'] >= 0 else "negative"
                            magnitude = abs(market_indices[idx]['percent_change'])
                            if magnitude < 0.5:
                                strength = "slightly"
                            elif magnitude < 1.0:
                                strength = "moderately"
                            else:
                                strength = "strongly"
                            
                            china_trend += f"The {idx} is trending {strength} {direction} at {magnitude:.2f}%. "
                    
                    if china_trend:
                        china_analysis = f"## China Market Analysis\n\n{china_trend}\n\nTechnology and Manufacturing sectors are currently the key drivers of market movement. Recent PBOC policies and trade agreements are influencing market sentiment."
                        return jsonify(china_analysis)
                elif is_eu_query:
                    eu_indices = ["Euro Stoxx 50", "DAX", "FTSE 100"]
                    eu_trend = ""
                    
                    for idx in eu_indices:
                        if idx in market_indices:
                            direction = "positive" if market_indices[idx]['change'] >= 0 else "negative"
                            magnitude = abs(market_indices[idx]['percent_change'])
                            if magnitude < 0.5:
                                strength = "slightly"
                            elif magnitude < 1.0:
                                strength = "moderately"
                            else:
                                strength = "strongly"
                            
                            eu_trend += f"The {idx} is trending {strength} {direction} at {magnitude:.2f}%. "
                    
                    if eu_trend:
                        eu_analysis = f"## EU Market Analysis\n\n{eu_trend}\n\nFinance and Energy sectors are currently the key drivers of market movement. ECB policies and trade negotiations are influencing market sentiment."
                        return jsonify(eu_analysis)
                elif is_uk_query:
                    uk_indices = ["FTSE 100", "FTSE 250"]
                    uk_trend = ""
                    
                    for idx in uk_indices:
                        if idx in market_indices:
                            direction = "positive" if market_indices[idx]['change'] >= 0 else "negative"
                            magnitude = abs(market_indices[idx]['percent_change'])
                            if magnitude < 0.5:
                                strength = "slightly"
                            elif magnitude < 1.0:
                                strength = "moderately"
                            else:
                                strength = "strongly"
                            
                            uk_trend += f"The {idx} is trending {strength} {direction} at {magnitude:.2f}%. "
                    
                    if uk_trend:
                        uk_analysis = f"## UK Market Analysis\n\n{uk_trend}\n\nBanking and Consumer sectors are currently the key drivers of market movement. BOE policies and trade agreements are influencing market sentiment."
                        return jsonify(uk_analysis)
                
                # If no specific country analysis, return condensed data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                condensed_analysis = f"## Market Overview (as of {timestamp})\n\nMajor indices show mixed performance today. Sector analysis indicates Finance is performing well while Consumer and Industrial sectors are facing pressure."
                return jsonify(condensed_analysis)
            except Exception as analysis_error:
                app.logger.error(f"Error in market analysis: {str(analysis_error)}")
                return jsonify("I encountered an error analyzing market trends. Here's what I can tell you: Markets are showing mixed signals with technology and healthcare sectors performing well. Would you like information about a specific stock instead?")
        
        # Check for stock queries
        stock_symbol, stock_text = check_for_stock_info(user_input, for_chat=True)
        if stock_symbol:
            try:
                # Get factual data from Yahoo Finance
                stock_info = get_specific_stock_info(stock_symbol)
                if stock_info:
                    # Try to get AI analysis
                    try:
                        prompt = f"""You are a financial expert. Based on this stock data:
{stock_info}

Provide a brief, factual analysis of {stock_symbol}. Focus on the current price, recent performance, and key metrics. Be concise and objective. Do not give investment advice."""

                        ai_analysis = generate_gemini_response(prompt)
                        
                        if ai_analysis:
                            return jsonify(f"{stock_info}\n\n**AI Analysis:**\n{ai_analysis}")
                    except Exception as ai_error:
                        app.logger.error(f"Error getting AI analysis for stock: {str(ai_error)}")
                    
                    return jsonify(stock_info)
                else:
                    return jsonify(f"I couldn't find information for {stock_text}. For Indian stocks, you might need to be more specific or add the .NS suffix.")
            except Exception as stock_error:
                app.logger.error(f"Error fetching stock info: {str(stock_error)}")
                return jsonify(f"I had trouble getting information for {stock_text}. Please try again later or try another stock symbol.")
        
        # For all other queries
        try:
            # Try to get AI response for general query
            prompt = f"""You are an AI assistant specialized in financial markets and investments. 
Your knowledge is focused on explaining financial concepts, market mechanics, and investment principles.
Provide helpful, concise, and accurate information about this financial question: "{user_input}"
Do not give specific investment advice or recommendations."""
            
            ai_response = generate_gemini_response(prompt)
            
            if ai_response:
                return jsonify(ai_response)
            
            # Fallback for general queries
            if is_recommendation_request(user_input):
                return jsonify("I can provide factual information about stocks, but I cannot make investment recommendations. To get information about a specific stock, please ask about it directly, like 'What's the current price of MSFT?' or 'Tell me about Apple stock.'")
            elif any(greeting in user_input.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
                return jsonify("Hello! I'm your AI Investment Assistant. I can help you with stock information and market analysis. What would you like to know about?")
            else:
                return jsonify(generate_local_response(user_input))
        except Exception as general_error:
            app.logger.error(f"Error handling general query: {str(general_error)}")
            return jsonify("I'm currently experiencing technical difficulties with advanced queries. Please try asking about a specific stock (like 'TATAMOTORS.NS' or 'MSFT') instead.")
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"Unexpected error in chat endpoint: {error_msg}")
        return jsonify("I'm currently experiencing technical difficulties. Please try asking about a specific stock instead.")

# Update the API status endpoint
@app.route('/api-status', methods=['GET'])
def api_status():
    try:
        # Test the API with a simple prompt
        response = generate_gemini_response("Hello, respond with 'API is working' if you receive this message.")
        
        if response and "API is working" in response:
            app.logger.info("API status check: successful")
            return jsonify({
                "status": "operational", 
                "code": 200, 
                "message": "Gemini API is working correctly",
                "model": "gemini-1.5-pro",
                "api_key": f"{api_key[:5]}...{api_key[-5:]}"
            })
        else:
            app.logger.warning("API status check: response received but invalid content")
            error_info = {
                "status": "error", 
                "code": 500, 
                "message": "API responded, but did not return the expected message",
                "response": response if response else "No response",
                "model": "gemini-1.5-pro"
            }
            return jsonify(error_info)
    except Exception as e:
        app.logger.error(f"API status check: failed with error: {str(e)}")
        return jsonify({
            "status": "error", 
            "code": 500,
            "message": str(e),
            "model": "gemini-1.5-pro",
            "api_key_prefix": api_key[:5]
        })

# Handle investment advice queries
def handle_investment_advice_query(query):
    try:
        # Create a prompt that focuses on market analysis rather than direct advice
        prompt = f"""You are a financial analyst providing educational information only, not investment advice. 
Based on this query: "{query}"

Provide factual market information and educational context that might be relevant.
Always clarify that you're not giving personal investment advice.
Focus on explaining market trends, fundamentals, and educational aspects.
Use clear headings, bullet points, and include relevant data points."""
        
        response = generate_gemini_response(prompt)
        
        if response:
            disclaimer = "\n\n**Disclaimer**: This is educational information only and not personalized investment advice. Always do your own research and consider consulting with a financial advisor before making investment decisions."
            return response + disclaimer
        else:
            return "I can provide factual market information, but cannot offer personalized investment advice. Would you like to know about specific market trends or stocks instead?"
    except Exception as e:
        app.logger.error(f"Error handling investment advice query: {str(e)}")
        return "I'm currently experiencing difficulties with providing detailed market analysis. Would you like me to provide information about a specific stock instead?"

testimonials = []  # Global list to store testimonials in memory

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    name = request.form.get('name') or request.json.get('name')
    role = request.form.get('role') or request.json.get('role')
    email = request.form.get('email') or request.json.get('email')
    rating = request.form.get('rating') or request.json.get('rating')
    testimonial = request.form.get('testimonial') or request.json.get('testimonial')

    # Save testimonial in memory
    testimonials.append({
        'name': name,
        'role': role,
        'email': email,
        'rating': int(rating) if rating else 0,
        'testimonial': testimonial
    })

    print(f"Feedback received: Name={name}, Role={role}, Email={email}, Rating={rating}, Testimonial={testimonial}")

    return jsonify({'success': True, 'message': 'Feedback submitted! (Python backend)'})

@app.route('/get_testimonials', methods=['GET'])
def get_testimonials():
    return jsonify(testimonials)

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
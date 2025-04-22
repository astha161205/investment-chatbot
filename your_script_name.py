import yfinance as yf
import requests
import json
from flask import Flask, request, jsonify, send_from_directory
import os
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
api_key = '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'

# Track API availability
openai_api_available = True  # Set to True to indicate OpenAI API is available

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    return data

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
    
    # Convert the data to a format that can be JSON serialized
    result = {}
    for column in data.columns:
        result[column] = {}
        for timestamp, value in data[column].items():
            # Convert timestamp to string to make it JSON serializable
            result[column][str(timestamp)] = value
    
    return jsonify(result)

# Function to get news for a stock using Yahoo Finance
# def get_stock_news(symbol, limit=3):
#     try:
#         app.logger.info(f"Fetching news for symbol: {symbol} using yfinance")
#         stock = yf.Ticker(symbol)
#         news = stock.news
        
#         app.logger.info(f"Raw news data received for {symbol}: {news}") # Log raw data
        
#         if not news:
#             app.logger.info(f"No news found for {symbol} via yfinance.")
#             return None
            
#         # Process the news items
#         news_items = []
#         app.logger.info(f"Processing {len(news)} news items for {symbol}...")
#         for i, item in enumerate(news[:limit]):  # Limit to the requested number of items
#             app.logger.info(f"Processing item {i}: {item}") # Log each item
            
#             # Format the date (provider_publish_time is usually a Unix timestamp)
#             published_date = "Unknown date"
#             if 'providerPublishTime' in item and item['providerPublishTime']:
#                 try:
#                     timestamp = item['providerPublishTime']
#                     dt = datetime.fromtimestamp(timestamp)
#                     published_date = dt.strftime("%b %d, %Y")
#                 except Exception as date_err:
#                     app.logger.warning(f"Could not parse news date for {symbol}: {date_err} (Timestamp: {item.get('providerPublishTime')})")
#             else:
#                 app.logger.warning(f"No providerPublishTime found for item {i} of {symbol}")

#             # Check for essential fields
#             title = item.get('title', 'No title available')
#             link = item.get('link', '#')
#             publisher = item.get('publisher', 'Unknown source')
#             summary = item.get('summary', 'No summary available') # yfinance might not have summary

#             if title == 'No title available' or link == '#':
#                 app.logger.warning(f"Skipping news item {i} for {symbol} due to missing title or link.")
#                 continue
            
#             # yfinance doesn't provide sentiment, so we'll set it to neutral
#             news_items.append({
#                 'title': title,
#                 'summary': summary,
#                 'url': link,
#                 'source': publisher,
#                 'published_date': published_date,
#                 'sentiment': 'neutral' # Set sentiment to neutral as yfinance doesn't provide it
#             })
            
#         app.logger.info(f"Finished processing news for {symbol}. Returning {len(news_items)} items.")
#         return news_items
#     except Exception as e:
#         app.logger.error(f"Error fetching yfinance news for {symbol}: {str(e)}", exc_info=True) # Log traceback
#         return None

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
    stock_info = check_for_stock_info(query)
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
        "industry outlook"
    ]
    return any(keyword in query for keyword in keywords)

# Function to fetch economic indicators (mocked for now)
def get_economic_indicators():
    # In a real implementation, fetch from an API or database
    indicators = """
    - U.S. unemployment rate: 3.7%
    - U.S. inflation rate (CPI): 4.2%
    - Federal Reserve interest rate: 5.25%
    - GDP growth rate (Q1 2024): 2.1%
    - Consumer confidence index: 98.5
    """
    return indicators

# Function to fetch recent policy shifts (mocked for now)
def get_policy_shifts():
    # In a real implementation, fetch from news or government sources
    policy_shifts = """
    - Recent U.S. tariffs on imported steel and aluminum increased by 10%
    - New trade agreements signed with several Asian countries
    - Proposed changes to corporate tax rates under review
    - Environmental regulations tightened for manufacturing sector
    """
    return policy_shifts

# Function to gather market data for enhanced analysis
def get_market_data_for_analysis():
    market_data = {}
    
    # Get data for key indices
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Russell 2000": "^RUT"
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
def get_sector_performance():
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

# Enhanced function to call OpenAI API with natural language market analysis support
def get_openai_response(user_message):
    global openai_api_available
    
    # First, check if this is a recommendation request
    if is_recommendation_request(user_message):
        symbol = extract_symbol_from_recommendation(user_message)
        if symbol:
            app.logger.info(f"Identified recommendation request for symbol: {symbol}")
            # Get stock info with recommendation
            stock_info = get_specific_stock_info(symbol, include_recommendation=True)
            if stock_info:
                return stock_info
    
    # Check if this is a natural language market analysis request
    if is_natural_language_market_analysis(user_message):
        app.logger.info("Detected natural language market analysis request")
        
        # Gather comprehensive market data
        economic_data = get_economic_indicators()
        policy_data = get_policy_shifts()
        market_indices = get_market_data_for_analysis()
        sectors = get_sector_performance()
        
        # Create market indices summaries
        indices_summary = ""
        for name, data in market_indices.items():
            direction = "up" if data['change'] >= 0 else "down"
            emoji = "📈" if data['change'] >= 0 else "📉"
            indices_summary += f"{emoji} {name}: {data['price']:.2f} ({direction} {abs(data['percent_change']):.2f}%)\n"
        
        # Create sector performance summary
        sector_summary = ""
        for sector, data in sectors.items():
            emoji = "🟢" if data['direction'] == "up" else "🔴"
            sector_summary += f"{emoji} {sector}: {data['change']:.2f}% ({data['strength']} {data['direction']})\n"
        
        # Create mock market news summaries since we've removed the news functionality
        news_summaries = """
        - S&P 500 gains momentum as tech sector rallies
        - Federal Reserve signals cautious approach to interest rates
        - Global markets respond to economic data releases
        - Earnings season shows mixed results across industries
        - Renewable energy stocks see increased investor interest
        """
        
        # Build enhanced system message with comprehensive market context
        system_message = (
            "You are an AI assistant specialized in market analysis and financial trends. "
            "You provide detailed, professional insights based on current market data. "
            "Use the following data to provide detailed insights in response to the user's query:\n\n"
            f"## Market Indices\n{indices_summary}\n"
            f"## Sector Performance\n{sector_summary}\n"
            f"## Market News Headlines\n{news_summaries}\n"
            f"## Economic Indicators\n{economic_data}\n"
            f"## Policy Shifts\n{policy_data}\n\n"
            "Provide a clear, concise, and insightful analysis based on the user's query. "
            "Include specific data points where relevant, and present a balanced view of market conditions. "
            "Conclude with potential implications for investors, but avoid making specific investment recommendations."
        )
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 400
        }
        
        url = f"{openai_api_base}/chat/completions"
        
        # Headers for project-based API keys (sk-proj-*)
        if api_key.startswith('sk-proj-'):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        
        try:
            app.logger.info(f"Calling OpenAI API for market analysis with URL: {url}")
            response = requests.post(url, headers=headers, json=data, timeout=15)
            
            app.logger.info(f"OpenAI API response status: {response.status_code}")
            
            if response.status_code != 200:
                app.logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                openai_api_available = False
                return generate_local_response(user_message)
            
            openai_api_available = True
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            app.logger.error(f"OpenAI API connection error: {str(e)}")
            openai_api_available = False
            return generate_local_response(user_message)
        except Exception as e:
            app.logger.error(f"OpenAI API unexpected error: {str(e)}")
            openai_api_available = False
            return generate_local_response(user_message)
    
    # If not a recommendation or market analysis request,
    # proceed with general stock info checking
    stock_info = check_for_stock_info(user_message)
    if stock_info:
        app.logger.info("Found direct stock information, returning without calling OpenAI")
        return stock_info
    
    # If API was previously marked as unavailable, check if we should retry
    if not openai_api_available:
        app.logger.info("OpenAI API was previously unavailable, using local response")
        return generate_local_response(user_message)
    
    url = f"{openai_api_base}/chat/completions"
    
    # Headers for project-based API keys (sk-proj-*)
    if api_key.startswith('sk-proj-'):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    # Add stock information to the prompt if it's about a specific stock
    stock_info = check_for_stock_info(user_message)
    system_message = "You are an AI assistant specialized in Indian and global stock markets, investments, and financial advice. Provide helpful, concise, and accurate information."
    
    if stock_info:
        system_message += f" Here is some recent data about the requested stock: {stock_info}"
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 300
    }
    
    try:
        app.logger.info(f"Calling OpenAI API with URL: {url}")
        app.logger.info(f"Using headers: {headers}")
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        # Log the response status code and detailed response for debugging
        app.logger.info(f"OpenAI API response status: {response.status_code}")
        
        if response.status_code != 200:
            app.logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            
            # Check for specific error types
            error_text = response.text.lower()
            if response.status_code == 401 and ("invalid" in error_text and "api key" in error_text):
                app.logger.error("Authentication failed: Invalid API key format or value")
                openai_api_available = False
                return "I'm having trouble with my AI capabilities right now, but I can still provide you with stock information. What stock would you like to know about?"
            
            openai_api_available = False
            return generate_local_response(user_message)
        
        # If we got here, the API is available
        openai_api_available = True
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        app.logger.error(f"OpenAI API connection error: {str(e)}")
        openai_api_available = False
        return generate_local_response(user_message)
    except Exception as e:
        app.logger.error(f"OpenAI API unexpected error: {str(e)}")
        openai_api_available = False
        return generate_local_response(user_message)

# Endpoint for chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        app.logger.info(f"Received chat request: {user_input}")
        
        # Use our response function which handles both OpenAI and local fallback
        reply = get_openai_response(user_input)
        
        # Always return a success response - the fallback mechanism handled any API errors
        return jsonify(reply)
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"Unexpected error in chat endpoint: {error_msg}")
        return jsonify("I'm currently experiencing technical difficulties. Please try again."), 500

# Diagnostic endpoint to check API status
@app.route('/api-status', methods=['GET'])
def api_status():
    try:
        # Simple test call to the OpenAI API
        url = f"{openai_api_base}/models"
        
        # Use the appropriate headers based on API key format
        if api_key.startswith('sk-proj-'):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                # Add organization ID if needed
            }
        else:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
        app.logger.info(f"Testing API with headers: {headers}")
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            return jsonify({"status": "operational", "code": response.status_code})
        else:
            # Return detailed error information
            error_info = {
                "status": "error", 
                "code": response.status_code, 
                "message": response.text
            }
            
            # If it's an authentication error, add a hint
            if response.status_code == 401:
                error_info["hint"] = "This appears to be an authentication error. Your API key might be invalid or have the wrong format."
                
            return jsonify(error_info)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# User-friendly API status page
@app.route('/status', methods=['GET'])
def status_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot API Status</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .status-box { padding: 20px; margin: 20px 0; border-radius: 5px; }
            .operational { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .hint { margin-top: 10px; padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Chatbot API Status</h1>
        <div id="status-container">Loading status...</div>
        
        <script>
            fetch('/api-status')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('status-container');
                    if (data.status === 'operational') {
                        container.innerHTML = `
                            <div class="status-box operational">
                                <h2>✅ API is operational</h2>
                                <p>Status code: ${data.code}</p>
                            </div>
                        `;
                    } else {
                        let html = `
                            <div class="status-box error">
                                <h2>❌ API error</h2>
                                <p>Status code: ${data.code}</p>
                                <p>Error details:</p>
                                <pre>${data.message}</pre>
                            </div>
                        `;
                        
                        if (data.hint) {
                            html += `<div class="hint"><strong>Hint:</strong> ${data.hint}</div>`;
                        }
                        
                        html += `
                            <h2>Using the Chatbot Without OpenAI</h2>
                            <p>Your chatbot will still work for stock-related queries, as it uses Yahoo Finance for that data.</p>
                            <p>Try asking questions about stock prices, like:</p>
                            <ul>
                                <li>"What's the price of AAPL?"</li>
                                <li>"How much is Tesla stock worth?"</li>
                                <li>"Tell me about Microsoft"</li>
                                <li>"RELIANCE.NS"</li>
                            </ul>
                        `;
                        
                        container.innerHTML = html;
                    }
                })
                .catch(error => {
                    document.getElementById('status-container').innerHTML = `
                        <div class="status-box error">
                            <h2>❌ Connection error</h2>
                            <p>Could not connect to the status endpoint: ${error.message}</p>
                        </div>
                    `;
                });
        </script>
    </body>
    </html>
    """
    return html

# Add a new endpoint for direct stock recommendations
@app.route('/recommend-stock', methods=['POST'])
def recommend_stock():
    try:
        symbol = request.json.get('symbol', '').strip()
        if not symbol:
            return jsonify({"error": "Please provide a stock symbol"}), 400
            
        # Try to get recommendation
        stock_info = get_specific_stock_info(symbol, include_recommendation=True)
        if stock_info:
            return jsonify({"result": stock_info})
        else:
            # Try common variations as you do in search-stock
            return jsonify({"error": f"Could not find stock information for '{symbol}'. Try adding an exchange extension (like .NS for Indian stocks) or check if the symbol is correct."}), 404
    except Exception as e:
        app.logger.error(f"Error in stock recommendation: {str(e)}")
        return jsonify({"error": "An error occurred while analyzing the stock"}), 500

# Endpoint to get news for a specific stock
# @app.route('/stock-news', methods=['GET'])
# def stock_news():
#     """Endpoint to get news for a specific stock symbol"""
#     try:
#         symbol = request.args.get('symbol')
#         limit = int(request.args.get('limit', 3))
#         
#         if not symbol:
#             return jsonify({"error": "No stock symbol provided"}), 400
#             
#         # News functionality has been removed
#         return jsonify({"news": []}), 404
#     except Exception as e:
#         app.logger.error(f"Error in news endpoint: {str(e)}")
#         return jsonify({"error": "An error occurred while fetching news"}), 500

# Function to check for stock information in the query
def check_for_stock_info(query):
    query = query.lower()
    
    # Check if this is a recommendation request
    want_recommendation = is_recommendation_request(query)
    
    # Define common patterns for stock price queries
    price_patterns = [
        r'(?:price|value|worth|stock|how (?:much|is)|what\'s|whats|what is).+?([a-zA-Z0-9&\^\._ ]+)(?:\s|$|stock|price|\?)',
        r'(?:tell|information|info|details|data).+?(?:about|on|for) ([a-zA-Z0-9&\^\._ ]+)(?:\s|$|\?)',
        r'(^[a-zA-Z0-9\.]+$)',  # Single word that might be a stock symbol
        r'search (?:for )?([a-zA-Z0-9&\^\._ ]+)'  # Direct search request
    ]
    
    # Add recommendation patterns if this is a recommendation request
    if want_recommendation:
        price_patterns.extend([
            r'(?:should i (?:buy|invest in|sell)|recommend) ([a-zA-Z0-9&\^\._ ]+)',
            r'(?:is|think about) ([a-zA-Z0-9&\^\._ ]+) (?:a good investment|worth buying|good stock)'
        ])
    
    # Try to extract potential stock symbols
    potential_symbols = []
    
    # Check all patterns
    for pattern in price_patterns:
        matches = re.findall(pattern, query)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    for group in match:
                        if group and len(group) > 0:
                            potential_symbols.append(group)
                else:
                    if match and len(match) > 0:
                        potential_symbols.append(match)
    
    # Add specific handling for exact matches on known stock symbols
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
    
    # Check if the query directly contains a known stock symbol
    for ticker in known_stocks.keys():
        if ticker.lower() in query.lower():
            app.logger.info(f"Found direct match for known stock: {ticker}")
            stock_info = get_specific_stock_info(ticker)
            if stock_info:
                return stock_info
    
    # Extract any potential stock symbols (words that look like tickers)
    words = re.findall(r'\b[A-Za-z0-9\.]+\b', query)
    for word in words:
        if (len(word) >= 2 and len(word) <= 10) or '.' in word:
            potential_symbols.append(word)
    
    # Common stock mapping for well-known companies
    stock_mapping = {
        "apple": "AAPL",
        "microsoft": "MSFT", 
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "meta": "META", 
        "facebook": "META",
        "netflix": "NFLX",
        "nvidia": "NVDA",
        "hdfc bank": "HDFCBANK.NS",
        "reliance": "RELIANCE.NS",
        "tata": "TCS.NS",
        "infosys": "INFY.NS",
        "state bank": "SBIN.NS",
        "sbi": "SBIN.NS",
        "airtel": "BHARTIARTL.NS",
        "airtel partly paid": "AIRTELPP.NS",
        "bharti airtel": "BHARTIARTL.NS",
        "bharti airtel partly paid": "AIRTELPP.NS",
        "s&p": "^GSPC",
        "s&p 500": "^GSPC",
        "dow": "^DJI",
        "dow jones": "^DJI",
        "nasdaq": "^IXIC",
        "nifty": "^NSEI",
        "nifty 50": "^NSEI",
        "sensex": "^BSESN"
    }
    
    # Remove duplicates and sort by length (shorter symbols are more likely to be valid tickers)
    potential_symbols = list(set(potential_symbols))
    potential_symbols.sort(key=len)
    
    app.logger.info(f"Potential stock symbols extracted: {potential_symbols}")
    
    # Clean up symbols and try them
    for symbol in potential_symbols:
        # Clean the symbol
        clean_symbol = symbol.strip().upper()
        clean_symbol = re.sub(r'(STOCK|PRICE|\?|OF|THE|FOR)$', '', clean_symbol).strip()
        
        # Skip very common words that might be extracted but aren't likely to be stocks
        if clean_symbol.lower() in ['a', 'the', 'and', 'or', 'of', 'to', 'for', 'in', 'on', 'by', 'all', 'any']:
            continue
            
        # Check if this matches known mappings
        symbol_key = clean_symbol.lower()
        if symbol_key in stock_mapping:
            ticker_to_try = stock_mapping[symbol_key]
            app.logger.info(f"Found mapping for {clean_symbol}: {ticker_to_try}")
            
            # Try to get stock info with recommendation if requested
            stock_info = get_specific_stock_info(ticker_to_try, include_recommendation=want_recommendation)
            if stock_info:
                return stock_info
        
        # Try different exchange variations
        exchange_variations = [
            '',         # Raw symbol
            '.NS',      # India NSE
            '.BO',      # India BSE
            '.L',       # London
            '.PA',      # Euronext Paris
            '.F',       # Frankfurt
            '.TO',      # Toronto
            '.AX',      # Australian Securities Exchange
            '.SA',      # Sao Paulo Stock Exchange
            '.T',       # Tokyo Stock Exchange
            '.HK',      # Hong Kong Stock Exchange
            '.SZ',      # Shenzhen Stock Exchange
            '.SS',      # Shanghai Stock Exchange
            '.SW',      # Swiss Exchange
            '-USD',     # Cryptocurrency format
            '.SI',      # Singapore Exchange
            '.JK',      # Jakarta Stock Exchange
            '.KS',      # Korea Stock Exchange
            '.KL',      # Kuala Lumpur Stock Exchange
            '.V',       # TSX Venture Exchange (Canada)
            '.MX',      # Mexico Stock Exchange
            '.ST',      # Stockholm Stock Exchange
            '.CO',      # Copenhagen Stock Exchange
            '.OL',      # Oslo Stock Exchange
            '.NZ',      # New Zealand Stock Exchange
            '.BR',      # Brussels Stock Exchange
            '.MC',      # Madrid Stock Exchange
            '.AT',      # Athens Stock Exchange
            '.VI',      # Vienna Stock Exchange
        ]
        
        # Index prefix for major indices
        if len(clean_symbol) <= 5:
            exchange_variations.append('^' + clean_symbol)  # Index format like ^GSPC
        
        # Try the raw symbol first
        if '.' not in clean_symbol:
            stock_info = get_specific_stock_info(clean_symbol, include_recommendation=want_recommendation)
            if stock_info:
                return stock_info
                
        # Try various exchanges - only if the symbol doesn't already have a suffix
        if '.' not in clean_symbol and '-' not in clean_symbol:
            for exchange in exchange_variations:
                variation = clean_symbol + exchange
                app.logger.info(f"Trying stock symbol with exchange: {variation}")
                stock_info = get_specific_stock_info(variation, include_recommendation=want_recommendation)
                if stock_info:
                    return stock_info
                    
        # If the symbol already has a suffix (like .NS) try it directly
        elif '.' in clean_symbol:
            stock_info = get_specific_stock_info(clean_symbol, include_recommendation=want_recommendation)
            if stock_info:
                return stock_info
                
        # Try removing spaces (for multi-word names)
        if ' ' in clean_symbol:
            no_spaces = clean_symbol.replace(' ', '')
            stock_info = get_specific_stock_info(no_spaces, include_recommendation=want_recommendation)
            if stock_info:
                return stock_info
                
            # Try with exchanges
            for exchange in exchange_variations:
                if exchange:  # Skip empty exchange
                    variation = no_spaces + exchange
                    app.logger.info(f"Trying no-space symbol with exchange: {variation}")
                    stock_info = get_specific_stock_info(variation, include_recommendation=want_recommendation)
                    if stock_info:
                        return stock_info
    
    return None

# Endpoint for direct stock search
@app.route('/search-stock', methods=['POST'])
def search_stock():
    try:
        symbol = request.json.get('symbol', '').strip()
        if not symbol:
            return jsonify({"error": "Please provide a stock symbol"}), 400
            
        # Try to get info
        stock_info = get_specific_stock_info(symbol)
        if stock_info:
            return jsonify({"result": stock_info})
        else:
            # Try with common exchange extensions
            exchange_extensions = [
                '.NS',      # India NSE
                '.BO',      # India BSE
                '.L',       # London
                '.PA',      # Paris
                '.F',       # Frankfurt
                '.TO',      # Toronto
                '^'         # Index prefix
            ]
            
            for ext in exchange_extensions:
                if '.' not in symbol and '^' not in symbol:  # Only if it doesn't already have an extension
                    var = symbol + ext if ext != '^' else '^' + symbol
                    info = get_specific_stock_info(var)
                    if info:
                        return jsonify({"result": info})
                        
            return jsonify({"error": f"Could not find stock information for '{symbol}'. Try adding an exchange extension (like .NS for Indian stocks) or check if the symbol is correct."}), 404
    except Exception as e:
        app.logger.error(f"Error in stock search: {str(e)}")
        return jsonify({"error": "An error occurred while searching for the stock"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
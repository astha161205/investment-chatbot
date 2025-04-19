import yfinance as yf
import requests
import json
from flask import Flask, request, jsonify, send_from_directory
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
api_key = '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'

# Configure OpenAI API base URL - this is important for project-based keys
openai_api_base = "https://api.openai.com/v1"

# Track API availability
openai_api_available = True

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

# Function to check for stock information in the query
def check_for_stock_info(query):
    query = query.lower()
    
    # Define common patterns for stock price queries
    price_patterns = [
        r'(?:price|value|worth|stock|how (?:much|is)|what\'s|whats|what is).+?([a-zA-Z0-9&\^\._ ]+)(?:\s|$|stock|price|\?)',
        r'(?:tell|information|info|details|data).+?(?:about|on|for) ([a-zA-Z0-9&\^\._ ]+)(?:\s|$|\?)',
        r'(^[a-zA-Z0-9]+$)',  # Single word that might be a stock symbol
        r'search (?:for )?([a-zA-Z0-9&\^\._ ]+)'  # Direct search request
    ]
    
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
    
    # Extract any potential stock symbols (words that look like tickers)
    # This helps catch direct mentions of tickers like "AAPL" or "RELIANCE.NS"
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
            
        # Check if this matches known mappings (case insensitive)
        symbol_key = clean_symbol.lower()
        if symbol_key in stock_mapping:
            ticker_to_try = stock_mapping[symbol_key]
            app.logger.info(f"Found mapping for {clean_symbol}: {ticker_to_try}")
            
            # Try to get stock info
            stock_info = get_specific_stock_info(ticker_to_try)
            if stock_info:
                return stock_info
        
        # Define common exchanges and variations to try
        # This covers most major global markets
        exchange_variations = [
            '',         # Raw symbol (e.g., AAPL)
            '.NS',      # India - National Stock Exchange
            '.BO',      # India - Bombay Stock Exchange
            '.L',       # London Stock Exchange
            '.PA',      # Euronext Paris
            '.F',       # Frankfurt Stock Exchange
            '.TO',      # Toronto Stock Exchange
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
        
        # Try the raw symbol first (highest priority)
        if '.' not in clean_symbol:  # Only if it doesn't already have an exchange suffix
            stock_info = get_specific_stock_info(clean_symbol)
            if stock_info:
                return stock_info
            
        # Try various exchanges - only if the symbol doesn't already have a suffix
        if '.' not in clean_symbol and '-' not in clean_symbol:
            for exchange in exchange_variations:
                variation = clean_symbol + exchange
                app.logger.info(f"Trying stock symbol with exchange: {variation}")
                stock_info = get_specific_stock_info(variation)
                if stock_info:
                    return stock_info
                    
        # If the symbol already has a suffix (like .NS) try it directly
        elif '.' in clean_symbol:
            stock_info = get_specific_stock_info(clean_symbol)
            if stock_info:
                return stock_info
                
        # Try removing spaces (for multi-word names)
        if ' ' in clean_symbol:
            no_spaces = clean_symbol.replace(' ', '')
            stock_info = get_specific_stock_info(no_spaces)
            if stock_info:
                return stock_info
                
            # Try with exchanges
            for exchange in exchange_variations:
                if exchange:  # Skip empty exchange
                    variation = no_spaces + exchange
                    app.logger.info(f"Trying no-space symbol with exchange: {variation}")
                    stock_info = get_specific_stock_info(variation)
                    if stock_info:
                        return stock_info
    
    return None

# Function to get information for a specific stock symbol
def get_specific_stock_info(symbol):
    try:
        app.logger.info(f"Fetching data for stock symbol: {symbol}")
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        
        if data.empty:
            app.logger.info(f"No data found for {symbol}")
            return None
            
        # Get basic info
        info = {}
        try:
            info = stock.info
        except Exception as e:
            app.logger.warning(f"Could not get additional info for {symbol}: {str(e)}")
            
        # Format the stock name
        stock_name = symbol
        if info and 'longName' in info and info['longName']:
            stock_name = info['longName']
        elif info and 'shortName' in info and info['shortName']:
            stock_name = info['shortName']
            
        # Get the current price
        last_price = data['Close'].iloc[-1]
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
                
        app.logger.info(f"Successfully retrieved info for {symbol}")
        return result
    except Exception as e:
        app.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
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

# Function to generate local responses with improved stock information
def generate_local_response(query):
    query = query.lower()
    
    # Try to find stock information first
    stock_info = check_for_stock_info(query)
    if stock_info:
        return stock_info
    
    # Check for greetings
    if any(word in query for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm your stock market assistant. You can ask me about any stock price by mentioning the stock symbol (like AAPL) or company name (like Apple)."
    
    # Check for stock search related queries
    if any(word in query for word in ['search', 'find', 'lookup', 'look up']):
        return "To look up a stock, simply mention its symbol (like AAPL, MSFT) or company name (like Apple, Microsoft). For Indian stocks, you can mention the company name or use the exchange suffix (like RELIANCE.NS or INFY.NS)."
    
    # Investment advice
    if any(word in query for word in ['invest', 'investment', 'portfolio', 'strategy']):
        return "As a financial assistant, I recommend diversifying your portfolio across different asset classes based on your risk tolerance and investment horizon. Consider consulting with a financial advisor for personalized advice."
    
    # Market trends
    if any(word in query for word in ['market', 'trend', 'bullish', 'bearish']):
        return "Market trends vary by sector and time period. It's important to consider economic indicators, company fundamentals, and global events when analyzing market conditions."
    
    # Default response with stock lookup instruction
    return "I can help you find information about any publicly traded stock. Just mention the company name or stock symbol (e.g., 'What's the price of AAPL?' or 'Tell me about Microsoft')."

# Function to call OpenAI API with better error handling
def get_openai_response(user_message):
    global openai_api_available
    
    # Check if this might be a direct stock query first
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
            # The line below is commented out as OpenAI often uses the Authorization header alone
            # "OpenAI-Organization": "org-..." # Add your org ID if you have one
        }
    else:
        # Regular API key headers
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

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import re
import secrets
import requests
import functools
import time
from cachetools import TTLCache

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///investment_portfolio.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login_test'
login_manager.login_message = 'Please log in to access this feature.'

# API Keys
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

# Initialize APIs and caches
cg = CoinGeckoAPI()
crypto_cache = TTLCache(maxsize=100, ttl=300)  # Cache crypto data for 5 minutes
stock_cache = TTLCache(maxsize=100, ttl=300)   # Cache stock data for 5 minutes

def rate_limited(max_per_second):
    """Rate limiting decorator"""
    min_interval = 1.0 / float(max_per_second)
    def decorator(func):
        last_time_called = [0.0]
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limited(0.5)  # Limit to 1 request per 2 seconds for CoinGecko
def get_crypto_data(query):
    """Fetch cryptocurrency data using CoinGecko API with caching"""
    try:
        # Check cache first
        cache_key = f"crypto_{query.lower()}"
        if cache_key in crypto_cache:
            return crypto_cache[cache_key]

        print(f"Attempting to fetch crypto data for: {query}")
        query = query.strip().lower()
        
        crypto_map = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'usdt': 'tether',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'ada': 'cardano',
            'doge': 'dogecoin',
            'sol': 'solana'
        }
        
        if query in crypto_map:
            query = crypto_map[query]
            
        try:
            data = cg.get_price(ids=query, vs_currencies='usd', include_24hr_change=True, include_24hr_vol=True)
            if data and query in data:
                price_data = data[query]
                result = {
                    'price': price_data.get('usd', 0),
                    'change_percent': price_data.get('usd_24h_change', 0),
                    'volume': price_data.get('usd_24h_vol', 0),
                    'name': query.capitalize(),
                    'symbol': query.upper(),
                    'type': 'crypto'
                }
                crypto_cache[cache_key] = result
                return result
        except Exception as e:
            print(f"Direct crypto lookup failed: {str(e)}")
            
        # Try simple list lookup instead of search
        coins = cg.get_coins_list()
        matching_coins = [coin for coin in coins if coin['symbol'].lower() == query or coin['id'].lower() == query]
        
        if matching_coins:
            coin = matching_coins[0]
            data = cg.get_price(ids=coin['id'], vs_currencies='usd', include_24hr_change=True, include_24hr_vol=True)
            if data and coin['id'] in data:
                price_data = data[coin['id']]
                result = {
                    'price': price_data.get('usd', 0),
                    'change_percent': price_data.get('usd_24h_change', 0),
                    'volume': price_data.get('usd_24h_vol', 0),
                    'name': coin['name'],
                    'symbol': coin['symbol'].upper(),
                    'type': 'crypto'
                }
                crypto_cache[cache_key] = result
                return result
                
    except Exception as e:
        print(f"Crypto error: {str(e)}")
    
    print(f"Could not find crypto data for {query}")
    return None

def get_stock_data(query):
    """Fetch stock data using Alpha Vantage API with yfinance fallback and caching"""
    try:
        # Check cache first
        cache_key = f"stock_{query.upper()}"
        if cache_key in stock_cache:
            return stock_cache[cache_key]

        print(f"Attempting to fetch data for query: {query}")
        query = query.strip().upper()
        
        # Try yfinance first
        try:
            print(f"Attempting yfinance lookup for: {query}")
            stock = yf.Ticker(query)
            info = stock.info
            print(f"yfinance info received: {info is not None}")
            
            if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                price_data = {
                    'price': info.get('regularMarketPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'high': info.get('dayHigh', 0),
                    'low': info.get('dayLow', 0),
                    'name': info.get('longName', query),
                    'symbol': query,
                    'type': 'stock'
                }
                stock_cache[cache_key] = price_data
                print(f"Successfully retrieved yfinance data: {price_data}")
                return price_data
            else:
                print("yfinance data was incomplete or invalid")
        except Exception as e:
            print(f"yfinance lookup failed: {str(e)}")

        # If yfinance fails, try Alpha Vantage as backup
        # Remove exchange prefixes if present
        query = query.replace('NYSE:', '').replace('NASDAQ:', '')
        
        # Alpha Vantage API endpoint
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={query}&apikey={ALPHA_VANTAGE_KEY}'
        
        print(f"Making API request to Alpha Vantage...")
        response = requests.get(url)
        print(f"API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error Response: {response.text}")
            raise Exception(f"API request failed with status code: {response.status_code}")
            
        data = response.json()
        print(f"API Response: {data}")

        # Check for rate limit message
        if "Note" in data:
            if "API call frequency" in data["Note"]:
                raise Exception("API rate limit reached. Please wait a few minutes before trying again.")
            elif "API call volume" in data["Note"]:
                raise Exception("Daily API call limit reached. Please try again tomorrow.")
        
        if 'Global Quote' in data and data['Global Quote']:
            quote = data['Global Quote']
            if not quote.get('05. price'):
                print(f"No price data found in Alpha Vantage response for {query}")
                return None
                
            price_data = {
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': float(quote.get('10. change percent', '0').replace('%', '')),
                'volume': int(quote.get('06. volume', 0)),
                'name': query,
                'symbol': query,
                'type': 'stock'
            }
            stock_cache[cache_key] = price_data
            print(f"Successfully retrieved Alpha Vantage data: {price_data}")
            return price_data
        
        print(f"Could not find stock data for {query}")
        return None
        
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None

def get_market_updates():
    """Fetch latest market updates and news"""
    try:
        # Get market indices to check market leaders
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^NSEI': 'Nifty 50',
            '^BSESN': 'Sensex'
        }
        
        market_summary = []
        
        # Get indices performance
        for symbol, name in indices.items():
            try:
                index = yf.Ticker(symbol)
                info = index.info
                if info and 'regularMarketPrice' in info:
                    market_summary.append({
                        'name': name,
                        'price': info['regularMarketPrice'],
                        'change': info['regularMarketChange'],
                        'change_percent': info['regularMarketChangePercent']
                    })
            except Exception as e:
                print(f"Error fetching {name}: {str(e)}")

        # Get top gaining and losing stocks from S&P 500
        sp500 = yf.Ticker('^GSPC')
        sp500_components = sp500.info.get('components', [])[:50]  # Get first 50 components
        
        stocks_data = []
        for symbol in sp500_components:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                if info and 'regularMarketPrice' in info:
                    stocks_data.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'price': info['regularMarketPrice'],
                        'change_percent': info['regularMarketChangePercent']
                    })
            except:
                continue

        # Sort to get top gainers and losers
        stocks_data.sort(key=lambda x: x['change_percent'], reverse=True)
        top_gainers = stocks_data[:5]
        top_losers = stocks_data[-5:]

        # Get latest market news
        news_items = []
        try:
            msft = yf.Ticker("MSFT")  # Using Microsoft as a proxy for market news
            news = msft.news
            for item in news[:10]:  # Get last 10 news items
                news_items.append({
                    'title': item['title'],
                    'publisher': item['publisher'],
                    'link': item['link'],
                    'published': datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')
                })
        except Exception as e:
            print(f"Error fetching news: {str(e)}")

        return {
            'market_summary': market_summary,
            'top_gainers': top_gainers,
            'top_losers': top_losers,
            'news': news_items
        }
    except Exception as e:
        print(f"Error in get_market_updates: {str(e)}")
        return None

@app.route('/api/market-updates')
def market_updates():
    """API endpoint for market updates"""
    updates = get_market_updates()
    if updates:
        return jsonify(updates)
    return jsonify({'error': 'Could not fetch market updates'}), 500

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    investments = db.relationship('Investment', backref='user', lazy=True)

class Investment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    shares = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.DateTime, nullable=False)
    asset_type = db.Column(db.String(10), default='stock')  # 'stock' or 'crypto'

    def get_current_value(self):
        if self.asset_type == 'crypto':
            data = get_crypto_data(self.symbol.lower())
        else:
            data = get_stock_data(self.symbol)
        if data:
            return data['price'] * self.shares
        return 0

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/login-test')
def login_test():
    """Create test user if it doesn't exist and log in"""
    try:
        if not User.query.filter_by(username='test').first():
            test_user = User(username='test', password='test')
            db.session.add(test_user)
            db.session.commit()
            print("Created test user")
        
        user = User.query.filter_by(username='test').first()
        login_user(user)
        print("Logged in test user")
        return jsonify({'message': 'Successfully logged in as test user'})
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Failed to log in'}), 500

@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    try:
        investments = current_user.investments
        portfolio_data = {
            'total_value': sum(inv.get_current_value() for inv in investments),
            'investments': [{
                'symbol': inv.symbol,
                'shares': inv.shares,
                'current_value': inv.get_current_value(),
                'purchase_price': inv.purchase_price,
                'profit_loss': inv.get_current_value() - (inv.purchase_price * inv.shares)
            } for inv in investments]
        }
        return jsonify(portfolio_data)
    except Exception as e:
        print(f"Portfolio error: {str(e)}")
        return jsonify({'error': 'Failed to fetch portfolio data'}), 500

@app.route('/api/add_investment', methods=['POST'])
@login_required
def add_investment():
    try:
        data = request.json
        symbol = data.get('symbol', '').strip().upper()
        shares = float(data.get('shares', 0))
        purchase_price = float(data.get('purchase_price', 0))
        asset_type = data.get('asset_type', 'stock')  # 'stock' or 'crypto'
        
        if not symbol or shares <= 0 or purchase_price <= 0:
            return jsonify({'error': 'Invalid investment details'}), 400
            
        # Verify the asset exists and get its current price
        if asset_type == 'crypto':
            asset_data = get_crypto_data(symbol.lower())
        else:
            asset_data = get_stock_data(symbol)
            
        if not asset_data:
            return jsonify({'error': f'Could not find {asset_type} with symbol {symbol}'}), 404
            
        # Create new investment
        investment = Investment(
            user_id=current_user.id,
            symbol=symbol,
            shares=shares,
            purchase_price=purchase_price,
            purchase_date=datetime.now(),
            asset_type=asset_type
        )
        
        db.session.add(investment)
        db.session.commit()
        
        return jsonify({
            'message': f'Successfully added {shares} {symbol} to your portfolio',
            'investment': {
                'symbol': investment.symbol,
                'shares': investment.shares,
                'purchase_price': investment.purchase_price,
                'current_value': investment.get_current_value()
            }
        })
        
    except Exception as e:
        print(f"Error adding investment: {str(e)}")
        return jsonify({'error': 'Failed to add investment'}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    try:
        message = request.json.get('message', '').lower()
        print(f"Received chat message: {message}")
        
        # Handle different types of queries
        if 'add' in message:
            print("Processing add investment request")
            # Extract investment details
            words = message.split()
            try:
                shares = None
                for i, word in enumerate(words):
                    try:
                        shares = float(word)
                        print(f"Found shares amount: {shares}")
                        break
                    except ValueError:
                        continue
                
                if not shares:
                    print("No shares amount found in message")
                    return jsonify({
                        'response': '''Please specify the number of shares/units to add. Examples:
- "Add 10 shares of AAPL"
- "Add 0.5 BTC"
- "Add 100 shares of MSFT"
- "Add 2 ETH"'''
                    })
                
                # Find stock symbol
                symbol = None
                for word in words:
                    if word.isupper() or (word.startswith('$') and word[1:].isupper()):
                        symbol = word.replace('$', '')
                        print(f"Found symbol: {symbol}")
                        break
                
                if not symbol:
                    # Try to find common crypto symbols
                    for word in words:
                        if word.lower() in ['eth', 'btc', 'usdt', 'bnb', 'xrp', 'ada', 'doge', 'sol']:
                            symbol = word.upper()
                            print(f"Found crypto symbol: {symbol}")
                            break
                
                if not symbol:
                    print("No symbol found in message")
                    return jsonify({
                        'response': '''Please specify the stock symbol or cryptocurrency. Examples:
- "Add 10 shares of AAPL"
- "Add 0.5 BTC"
- "Add 100 shares of MSFT"
- "Add 2 ETH"'''
                    })
                
                # Determine asset type
                asset_type = 'crypto' if any(word in message.lower() for word in ['crypto', 'bitcoin', 'eth', 'btc']) or symbol.upper() in ['ETH', 'BTC', 'USDT', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL'] else 'stock'
                print(f"Determined asset type: {asset_type}")
                
                # Get current price
                if asset_type == 'crypto':
                    print(f"Fetching crypto data for {symbol}")
                    asset_data = get_crypto_data(symbol.lower())
                else:
                    print(f"Fetching stock data for {symbol}")
                    asset_data = get_stock_data(symbol)
                
                print(f"Asset data received: {asset_data}")
                
                if not asset_data:
                    return jsonify({
                        'response': f'Could not find {asset_type} with symbol {symbol}. Please check the symbol and try again.'
                    })
                
                # Create investment
                investment = Investment(
                    user_id=current_user.id,
                    symbol=symbol,
                    shares=shares,
                    purchase_price=asset_data['price'],
                    purchase_date=datetime.now(),
                    asset_type=asset_type
                )
                
                db.session.add(investment)
                db.session.commit()
                print(f"Added investment to database: {symbol}, {shares} shares")
                
                response = f"✅ Added {shares} {symbol} to your portfolio\n"
                response += f"Purchase Price: ${asset_data['price']:,.2f}\n"
                response += f"Total Value: ${(shares * asset_data['price']):,.2f}"
                return jsonify({'response': response})
                
            except Exception as e:
                print(f"Error processing add investment: {str(e)}")
                return jsonify({
                    'response': f'Error adding investment: {str(e)}. Please try again with the format:\n- "Add 10 shares of AAPL"\n- "Add 0.5 BTC"'
                })
        
        elif 'portfolio' in message or 'holdings' in message:
            investments = current_user.investments
            if not investments:
                return jsonify({
                    'response': '''Your portfolio is currently empty. You can add investments using commands like:
- "Add 10 shares of AAPL"
- "Add 0.5 BTC"
- "Add 100 shares of MSFT"
- "Add 2 ETH"'''
                })
            
            total_value = sum(inv.get_current_value() for inv in investments)
            response = f'Your total portfolio value is ${total_value:.2f}. Here are your holdings:\n'
            
            for inv in investments:
                current_value = inv.get_current_value()
                pl = current_value - (inv.purchase_price * inv.shares)
                asset_type = "Crypto" if inv.asset_type == 'crypto' else "Stock"
                response += f'\n{asset_type} {inv.symbol}: {inv.shares} units, Current Value: ${current_value:.2f}, P/L: ${pl:.2f}'
            
            return jsonify({'response': response})
        
        elif 'price' in message or 'stock' in message or 'show' in message or 'crypto' in message:
            # Extract the query (company name or symbol)
            words = message.split()
            query = None
            
            # Try to find the query after keywords
            keywords = ['of', 'for', 'me']
            for keyword in keywords:
                if keyword in words:
                    idx = words.index(keyword)
                    if idx + 1 < len(words):
                        query = ' '.join(words[idx + 1:])
                        break
            
            # If no keyword found, use the last word or phrase
            if not query:
                # Remove common words that shouldn't be part of the query
                ignore_words = {'price', 'stock', 'show', 'what', 'is', 'the', 'how', 'much', 'tell', 'about', 'get', 'crypto', 'of'}
                query_words = [w for w in words if w.lower() not in ignore_words]
                if query_words:
                    query = ' '.join(query_words)
            
            if not query:
                return jsonify({
                    'response': '''Please specify a company name, stock symbol, or cryptocurrency. Examples:
- "What's the price of Apple?" or "price of AAPL"
- "Show me Tesla stock"
- "How much is Bank of America?"
- "Price of JPM"

For cryptocurrencies:
- "Price of Bitcoin" or "BTC"
- "Show me Ethereum" or "ETH"
- "How much is Dogecoin?" or "DOGE"'''
                })
            
            print(f"Extracted query: {query}")
            
            # Clean up the query
            query = query.strip()
            
            # Common crypto symbols and keywords
            crypto_keywords = ['crypto', 'coin', 'bitcoin', 'ethereum', 'btc', 'eth', 'usdt', 'bnb', 'xrp', 'ada', 'doge', 'sol']
            
            # Try to get crypto data first if crypto-related query
            is_crypto = ('crypto' in message.lower() or 
                        any(word in query.lower() for word in crypto_keywords) or
                        any(word.lower() in crypto_keywords for word in query.split()))
            
            if is_crypto:
                print(f"Detected as crypto query: {query}")
                data = get_crypto_data(query)
                if data and data['price'] > 0:
                    response = f"🪙 {data['name']} ({data['symbol']})\n\n"
                    response += f"Current Price: ${data['price']:,.2f}\n"
                    response += f"24h Change: {data['change_percent']:+.2f}%\n"
                    response += f"24h Volume: ${data['volume']:,.2f}"
                    return jsonify({'response': response})
                else:
                    print(f"Crypto data not found for: {query}")
            
            # If not crypto or crypto lookup failed, try stock data
            # First try exact query
            data = get_stock_data(query)
            if not data and ' ' in query:
                # If space in query, try the last word as it might be the symbol
                symbol = query.split()[-1].upper()
                print(f"Trying last word as symbol: {symbol}")
                data = get_stock_data(symbol)
                
            if not data and len(query.split()) > 1:
                # If multiple words, try first word as it might be the symbol
                symbol = query.split()[0].upper()
                print(f"Trying first word as symbol: {symbol}")
                data = get_stock_data(symbol)
            
            if data and data['price'] > 0:
                response = f"📈 {data['name']} ({data['symbol']})\n\n"
                response += f"Current Price: ${data['price']:,.2f}\n"
                response += f"Change: {data['change_percent']:+.2f}% today\n"
                response += f"Volume: {data['volume']:,}"
                return jsonify({'response': response})
            else:
                suggestions = ""
                query_upper = query.upper()
                
                # Common symbol corrections
                corrections = {
                    'APPL': 'AAPL (Apple Inc.)',
                    'GOOGL': 'GOOG (Alphabet/Google)',
                    'GOOGLE': 'GOOG (Alphabet/Google)',
                    'AMAZON': 'AMZN (Amazon.com)',
                    'BITCOIN': 'BTC (Bitcoin)',
                    'ETHEREUM': 'ETH (Ethereum)',
                    'TESLA': 'TSLA (Tesla Inc.)',
                    'MICROSOFT': 'MSFT (Microsoft)',
                    'META': 'META (Meta Platforms/Facebook)',
                    'FACEBOOK': 'META (Meta Platforms)',
                    'TCS': 'TCS.NS (Tata Consultancy Services)',
                    'BHARTI': 'BHARTIARTL.NS (Bharti Airtel)',
                    'AIRTEL': 'BHARTIARTL.NS (Bharti Airtel)',
                    'RELIANCE': 'RELIANCE.NS (Reliance Industries)',
                    'INFOSYS': 'INFY.NS (Infosys Limited)'
                }
                
                for wrong, correct in corrections.items():
                    if wrong in query_upper:
                        suggestions = f"\n\nDid you mean {correct}?"
                        break
                
                return jsonify({
                    'response': f'Could not find data for "{query}". Please try:\n1. Using the exact symbol (e.g., AAPL for Apple, MSFT for Microsoft)\n2. Using the full company name\n3. For Indian stocks, add .NS (e.g., TCS.NS, RELIANCE.NS)\n4. For cryptocurrencies, use symbols like BTC, ETH, etc.{suggestions}'
                })
        
        elif 'help' in message:
            return jsonify({
                'response': '''I can help you with:

1. Getting stock prices:
   - "Price of AAPL"
   - "Show me MSFT"
   - For Indian stocks: "Price of TCS.NS"

2. Getting crypto prices:
   - "Price of BTC"
   - "Show me ETH"

3. Managing your portfolio:
   - "Show my portfolio"
   - "Add 10 shares of AAPL"
   - "Add 0.5 BTC"

4. Market updates:
   - "Show market updates"
   - "How are the markets doing?"'''
            })
        
        else:
            return jsonify({
                'response': '''I can help you with:
1. Getting stock/crypto prices (e.g., "Price of AAPL" or "Price of BTC")
2. Managing your portfolio (e.g., "Show portfolio" or "Add 10 shares of AAPL")
3. Market updates (e.g., "Show market updates")

Type "help" for more details.'''
            })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': 'Sorry, there was an error processing your request. Please try again.'
        })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

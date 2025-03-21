# investment-chatbot
An intelligent investment assistant that helps users manage their portfolio, track market updates, and make informed investment decisions. Built with Python, Flask, and modern web technologies.

## Features

- **Real-time Market Updates**: Track major market indices (S&P 500, Dow Jones, NASDAQ, Nifty 50, Sensex)
- **Live Market News**: Get the latest market news, top gainers/losers, and company updates
- **Portfolio Management**: Track your investments, monitor performance, and view profit/loss
- **AI-Powered Chat**: Get investment advice and market insights through natural conversation
- **User Authentication**: Secure login system to protect your investment data
- **Responsive Design**: Beautiful interface that works on both desktop and mobile
- **Dynamic Background**: Engaging video background for enhanced user experience

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite with SQLAlchemy
- **APIs**: 
  - Alpha Vantage (Market News & Data)
  - Yahoo Finance (Stock Data)
  - CoinGecko (Cryptocurrency Data)
- **Authentication**: Flask-Login
- **UI Framework**: Custom CSS with modern design patterns

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/investment-chatbot.git
cd investment-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
SECRET_KEY=your_secure_secret_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key
```

5. Initialize the database:
```bash
flask db upgrade
```

6. Run the application:
```bash
python app.py
```

Visit `http://localhost:5000` in your browser to use the application.

## Features in Detail

### Market Updates
- Real-time tracking of major market indices
- Latest market news and updates
- Top gainers and losers
- Company-specific news

### Portfolio Management
- Add and track investments
- Real-time portfolio valuation
- Profit/Loss tracking
- Investment history

### Chat Interface
- Natural language processing for investment queries
- Market analysis and insights
- Investment recommendations
- Portfolio management commands

## API Usage

The application uses several APIs:
- **Alpha Vantage**: For market news and stock data (5 calls/minute limit on free tier)
- **Yahoo Finance**: For real-time stock quotes and market data
- **CoinGecko**: For cryptocurrency data

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Alpha Vantage for providing market data
- Thanks to Yahoo Finance for stock information
- Thanks to CoinGecko for cryptocurrency data

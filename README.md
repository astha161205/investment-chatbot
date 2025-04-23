# Investment Assistant

A web-based chatbot application that provides real-time stock information and market analysis.

## Features

- Real-time stock data from Yahoo Finance
- Support for major US and Indian stocks
- Chat interface for natural language queries
- Market analysis for major indices
- Responsive design for desktop and mobile

## Setup Instructions

1. Clone the repository
   ```
   git clone <repository-url>
   cd investment-assistant
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix/MacOS
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the application
   ```
   python your_script_name.py
   ```

5. Open your browser and navigate to http://localhost:5000

## Usage Examples

- "What's the price of TSLA?"
- "Tell me about HDFC Bank"
- "How is the US market doing today?"
- "Give me information about AAPL"
- "What's the current price of MSFT?"

## Stock Symbol Format

- US Stocks: Use the ticker symbol (e.g., AAPL, MSFT, TSLA)
- Indian Stocks: Use the ticker symbol with .NS suffix (e.g., HDFCBANK.NS, ICICIBANK.NS)

## Supported Query Types

- Stock price information: "What's the price of AAPL?" or "Tell me about Tesla"
- Market analysis: "How is the market doing today?"
- Investment advice: "Should I invest in tech stocks?"

## Supported Stocks

- US stocks: AAPL (Apple), MSFT (Microsoft), TSLA (Tesla), etc.
- Indian stocks: HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, etc.



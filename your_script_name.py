import yfinance as yf
import openai
from flask import Flask, request, jsonify, render_template, send_from_directory
import os

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = 'sk-proj-896S4yljeM-YEs9n1HHYiTsDNqqpSUD6c3q1adZncx1fKKQx3JdfAsCmy-22FB2ays9t5T5kST3BlbkFJ_o2Pka0kDM1tiAH368fEqLq64lZitPAhBENmeG0jISYoXJWuuXEoerlwbnEIMNaDfi_9HI-QA'

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

# Endpoint for chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    # Using the updated OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in Indian and global stock markets, investments, and financial advice. Provide helpful, concise, and accurate information."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )
    
    # Extract the assistant's reply from the response
    reply = response.choices[0].message.content
    return jsonify(reply)

# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
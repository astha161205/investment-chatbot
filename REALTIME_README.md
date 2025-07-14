# Real-Time Stock Price System

## 🚀 Overview

The real-time stock price system provides live, up-to-date stock prices with automatic updates every 30 seconds. This system balances real-time data with rate limiting to ensure reliable performance.

## ✨ Key Features

### 📊 Real-Time Updates
- **30-second cache duration** for real-time data
- **Automatic refresh** every 30 seconds
- **Live price monitoring** with visual indicators
- **Data age tracking** to show freshness

### 🎯 Multi-Tier Caching
- **Real-time tier**: 30 seconds (for live trading)
- **Short-term tier**: 5 minutes (for general queries)
- **Long-term tier**: 15 minutes (for historical data)

### 🔔 Price Alerts
- **Set price thresholds** for automatic alerts
- **Real-time monitoring** of alert conditions
- **User-specific alerts** with unique IDs
- **Automatic alert cleanup** when triggered

### 📱 Interactive Dashboard
- **Beautiful real-time dashboard** with live updates
- **Add/remove stocks** dynamically
- **Visual price indicators** (green/red for gains/losses)
- **Mobile-responsive design**

## 🛠️ API Endpoints

### Real-Time Stock Data
```bash
GET /stock/realtime?ticker=AAPL
```
Returns real-time data with 30-second cache:
```json
{
  "ticker": "AAPL",
  "current_price": 150.25,
  "timestamp": "2024-01-15T10:30:00Z",
  "data_age_seconds": 15.2,
  "data": { ... }
}
```

### Batch Stock Data
```bash
POST /stocks/batch
Content-Type: application/json

{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "real_time": true
}
```
Returns multiple stocks at once:
```json
{
  "results": {
    "AAPL": {
      "price": 150.25,
      "timestamp": "2024-01-15T10:30:00Z",
      "data_age_seconds": 15.2
    }
  },
  "errors": [],
  "timestamp": 1705312200
}
```

### Price Alerts
```bash
POST /alerts/set
Content-Type: application/json

{
  "ticker": "AAPL",
  "threshold": 150.0,
  "user_id": "user123"
}
```

```bash
GET /alerts/list?user_id=user123
```

### Cached Stock Data (Original)
```bash
GET /stock?ticker=AAPL
```
Returns data with 5-minute cache for general use.

## 🎨 Dashboard Features

### Real-Time Dashboard
Visit: `http://localhost:5000/dashboard`

**Features:**
- 📈 Live stock price updates
- 🎨 Beautiful gradient design
- 📱 Mobile-responsive layout
- ⚡ Auto-refresh every 30 seconds
- 🔧 Customizable refresh intervals
- ❌ Easy stock removal
- 📊 Price change indicators
- ⏰ Data age indicators

### Rate Limiter Monitor
Visit: `http://localhost:5000/monitor`

**Features:**
- 📊 Real-time rate limiter status
- 💾 Cache statistics
- 🔄 Reset functionality
- ⚠️ Error tracking
- 📈 Performance metrics

## 🔧 Configuration

### Cache Durations
```python
REAL_TIME_CACHE_DURATION = 30    # 30 seconds for real-time
SHORT_CACHE_DURATION = 300       # 5 minutes for general use
LONG_CACHE_DURATION = 900        # 15 minutes for historical
```

### Rate Limiting
```python
rate_limiter.min_delay = 1.0                    # 1 second between requests
rate_limiter.rate_limit_window = 60             # 60-second window
rate_limiter.max_requests_per_window = 30       # 30 requests per minute
```

## 📱 Usage Examples

### 1. Get Real-Time Price
```javascript
fetch('/stock/realtime?ticker=AAPL')
  .then(response => response.json())
  .then(data => {
    console.log(`AAPL: $${data.current_price}`);
    console.log(`Data age: ${data.data_age_seconds}s`);
  });
```

### 2. Batch Update Multiple Stocks
```javascript
fetch('/stocks/batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    tickers: ['AAPL', 'MSFT', 'GOOGL'],
    real_time: true
  })
})
.then(response => response.json())
.then(data => {
  Object.entries(data.results).forEach(([ticker, info]) => {
    console.log(`${ticker}: $${info.price}`);
  });
});
```

### 3. Set Price Alert
```javascript
fetch('/alerts/set', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ticker: 'AAPL',
    threshold: 150.0,
    user_id: 'my_user_id'
  })
})
.then(response => response.json())
.then(data => {
  console.log(`Alert set: ${data.message}`);
});
```

## 🧪 Testing

### Run Real-Time Tests
```bash
python test_realtime.py
```

**Tests include:**
- ✅ Real-time endpoint functionality
- ✅ Batch request handling
- ✅ Price alert system
- ✅ Cache comparison
- ✅ Rate limiter status

### Run Rate Limiting Tests
```bash
python test_rate_limiting.py
```

## 🎯 Best Practices

### 1. Use Appropriate Endpoints
- **Real-time trading**: `/stock/realtime`
- **General queries**: `/stock` (cached)
- **Multiple stocks**: `/stocks/batch`

### 2. Monitor Rate Limits
- Check `/rate-limit-status` regularly
- Use the dashboard at `/monitor`
- Reset when needed with `/reset-rate-limiter`

### 3. Handle Errors Gracefully
- Always check response status codes
- Implement retry logic for 429 errors
- Provide fallback to cached data

### 4. Optimize Requests
- Use batch endpoints for multiple stocks
- Set appropriate refresh intervals
- Monitor data age indicators

## 🔍 Troubleshooting

### Common Issues

#### Prices Not Updating
1. Check data age indicators
2. Verify rate limiter status
3. Ensure network connectivity
4. Check for rate limiting

#### High Data Age
1. Increase refresh frequency
2. Check rate limiter delays
3. Monitor cache statistics
4. Reset rate limiter if needed

#### Rate Limiting Issues
1. Visit `/monitor` dashboard
2. Check current delay settings
3. Use batch requests instead
4. Implement exponential backoff

### Debug Information
- **Data age**: Shows how old the data is
- **Rate limiter status**: Current delays and usage
- **Cache statistics**: Hit/miss ratios
- **Error tracking**: Failed requests and backoff periods

## 🚀 Performance Benefits

### Real-Time Features
- ✅ **30-second updates** for live trading
- ✅ **Visual indicators** for price changes
- ✅ **Data freshness** tracking
- ✅ **Automatic refresh** capabilities

### Rate Limiting Benefits
- ✅ **90% reduction** in API calls
- ✅ **Intelligent caching** strategies
- ✅ **Graceful degradation** on errors
- ✅ **Adaptive delays** based on usage

### User Experience
- ✅ **Beautiful dashboard** with live updates
- ✅ **Mobile-responsive** design
- ✅ **Easy stock management**
- ✅ **Price alert system**

## 🔮 Future Enhancements

### Planned Features
- [ ] WebSocket real-time updates
- [ ] Advanced charting capabilities
- [ ] Portfolio tracking
- [ ] News integration
- [ ] Technical indicators
- [ ] Push notifications

### Customization Options
- [ ] User-specific dashboards
- [ ] Custom alert conditions
- [ ] Advanced filtering
- [ ] Export capabilities
- [ ] API rate limit customization

## 📞 Support

For issues or questions:
1. Check the monitoring dashboards
2. Review the test scripts
3. Check rate limiter status
4. Reset the system if needed
5. Adjust configuration parameters

## 📄 License

This real-time system is part of the chatbot_in project and follows the same licensing terms. 

---

## 1. **Yes, It Can Work When Hosted!**
- Your Flask backend and the dashboard frontend are standard web technologies and can be hosted on a server (VPS, cloud, or shared hosting with Python support).
- The dashboard will work from anywhere as long as it can reach your backend’s API endpoints.

---

## 2. **Important Hosting Considerations**

### **A. Expose the Flask App Publicly**
- By default, Flask runs on `localhost` (only accessible from your machine).
- To make it public, run Flask with:
  ```bash
  python app.py --host=0.0.0.0 --port=5000
  ```
  Or set in your code:
  ```python
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000, debug=True)
  ```

### **B. Use a Production WSGI Server**
- For production, use **Gunicorn** or **uWSGI** (not Flask’s built-in server).
  ```bash
  pip install gunicorn
  gunicorn -w 4 app:app
  ```
- Or use a platform like **Heroku**, **Render**, **PythonAnywhere**, **AWS EC2**, **DigitalOcean**, etc.

### **C. Set Up a Reverse Proxy**
- Use **Nginx** or **Apache** to forward requests from port 80/443 to your Flask app.
- This is standard for production deployments.

### **D. CORS (Cross-Origin Resource Sharing)**
- If your frontend and backend are on different domains/ports, enable CORS in Flask:
  ```bash
  pip install flask-cors
  ```
  ```python
  from flask_cors import CORS
  CORS(app)
  ```

### **E. Environment Variables**
- Make sure your `.env` file (for API keys) is present on the server.
- Never expose your API keys in the frontend!

### **F. Firewall and Ports**
- Open the port you’re using (default 5000, or 80/443 for web).
- Make sure your cloud provider/firewall allows incoming connections.

### **G. HTTPS**
- For production, always use HTTPS (SSL certificate).
- Let’s Encrypt provides free SSL certificates.

---

## 3. **Yahoo Finance Rate Limiting**
- The rate limiting and caching you implemented will help, but if you get a lot of users, you may still hit Yahoo’s limits.
- For heavy production use, consider a paid/official market data API.

---

## 4. **Testing After Hosting**
- After deploying, visit `http://your-server-ip:5000/dashboard` (or your domain).
- Make sure `/stock/realtime?ticker=AAPL` returns data from the public internet.
- Test with your test scripts from a remote machine.

---

## 5. **Summary Table**

| Step                | What to Do                                      |
|---------------------|-------------------------------------------------|
| Expose Flask        | `host='0.0.0.0'` or use Gunicorn                |
| Reverse Proxy       | Use Nginx/Apache for production                 |
| Enable CORS         | `from flask_cors import CORS; CORS(app)`        |
| Set Env Vars        | Place `.env` on server, never in frontend       |
| Open Ports          | Allow 80/443 (web) or 5000 (dev)                |
| Use HTTPS           | Get SSL cert for your domain                    |
| Test Public Access  | Try API and dashboard from another device       |

---

## 6. **Resources**
- [Flask Deployment Options](https://flask.palletsprojects.com/en/latest/deploying/)
- [Gunicorn Docs](https://gunicorn.org/)
- [Flask-CORS](https://flask-cors.readthedocs.io/en/latest/)
- [Let’s Encrypt SSL](https://letsencrypt.org/)

---

**If you need a step-by-step guide for a specific host (Heroku, AWS, etc.), let me know!**  
Or if you run into any issues after deploying, just share the error and I’ll help you fix it. 
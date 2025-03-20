// Portfolio update function
function updatePortfolio() {
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            const summaryHtml = `
                <h5>Total Portfolio Value: $${data.total_value.toFixed(2)}</h5>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Type</th>
                            <th>Shares</th>
                            <th>Current Value</th>
                            <th>P/L</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.investments.map(inv => `
                            <tr>
                                <td>${inv.symbol}</td>
                                <td>${inv.type}</td>
                                <td>${inv.shares}</td>
                                <td>$${inv.current_value.toFixed(2)}</td>
                                <td class="${inv.profit_loss >= 0 ? 'text-success' : 'text-danger'}">
                                    $${inv.profit_loss.toFixed(2)}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            document.getElementById('portfolio-summary').innerHTML = summaryHtml;
        });
}

// Chat functionality
function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (message) {
        addMessage('user', message);
        
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            addMessage('bot', data.response);
        });
        
        input.value = '';
    }
}

function addMessage(type, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to fetch and display market updates
async function updateMarketData() {
    try {
        const response = await fetch('/api/market-updates');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error fetching market updates:', data.error);
            return;
        }

        // Update market indices
        const indicesHtml = data.market_summary.map(index => `
            <div class="index-card">
                <div class="name">${index.name}</div>
                <div class="price">${index.price.toFixed(2)}</div>
                <div class="change ${index.change > 0 ? 'positive-change' : 'negative-change'}">
                    ${index.change.toFixed(2)} (${index.change_percent.toFixed(2)}%)
                </div>
            </div>
        `).join('');
        document.getElementById('market-indices').innerHTML = indicesHtml;

        // Update top gainers
        const gainersHtml = data.top_gainers.map(stock => `
            <div class="stock-card">
                <div class="symbol">${stock.symbol}</div>
                <div class="name">${stock.name}</div>
                <div class="change positive-change">
                    +${stock.change_percent.toFixed(2)}%
                </div>
            </div>
        `).join('');
        document.getElementById('top-gainers').innerHTML = gainersHtml;

        // Update top losers
        const losersHtml = data.top_losers.map(stock => `
            <div class="stock-card">
                <div class="symbol">${stock.symbol}</div>
                <div class="name">${stock.name}</div>
                <div class="change negative-change">
                    ${stock.change_percent.toFixed(2)}%
                </div>
            </div>
        `).join('');
        document.getElementById('top-losers').innerHTML = losersHtml;

        // Update news
        const newsHtml = data.news.map(item => `
            <div class="news-item">
                <a href="${item.link}" target="_blank">
                    <div class="title">${item.title}</div>
                    <div class="meta">
                        ${item.publisher} • ${new Date(item.published).toLocaleString()}
                    </div>
                </a>
            </div>
        `).join('');
        document.getElementById('news-list').innerHTML = newsHtml;

    } catch (error) {
        console.error('Error updating market data:', error);
    }
}

// Update portfolio every minute
updatePortfolio();
setInterval(updatePortfolio, 60000);

// Update market data every 5 minutes
updateMarketData();
setInterval(updateMarketData, 5 * 60 * 1000);

// Event listener for Enter key in chat
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

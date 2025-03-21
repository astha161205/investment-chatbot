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
        let indicesHtml = '<div class="market-indices-grid">';
        if (data.market_summary && data.market_summary.length > 0) {
            indicesHtml += data.market_summary.map(index => `
                <div class="index-card">
                    <div class="name">${index.name}</div>
                    <div class="price">${index.price.toFixed(2)}</div>
                    <div class="change ${index.change > 0 ? 'positive-change' : 'negative-change'}">
                        ${index.change.toFixed(2)} (${index.change_percent.toFixed(2)}%)
                    </div>
                </div>
            `).join('');
        } else {
            indicesHtml += '<div class="error-message">Market data temporarily unavailable</div>';
        }
        indicesHtml += '</div>';
        
        // Update market news
        let newsHtml = '<div class="market-news">';
        if (data.news && data.news.length > 0) {
            newsHtml += '<h5>Latest Market News</h5>';
            newsHtml += data.news.map(item => `
                <div class="news-item">
                    <a href="${item.link}" target="_blank" class="news-title">${item.title}</a>
                    <div class="news-meta">
                        <span class="publisher">${item.publisher}</span>
                        <span class="published">${item.published}</span>
                    </div>
                </div>
            `).join('');
        } else {
            newsHtml += '<div class="error-message">News temporarily unavailable</div>';
        }
        newsHtml += '</div>';

        // Update the DOM
        document.getElementById('market-updates').innerHTML = indicesHtml + newsHtml;

    } catch (error) {
        console.error('Error in updateMarketData:', error);
        document.getElementById('market-updates').innerHTML = 
            '<div class="error-message">Failed to load market data. Please try again later.</div>';
    }
}

// Initial update and set interval
updateMarketData();
setInterval(updateMarketData, 30000); // Update every 30 seconds

// Update portfolio every minute
updatePortfolio();
setInterval(updatePortfolio, 60000);

// Event listener for Enter key in chat
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

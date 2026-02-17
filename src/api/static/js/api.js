const API_BASE = window.location.origin;

class ApiClient {
    constructor(baseUrl = API_BASE) {
        this.baseUrl = baseUrl;
        this.token = localStorage.getItem('api_token') || '';
    }

    setToken(token) {
        this.token = token;
        localStorage.setItem('api_token', token);
    }

    async request(endpoint, options = {}, retries = 3) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        let lastError;
        for (let attempt = 0; attempt < retries; attempt++) {
            try {
                const response = await fetch(url, {
                    ...options,
                    headers
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || `HTTP ${response.status}`);
                }

                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await response.json();
                }
                return await response.text();
            } catch (error) {
                console.error(`API Error [${endpoint}] (attempt ${attempt + 1}):`, error);
                lastError = error;
                
                if (attempt < retries - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
                }
            }
        }
        throw lastError;
    }

    get(endpoint, options = {}) {
        return this.request(endpoint, { method: 'GET' }, options.retries || 3);
    }

    post(endpoint, data, options = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        }, options.retries || 1);
    }

    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
}

const api = new ApiClient();

// WebSocket Client for real-time updates
class WebSocketClient {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.listeners = new Map();
        this.subscriptions = new Set();
        this.connected = false;
        this.connectionState = 'disconnected'; // disconnected, connecting, connected
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.connectionState = 'connecting';
        this.notifyListeners('connectionState', this.connectionState);

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.connected = true;
                this.connectionState = 'connected';
                this.reconnectAttempts = 0;
                this.notifyListeners('connectionState', this.connectionState);
                
                // Re-subscribe to channels
                if (this.subscriptions.size > 0) {
                    this.subscribe(Array.from(this.subscriptions));
                }
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                this.connected = false;
                this.connectionState = 'disconnected';
                this.notifyListeners('connectionState', this.connectionState);
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.notifyListeners('error', error);
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
            
            this.connectionState = 'reconnecting';
            this.notifyListeners('connectionState', this.connectionState);
            
            setTimeout(() => this.connect(), delay);
        } else {
            console.error('Max reconnect attempts reached');
            this.connectionState = 'failed';
            this.notifyListeners('connectionState', this.connectionState);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    subscribe(channels) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            channels.forEach(c => this.subscriptions.add(c));
            return;
        }

        this.ws.send(JSON.stringify({
            type: 'subscribe',
            channels: channels
        }));
        
        channels.forEach(c => this.subscriptions.add(c));
    }

    unsubscribe(channels) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'unsubscribe',
                channels: channels
            }));
        }
        
        channels.forEach(c => this.subscriptions.delete(c));
    }

    handleMessage(message) {
        const { type, data } = message;
        
        switch (type) {
            case 'market_update':
                this.notifyListeners('marketUpdate', data);
                break;
            case 'orderbook':
                this.notifyListeners('orderbook', data);
                break;
            case 'opportunity':
                this.notifyListeners('opportunity', data);
                break;
            case 'trade':
                this.notifyListeners('trade', data);
                break;
            case 'portfolio':
                this.notifyListeners('portfolio', data);
                break;
            case 'system':
                this.notifyListeners('system', data);
                break;
            case 'subscribed':
                console.log('Subscribed to:', message.channels);
                break;
            case 'unsubscribed':
                console.log('Unsubscribed from:', message.channels);
                break;
            case 'pong':
                // Keep-alive response
                break;
            default:
                console.log('Unknown message type:', type);
        }
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
        
        return () => this.off(event, callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
    }

    notifyListeners(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (e) {
                    console.error('Listener error:', e);
                }
            });
        }
    }

    getConnectionState() {
        return this.connectionState;
    }
}

// Global WebSocket instance
const wsClient = new WebSocketClient();

// Auto-connect on load
wsClient.connect();

// Keep-alive ping
setInterval(() => {
    if (wsClient.ws && wsClient.ws.readyState === WebSocket.OPEN) {
        wsClient.ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);

const Api = {
    // ... existing methods remain the same
    async getBotStatus() {
        return await api.get('/api/v2/bot/status');
    },

    async getHealth() {
        return await api.get('/api/health');
    },

    async toggleBot(enabled) {
        return await api.post('/api/v2/bot/toggle', { enabled });
    },

    async shutdownBot() {
        return await api.post('/api/v2/bot/shutdown', {});
    },

    async getDashboardSummary() {
        return await api.get('/api/v2/dashboard/summary');
    },

    async getOrderbooks() {
        return await api.get('/api/v2/dashboard/orderbooks');
    },

    async getArbitrageOpportunities() {
        return await api.get('/api/v2/arbitrage/opportunities');
    },

    async getStrategies() {
        return await api.get('/api/v2/strategies');
    },

    async toggleStrategy(strategyName, enabled) {
        return await api.post(`/api/v2/strategies/${strategyName}/toggle`, { enabled });
    },

    async updateStrategyConfig(strategyName, config) {
        return await api.post(`/api/v2/strategies/${strategyName}/config`, config);
    },

    async getRiskMetrics() {
        return await api.get('/api/v2/risk/metrics');
    },

    async getStressTests() {
        return await api.get('/api/v2/risk/stress-tests');
    },

    async getRollingRisk(window = 20) {
        return await api.get(`/api/v2/risk/rolling?window=${window}`);
    },

    async addReturnObservation(returnValue) {
        return await api.post('/api/v2/risk/returns', { return: returnValue });
    },

    async getCorrelations() {
        return await api.get('/api/v2/analytics/correlations');
    },

    async getCorrelationMatrix() {
        return await api.get('/api/v2/analytics/correlation-matrix');
    },

    async getAnalyticsStats() {
        return await api.get('/api/v2/analytics/stats');
    },

    async getPerformance(period = '7d') {
        return await api.get(`/api/v2/analytics/performance?period=${period}`);
    },

    async getAttribution() {
        return await api.get('/api/v2/analytics/attribution');
    },

    async runBacktest(config) {
        return await api.post('/api/v2/backtesting/run', config);
    },

    async getBacktestResults(taskId) {
        return await api.get(`/api/v2/backtesting/results/${taskId}`);
    },

    async downloadBacktestResults(taskId) {
        return await api.get(`/api/v2/backtesting/results/${taskId}/download`);
    },

    async getPaperBalance() {
        return await api.get('/api/v2/paper/balance');
    },

    async getPaperStats() {
        return await api.get('/api/v2/paper/stats');
    },

    async resetPaper() {
        return await api.post('/api/v2/paper/reset', {});
    },

    async submitOrder(order) {
        return await api.post('/api/v2/orders', order);
    },

    async getOrders() {
        return await api.get('/api/v2/orders');
    },

    async getOrder(orderId) {
        return await api.get(`/api/v2/orders/${orderId}`);
    },

    async getMLModels() {
        return await api.get('/api/v2/ml/models');
    },

    async createMLModel(config) {
        return await api.post('/api/v2/ml/models', config);
    },

    async getMLModel(modelId) {
        return await api.get(`/api/v2/ml/models/${modelId}`);
    },

    async deployMLModel(modelId) {
        return await api.post(`/api/v2/ml/models/${modelId}/deploy`, {});
    },

    async getExchanges() {
        return await api.get('/api/v2/exchanges');
    },

    async getExchangeMarkets(exchangeName) {
        return await api.get(`/api/v2/exchanges/${exchangeName}/markets`);
    },

    // Trade Journal methods
    async getJournalEntries() {
        return await api.get('/api/v2/journal');
    },

    async createJournalEntry(entry) {
        return await api.post('/api/v2/journal', entry);
    },

    async updateJournalEntry(id, entry) {
        return await api.post(`/api/v2/journal/${id}`, entry);
    },

    async deleteJournalEntry(id) {
        return await api.delete(`/api/v2/journal/${id}`);
    },

    // Export methods
    async exportJournal(format = 'csv') {
        const response = await fetch(`${API_BASE}/api/v2/export/journal?format=${format}`, {
            headers: this.token ? { 'Authorization': `Bearer ${this.token}` } : {}
        });
        
        if (format === 'json') {
            return await response.json();
        }
        
        return await response.text();
    },

    async exportOrders(format = 'csv', status = null) {
        let url = `${API_BASE}/api/v2/export/orders?format=${format}`;
        if (status) url += `&status=${status}`;
        
        const response = await fetch(url, {
            headers: this.token ? { 'Authorization': `Bearer ${this.token}` } : {}
        });
        
        if (format === 'json') {
            return await response.json();
        }
        
        return await response.text();
    },

    // Notifications methods
    async getNotifications(unreadOnly = false, limit = 50) {
        return await api.get(`/api/v2/notifications?unread_only=${unreadOnly}&limit=${limit}`);
    },

    async markNotificationRead(id) {
        return await api.post(`/api/v2/notifications/${id}/read`, {});
    },

    async markAllNotificationsRead() {
        return await api.post(`/api/v2/notifications/read-all`, {});
    },

    async deleteNotification(id) {
        return await api.delete(`/api/v2/notifications/${id}`);
    },

    // Advanced Orders methods
    async createStopLossOrder(order) {
        return await api.post('/api/v2/orders/stop-loss', order);
    },

    async createTakeProfitOrder(order) {
        return await api.post('/api/v2/orders/take-profit', order);
    },

    async createOCOOrder(order) {
        return await api.post('/api/v2/orders/oco', order);
    },

    async createTrailingStopOrder(order) {
        return await api.post('/api/v2/orders/trailing', order);
    },

    async createTWAPOrder(order) {
        return await api.post('/api/v2/orders/twap', order);
    },

    async getAdvancedOrders(orderType = null, status = null) {
        let url = '/api/v2/orders/advanced';
        const params = [];
        if (orderType) params.push(`order_type=${orderType}`);
        if (status) params.push(`status=${status}`);
        if (params.length) url += '?' + params.join('&');
        return await api.get(url);
    },

    async getAdvancedOrder(orderId) {
        return await api.get(`/api/v2/orders/advanced/${orderId}`);
    },

    async cancelAdvancedOrder(orderId) {
        return await api.delete(`/api/v2/orders/advanced/${orderId}`);
    },

    // Arbitrage Detection methods
    async getArbitrageStatus() {
        return await api.get('/api/v2/arbitrage/status');
    },

    async startArbitrageDetection() {
        return await api.post('/api/v2/arbitrage/start', {});
    },

    async stopArbitrageDetection() {
        return await api.post('/api/v2/arbitrage/stop', {});
    },

    async getArbitrageOpportunities(limit = 20) {
        return await api.get(`/api/v2/arbitrage/opportunities?limit=${limit}`);
    },

    // Market Data methods
    async getMarkets(status = 'open', limit = 50) {
        return await api.get(`/api/v2/markets?status=${status}&limit=${limit}`);
    },

    async getLiveMarkets(limit = 50) {
        return await api.get(`/api/v2/markets/live?limit=${limit}`);
    },

    async getMarketOrderbook(marketId) {
        return await api.get(`/api/v2/markets/${marketId}/orderbook`);
    },

    async refreshMarkets() {
        return await api.post('/api/v2/markets/refresh', {});
    },

    // Auto-Trading methods
    async getAutoTradingStatus() {
        return await api.get('/api/v2/auto-trading/status');
    },

    async enableAutoTrading(config) {
        return await api.post('/api/v2/auto-trading/enable', config);
    },

    async disableAutoTrading() {
        return await api.post('/api/v2/auto-trading/disable', {});
    },

    async updateAutoTradingConfig(config) {
        return await api.put('/api/v2/auto-trading/config', config);
    },

    // Position Sizing methods
    async calculatePositionSize(data) {
        return await api.post('/api/v2/position-sizing/calculate', data);
    },

    async getPositionSizingHistory() {
        return await api.get('/api/v2/position-sizing/history');
    },

    // Exchange Trading methods
    async getExchangeStatus() {
        return await api.get('/api/v2/exchange/status');
    },

    async placeOrder(order) {
        return await api.post('/api/v2/exchange/orders', order);
    },

    async getOrders(status = null, limit = 50) {
        let url = `/api/v2/exchange/orders?limit=${limit}`;
        if (status) url += `&status=${status}`;
        return await api.get(url);
    },

    async getOrder(orderId) {
        return await api.get(`/api/v2/exchange/orders/${orderId}`);
    },

    async cancelOrder(orderId) {
        return await api.delete(`/api/v2/exchange/orders/${orderId}`);
    },

    async getPositions() {
        return await api.get('/api/v2/exchange/positions');
    },

    async getBalance() {
        return await api.get('/api/v2/exchange/balance');
    },

    async setPaperMode(enabled) {
        return await api.post(`/api/v2/exchange/paper-mode?enabled=${enabled}`, {});
    },

    async getPaperReport() {
        return await api.get('/api/v2/paper/report');
    },

    async resetPaperTrading(initialBalance = null) {
        let url = '/api/v2/paper/reset';
        if (initialBalance) url += `?initial_balance=${initialBalance}`;
        return await api.post(url, {});
    },

    // Spread Alerts methods
    async getSpreadAlerts() {
        return await api.get('/api/v2/spread-alerts');
    },

    async createSpreadAlert(alert) {
        return await api.post('/api/v2/spread-alerts', alert);
    },

    async updateSpreadAlert(alertId, alert) {
        return await api.put(`/api/v2/spread-alerts/${alertId}`, alert);
    },

    async deleteSpreadAlert(alertId) {
        return await api.delete(`/api/v2/spread-alerts/${alertId}`);
    },

    async triggerSpreadAlert(alertId) {
        return await api.post(`/api/v2/spread-alerts/${alertId}/trigger`, {});
    },

    // One-Click Trading methods
    async enableOneClick() {
        return await api.post('/api/v2/one-click/enable', {});
    },

    async disableOneClick() {
        return await api.post('/api/v2/one-click/disable', {});
    },

    async getOneClickStatus() {
        return await api.get('/api/v2/one-click/status');
    },

    async executeQuickTrade(trade) {
        return await api.post('/api/v2/one-click/trade', trade);
    },

    // Database methods
    async getDatabaseStats() {
        return await api.get('/api/v2/db/stats');
    },

    async saveTradeToDb(trade) {
        return await api.post('/api/v2/db/trades', trade);
    },

    async getTradesFromDb(limit = 100, offset = 0) {
        return await api.get(`/api/v2/db/trades?limit=${limit}&offset=${offset}`);
    },

    async saveSetting(key, value) {
        return await api.post(`/api/v2/db/settings?key=${encodeURIComponent(key)}&value=${encodeURIComponent(value)}`, {});
    },

    async getSetting(key) {
        return await api.get(`/api/v2/db/settings/${encodeURIComponent(key)}`);
    }
};

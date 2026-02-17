const AppState = {
    currentPage: 'dashboard',
    botRunning: false,
    theme: localStorage.getItem('theme') || 'light',
    lastUpdated: null,
    pollingIntervals: {},
    data: {},
    quickActionsOpen: false,
    calculatorOpen: false
};

const KeyboardShortcuts = {
    'd': 'dashboard',
    'm': 'markets',
    'a': 'arbitrage',
    'p': 'portfolio',
    'r': 'risk',
    's': 'strategies',
    'o': 'orders',
    'j': 'journal',
    't': 'toggleTheme',
    'b': 'toggleBot',
    '?': 'showHelp',
    'Escape': 'closeModals'
};

function initApp() {
    loadTheme();
    setupRouter();
    setupEventListeners();
    setupKeyboardShortcuts();
    updateBotStatus();
    startGlobalPolling();
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't trigger shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            return;
        }
        
        const key = e.key.toLowerCase();
        const action = KeyboardShortcuts[key];
        
        if (!action) return;
        
        e.preventDefault();
        
        switch(action) {
            case 'toggleTheme':
                toggleTheme();
                break;
            case 'toggleBot':
                toggleBot();
                break;
            case 'showHelp':
                showKeyboardShortcuts();
                break;
            case 'closeModals':
                closeAllModals();
                break;
            default:
                if (Pages[action]) {
                    navigateTo(action);
                }
        }
    });
}

function showKeyboardShortcuts() {
    const shortcuts = Object.entries(KeyboardShortcuts).map(([key, action]) => {
        const page = Pages[action];
        return `<div class="shortcut-item"><kbd>${key === 'Escape' ? 'Esc' : key.toUpperCase()}</kbd><span>${page ? page.title : action}</span></div>`;
    }).join('');
    
    const html = `
        <div class="modal-overlay" onclick="closeAllModals()">
            <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 400px;">
                <div class="modal-header">
                    <h3>Keyboard Shortcuts</h3>
                    <button class="btn btn-icon" onclick="closeAllModals()">${Icons.close}</button>
                </div>
                <div class="modal-body">
                    <div class="shortcuts-grid">
                        ${shortcuts}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    const div = document.createElement('div');
    div.innerHTML = html;
    document.body.appendChild(div);
}

function closeAllModals() {
    document.querySelectorAll('.modal-overlay').forEach(m => m.remove());
}

function loadTheme() {
    if (AppState.theme === 'dark') {
        document.body.classList.add('dark');
    }
}

function toggleTheme() {
    AppState.theme = AppState.theme === 'light' ? 'dark' : 'light';
    document.body.classList.toggle('dark');
    localStorage.setItem('theme', AppState.theme);
    destroyAllCharts();
    renderCurrentPage();
}

function setupRouter() {
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash || '#/';
        const page = hash.slice(2) || 'dashboard';
        navigateTo(page);
    });
    
    const initialPage = window.location.hash.slice(2) || 'dashboard';
    navigateTo(initialPage);
}

function navigateTo(page) {
    AppState.currentPage = page;
    stopPagePolling();
    updateActiveNav();
    renderCurrentPage();
    startPagePolling();
}

function stopPagePolling() {
    Object.values(AppState.pollingIntervals).forEach(clearInterval);
    AppState.pollingIntervals = {};
}

function startGlobalPolling() {
    AppState.pollingIntervals.global = setInterval(updateBotStatus, 5000);
}

function startPagePolling() {
    const page = AppState.currentPage;
    
    switch(page) {
        case 'markets':
            AppState.pollingIntervals.page = setInterval(() => renderPage('markets'), 5000);
            break;
        case 'arbitrage':
            AppState.pollingIntervals.page = setInterval(() => renderPage('arbitrage'), 3000);
            break;
        case 'system':
            AppState.pollingIntervals.page = setInterval(() => renderPage('system'), 3000);
            break;
    }
}

function updateActiveNav() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.page === AppState.currentPage);
    });
}

async function updateBotStatus() {
    try {
        const health = await Api.getHealth();
        AppState.botRunning = health.bot_running || false;
        AppState.lastUpdated = new Date();
        updateHeader();
    } catch (e) {
        console.error('Failed to get bot status:', e);
    }
}

function updateHeader() {
    const statusEl = document.getElementById('botStatus');
    const toggleBtn = document.getElementById('toggleBtn');
    const lastUpdatedEl = document.getElementById('lastUpdated');
    
    if (statusEl) {
        statusEl.textContent = AppState.botRunning ? 'Running' : 'Stopped';
        statusEl.className = `bot-status ${AppState.botRunning ? 'running' : 'stopped'}`;
    }
    
    if (toggleBtn) {
        toggleBtn.innerHTML = AppState.botRunning 
            ? `${Icons.stop}<span>Stop</span>` 
            : `${Icons.play}<span>Start</span>`;
    }
    
    if (lastUpdatedEl && AppState.lastUpdated) {
        lastUpdatedEl.textContent = `Updated: ${AppState.lastUpdated.toLocaleTimeString()}`;
    }
}

function setupEventListeners() {
    document.addEventListener('click', (e) => {
        if (e.target.closest('.mobile-menu-btn')) {
            document.querySelector('.sidebar').classList.toggle('open');
        }
        
        if (e.target.closest('.theme-toggle')) {
            toggleTheme();
        }
        
        if (e.target.closest('#toggleBtn')) {
            toggleBot();
        }
    });
    
    document.addEventListener('submit', (e) => {
        e.preventDefault();
    });
}

async function toggleBot() {
    showToast(AppState.botRunning ? 'Stopping bot...' : 'Starting bot...', 'info');
    
    try {
        await Api.toggleBot(!AppState.botRunning);
        AppState.botRunning = !AppState.botRunning;
        updateBotStatus();
        showToast(AppState.botRunning ? 'Bot started' : 'Bot stopped', 'success');
    } catch (e) {
        showToast('Failed to toggle bot: ' + e.message, 'error');
    }
}

function renderCurrentPage() {
    renderPage(AppState.currentPage);
}

async function renderPage(page) {
    const content = document.getElementById('pageContent');
    if (!content) return;
    
    const pageConfig = Pages[page];
    if (!pageConfig) {
        content.innerHTML = renderEmptyState('Page not found');
        return;
    }
    
    content.innerHTML = `
        <div class="header-left">
            ${renderPageTitle(pageConfig.title)}
        </div>
        <div class="header-right">
            <span class="last-updated" id="lastUpdated"></span>
            <button class="notification-btn" id="notificationBtn" title="Notifications" onclick="toggleNotificationsPanel()">
                ${Icons.bell}
                <span class="notification-badge" id="notificationBadge" style="display: none;">0</span>
            </button>
            <button class="theme-toggle" title="Toggle theme">${AppState.theme === 'light' ? Icons.moon : Icons.sun}</button>
            <button class="btn btn-sm" id="toggleBtn">
                ${AppState.botRunning ? Icons.stop + '<span>Stop</span>' : Icons.play + '<span>Start</span>'}
            </button>
        </div>
    `;
    
    let pageContent = '';
    
    try {
        switch(page) {
            case 'dashboard':
                pageContent = await renderDashboardPage();
                break;
            case 'markets':
                pageContent = await renderMarketsPage();
                break;
            case 'arbitrage':
                pageContent = await renderArbitragePage();
                break;
            case 'portfolio':
                pageContent = await renderPortfolioPage();
                break;
            case 'risk':
                pageContent = await renderRiskPage();
                break;
            case 'strategies':
                pageContent = await renderStrategiesPage();
                break;
            case 'analytics':
                pageContent = await renderAnalyticsPage();
                break;
            case 'backtesting':
                pageContent = await renderBacktestingPage();
                break;
            case 'ml':
                pageContent = await renderMLPage();
                break;
            case 'orders':
                pageContent = await renderOrdersPage();
                break;
            case 'journal':
                pageContent = await renderJournalPage();
                break;
            case 'calculator':
                pageContent = renderCalculatorPage();
                break;
            case 'alerts':
                pageContent = await renderAlertsPage();
                break;
            case 'advanced-orders':
                pageContent = await renderAdvancedOrdersPage();
                break;
            case 'charts':
                pageContent = await renderChartsPage();
                break;
            case 'auto-trading':
                pageContent = await renderAutoTradingPage();
                break;
            case 'position-sizing':
                pageContent = await renderPositionSizingPage();
                break;
            case 'spread-alerts':
                pageContent = await renderSpreadAlertsPage();
                break;
            case 'one-click':
                pageContent = await renderOneClickPage();
                break;
            case 'database':
                pageContent = await renderDatabasePage();
                break;
            case 'settings':
                pageContent = await renderSettingsPage();
                break;
            case 'system':
                pageContent = await renderSystemPage();
                break;
            default:
                pageContent = renderEmptyState('Page not found');
        }
    } catch (e) {
        console.error('Error rendering page:', e);
        pageContent = `<div class="card"><div class="card-body">Error: ${e.message}</div></div>`;
    }
    
    content.innerHTML = pageContent;
    
    setTimeout(() => {
        const toggleBtn = document.getElementById('toggleBtn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', toggleBot);
        }
        
        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);
        }
    }, 0);
}

// Dashboard widget system
const DashboardWidgets = {
    profit: { id: 'profit', title: 'Profit Over Time', type: 'chart', size: 'half' },
    trades: { id: 'trades', title: 'Trade Distribution', type: 'chart', size: 'half' },
    stats: { id: 'stats', title: 'Quick Stats', type: 'stats', size: 'half' },
    opportunities: { id: 'opportunities', title: 'Recent Opportunities', type: 'table', size: 'full' },
    positions: { id: 'positions', title: 'Open Positions', type: 'table', size: 'full' },
    risk: { id: 'risk', title: 'Risk Overview', type: 'metrics', size: 'half' }
};

let dashboardWidgets = JSON.parse(localStorage.getItem('dashboardWidgets') || JSON.stringify([
    'profit', 'trades', 'stats', 'risk', 'opportunities'
]));

function saveDashboardLayout() {
    localStorage.setItem('dashboardWidgets', JSON.stringify(dashboardWidgets));
}

function renderDashboardPage() {
    const status = AppState.data.botStatus || {};
    const summary = AppState.data.dashboardSummary || {};
    
    const portfolio = status.portfolio || {};
    const stats = summary.paper_trading_stats || {};
    
    const totalProfit = portfolio.total_profit || 0;
    const cashBalance = portfolio.cash_balance || 10000;
    const openPositions = portfolio.open_positions || 0;
    const winRate = portfolio.win_rate || 0;
    const totalTrades = portfolio.completed_trades || 0;
    const dailyPnl = stats.daily_pnl || 0;
    
    let widgetsHtml = dashboardWidgets.map(widgetId => {
        const widget = DashboardWidgets[widgetId];
        if (!widget) return '';
        
        switch(widgetId) {
            case 'profit':
                return renderWidget(widget, `<div class="chart-container"><canvas id="profitChart"></canvas></div>`);
            case 'trades':
                return renderWidget(widget, `<div class="chart-container"><canvas id="tradeChart"></canvas></div>`);
            case 'stats':
                return renderWidget(widget, `
                    <div class="grid grid-cols-2 gap-4">
                        <div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">${totalTrades}</div></div>
                        <div class="stat-card"><div class="stat-label">Daily P&L</div><div class="stat-value ${dailyPnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(dailyPnl)}</div></div>
                    </div>
                `);
            case 'risk':
                return renderWidget(widget, `
                    <div class="grid grid-cols-3 gap-2">
                        <div class="stat-card"><div class="stat-label">VaR 95%</div><div class="stat-value text-sm">--</div></div>
                        <div class="stat-card"><div class="stat-label">Sharpe</div><div class="stat-value text-sm">--</div></div>
                        <div class="stat-card"><div class="stat-label">Drawdown</div><div class="stat-value text-sm">--</div></div>
                    </div>
                `);
            case 'opportunities':
                return renderWidget(widget, `<p class="text-muted text-center py-4">Loading opportunities...</p>`);
            default:
                return '';
        }
    }).join('');
    
    let html = `
        <div class="dashboard-controls mb-4 flex justify-between items-center">
            <h2 class="text-lg font-semibold">Dashboard</h2>
            <div class="flex gap-2">
                <button class="btn btn-secondary btn-sm" onclick="toggleDashboardEdit()">
                    ${Icons.settings} Customize
                </button>
                <button class="btn btn-secondary btn-sm" onclick="refreshDashboard()">
                    ${Icons.refresh} Refresh
                </button>
            </div>
        </div>
        
        <div class="dashboard-grid" id="dashboardGrid">
            ${widgetsHtml}
        </div>
        
        <div class="dashboard-edit-panel" id="dashboardEditPanel" style="display: none;">
            <div class="card mt-4">
                <div class="card-header">
                    <h3 class="card-title">Customize Dashboard</h3>
                </div>
                <div class="card-body">
                    <p class="text-sm text-muted mb-4">Drag widgets to reorder, toggle to show/hide</p>
                    <div class="widget-toggles">
                        ${Object.entries(DashboardWidgets).map(([id, w]) => `
                            <label class="flex items-center gap-2 mb-2 cursor-pointer">
                                <input type="checkbox" ${dashboardWidgets.includes(id) ? 'checked' : ''} 
                                       onchange="toggleDashboardWidget('${id}')">
                                <span>${w.title}</span>
                            </label>
                        `).join('')}
                    </div>
                    <button class="btn btn-secondary mt-4" onclick="toggleDashboardEdit()">Done</button>
                </div>
            </div>
        </div>
    `;
    
    // Schedule chart rendering after DOM update
    setTimeout(() => {
        const mockProfitData = generateMockProfitData();
        updateProfitChart('profitChart', mockProfitData);
        
        const wins = Math.round(totalTrades * winRate);
        const losses = totalTrades - wins;
        updateWinLossChart('tradeChart', wins || 1, losses || 1);
    }, 100);
    
    return html;
}

function renderWidget(widget, content) {
    return `
        <div class="dashboard-widget" data-widget-id="${widget.id}" draggable="true">
            <div class="widget-header">
                <h3 class="widget-title">${widget.title}</h3>
                <span class="widget-drag-handle">${Icons.activity}</span>
            </div>
            <div class="widget-content">
                ${content}
            </div>
        </div>
    `;
}

function toggleDashboardEdit() {
    const panel = document.getElementById('dashboardEditPanel');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

function toggleDashboardWidget(widgetId) {
    const index = dashboardWidgets.indexOf(widgetId);
    if (index > -1) {
        dashboardWidgets.splice(index, 1);
    } else {
        dashboardWidgets.push(widgetId);
    }
    saveDashboardLayout();
    renderPage('dashboard');
}

async function refreshDashboard() {
    showToast('Refreshing...', 'info');
    try {
        const [status, summary] = await Promise.all([
            Api.getBotStatus().catch(() => ({})),
            Api.getDashboardSummary().catch(() => ({}))
        ]);
        AppState.data.botStatus = status;
        AppState.data.dashboardSummary = summary;
        renderPage('dashboard');
        showToast('Refreshed', 'success');
    } catch (e) {
        showToast('Refresh failed: ' + e.message, 'error');
    }
}
    
    setTimeout(() => {
        const mockProfitData = generateMockProfitData();
        updateProfitChart('profitChart', mockProfitData);
        
        const wins = Math.round(totalTrades * winRate);
        const losses = totalTrades - wins;
        updateWinLossChart('tradeChart', wins || 1, losses || 1);
    }, 100);
    
    return html;
}

function generateMockProfitData() {
    const data = [];
    let profit = 0;
    for (let i = 30; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        profit += (Math.random() - 0.4) * 100;
        data.push({
            date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            profit: profit
        });
    }
    return data;
}

async function renderMarketsPage() {
    const [markets, liveData] = await Promise.all([
        Api.getMarkets('open', 50).catch(() => ({ markets: [] })),
        Api.getLiveMarkets(50).catch(() => ({ markets: [] }))
    ]);
    
    const liveMarkets = liveData.markets || [];
    const marketsList = markets.markets || [];
    
    const searchInput = `
        <div class="mb-4 flex justify-between items-center">
            <div class="search-input flex-1">
                ${Icons.search}
                <input type="text" class="form-input" id="marketSearch" placeholder="Search markets..." oninput="filterMarkets(this.value)">
            </div>
            <button class="btn btn-sm ml-2" onclick="refreshMarketData()" title="Refresh">
                ${Icons.refresh}
            </button>
        </div>
    `;
    
    const rows = liveMarkets.map(ob => {
        const spreadClass = ob.spread <= 1 ? 'text-positive' : (ob.spread <= 5 ? 'text-muted' : 'text-negative');
        return [
            ob.market_id || '-',
            formatCurrency(ob.best_bid || 0),
            formatCurrency(ob.best_ask || 0),
            formatCurrency(ob.mid_price || 0),
            `<span class="${spreadClass}">${ob.spread || 0}</span>`,
            `${(ob.spread_percent || 0).toFixed(2)}%`,
            ob.bid_depth || 0,
            ob.ask_depth || 0,
            ob.last_update ? formatTimestamp(ob.last_update) : '-'
        ];
    });
    
    const table = renderTable(
        ['Market', 'Best Bid', 'Best Ask', 'Mid Price', 'Spread (¢)', 'Spread %', 'Bid Depth', 'Ask Depth', 'Last Update'],
        rows,
        liveMarkets.length === 0 ? 'No live market data. Click refresh to load.' : 'No markets available'
    );
    
    return searchInput + renderCard(`Live Markets (${liveMarkets.length})`, table);
}

async function refreshMarketData() {
    try {
        await Api.refreshMarkets();
        renderPage(AppState.currentPage);
    } catch (e) {
        alert('Error refreshing markets: ' + e.message);
    }
}

function filterMarkets(query) {
    const rows = document.querySelectorAll('#marketsTable tbody tr');
    const q = query.toLowerCase();
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(q) ? '' : 'none';
    });
}

async function renderArbitragePage() {
    const [status, opportunities] = await Promise.all([
        Api.getArbitrageStatus().catch(() => ({ running: false, opportunities_found: 0 })),
        Api.getArbitrageOpportunities().catch(() => ({ opportunities: [] }))
    ]);
    
    const oppList = opportunities.opportunities || opportunities || [];
    
    const rows = oppList.slice(0, 20).map(opp => [
        formatTimestamp(opp.timestamp),
        opp.market_id_1 || opp.buy_market_id || '-',
        opp.type || 'cross_market',
        formatCurrency(opp.profit_cents || opp.net_profit_cents || 0),
        `${((opp.confidence || 0) * 100).toFixed(0)}%`,
        opp.risk_level || 'low',
        opp.status || 'pending'
    ]);
    
    const table = renderTable(
        ['Time', 'Market', 'Type', 'Profit', 'Confidence', 'Risk', 'Status'],
        rows,
        'No arbitrage opportunities found'
    );
    
    return `
        <div class="grid grid-cols-3 gap-4 mb-6">
            ${renderStatCard('Detection Status', status.running ? 'Running' : 'Stopped', null, '', status.running ? ' (Active)' : '')}
            ${renderStatCard('Opportunities Found', status.opportunities_found || oppList.length)}
            ${renderStatCard('Last Opportunity', status.last_opportunity ? formatTimestamp(status.last_opportunity) : 'None')}
        </div>
        
        <div class="card mb-6">
            <div class="card-header flex justify-between items-center">
                <h3 class="card-title">Detection Control</h3>
                <div class="flex gap-2">
                    <button class="btn btn-success" onclick="toggleArbitrageDetection(true)" ${status.running ? 'disabled' : ''}>
                        ${Icons.play} Start
                    </button>
                    <button class="btn btn-danger" onclick="toggleArbitrageDetection(false)" ${!status.running ? 'disabled' : ''}>
                        ${Icons.stop} Stop
                    </button>
                </div>
            </div>
        </div>
        
        ${renderCard('Current Opportunities', table)}
    `;
}

async function toggleArbitrageDetection(start) {
    try {
        if (start) {
            await Api.startArbitrageDetection();
        } else {
            await Api.stopArbitrageDetection();
        }
        renderPage(AppState.currentPage);
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function renderPortfolioPage() {
    const [status, paperStats, exchangeStatus] = await Promise.all([
        Api.getBotStatus().catch(() => ({})),
        Api.getPaperStats().catch(() => ({})),
        Api.getExchangeStatus().catch(() => ({}))
    ]);
    
    const portfolio = status.portfolio || {};
    const stats = paperStats || {};
    const exchange = exchangeStatus || {};
    
    const cashBalance = portfolio.cash_balance || exchange.config?.max_order_value || 10000;
    const positionsValue = portfolio.positions_value || 0;
    const totalPnl = portfolio.total_pnl || stats.total_pnl || 0;
    const dailyPnl = portfolio.daily_pnl || exchange.stats?.daily_pnl || 0;
    const winRate = stats.win_rate || 0;
    const totalTrades = stats.filled_orders || exchange.stats?.total_trades || 0;
    
    return `
        <div class="grid grid-cols-4 gap-4 mb-6">
            ${renderStatCard('Cash Balance', formatCurrency(cashBalance), null, '$')}
            ${renderStatCard('Positions Value', formatCurrency(positionsValue), null, '$')}
            ${renderStatCard('Total P&L', formatCurrency(totalPnl), null, '$')}
            ${renderStatCard('Daily P&L', formatCurrency(dailyPnl), null, '$')}
        </div>
        
        <div class="grid grid-cols-4 gap-4 mb-6">
            ${renderStatCard('Win Rate', winRate.toFixed(1), null, '', '%')}
            ${renderStatCard('Total Trades', totalTrades)}
            ${renderStatCard('Mode', exchange.paper_mode ? 'Paper' : 'Live')}
            ${renderStatCard('Realized P&L', formatCurrency(stats.realized_pnl || 0), null, '$')}
        </div>
        
        <div class="grid grid-cols-2 gap-6">
            ${renderCard('Positions', `
                <p class="text-muted text-center py-4">Open positions will appear here</p>
            `)}
            ${renderCard('Trade History', `
                <p class="text-muted text-center py-4">Trade history will appear here</p>
            `)}
        </div>
    `;
}

async function renderRiskPage() {
    const [metrics, stressTests] = await Promise.all([
        Api.getRiskMetrics().catch(() => ({})),
        Api.getStressTests().catch(() => ({}))
    ]);
    
    const risk = metrics || {};
    const stress = stressTests || {};
    
    return `
        <div class="grid grid-cols-4 gap-4 mb-6">
            ${renderStatCard('VaR 95%', (risk.var_95 || 0) * 100, null, '', '%')}
            ${renderStatCard('VaR 99%', (risk.var_99 || 0) * 100, null, '', '%')}
            ${renderStatCard('Max Drawdown', (risk.max_drawdown || 0) * 100, null, '', '%')}
            ${renderStatCard('Sharpe Ratio', risk.sharpe_ratio || 0)}
        </div>
        
        <div class="grid grid-cols-2 gap-6 mb-6">
            ${renderCard('Risk Metrics', '<div class="chart-container"><canvas id="riskChart"></canvas></div>')}
            ${renderCard('Volatility', (risk.volatility || 0) * 100, null, '', '%')}
        </div>
        
        ${renderCard('Stress Tests', `
            <div class="grid grid-cols-2 gap-4">
                ${Object.entries(stress).map(([name, value]) => `
                    <div class="flex justify-between items-center p-4 bg-tertiary rounded">
                        <span>${name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                        <span class="${(value || 0) < 0 ? 'text-negative' : 'text-positive'}">
                            ${formatPercent(value || 0)}
                        </span>
                    </div>
                `).join('')}
            </div>
        `)}
    `;
}

async function renderStrategiesPage() {
    const strategies = await Api.getStrategies().catch(() => []);
    
    const html = `
        <div class="grid grid-cols-2 gap-4 mb-6">
            ${strategies.map(strategy => `
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">${strategy.display_name || strategy.name}</h3>
                        <div class="toggle ${strategy.enabled ? 'active' : ''}" 
                             onclick="toggleStrategy('${strategy.name}', ${!strategy.enabled})"></div>
                    </div>
                    <div class="card-body">
                        <p class="text-sm text-muted mb-4">${strategy.description || ''}</p>
                        <div class="grid grid-cols-4 gap-2 text-sm">
                            <div>
                                <div class="text-muted">Return</div>
                                <div class="${(strategy.total_return || 0) >= 0 ? 'text-positive' : 'text-negative'}">
                                    ${formatPercent(strategy.total_return || 0)}
                                </div>
                            </div>
                            <div>
                                <div class="text-muted">Sharpe</div>
                                <div>${(strategy.sharpe_ratio || 0).toFixed(2)}</div>
                            </div>
                            <div>
                                <div class="text-muted">Drawdown</div>
                                <div class="text-negative">${formatPercent(strategy.max_drawdown || 0)}</div>
                            </div>
                            <div>
                                <div class="text-muted">Win Rate</div>
                                <div>${formatPercent(strategy.win_rate || 0)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    return strategies.length ? html : renderEmptyState('No strategies available', Icons.strategies);
}

async function toggleStrategy(name, enabled) {
    try {
        await Api.toggleStrategy(name, enabled);
        showToast(`Strategy ${name} ${enabled ? 'enabled' : 'disabled'}`, 'success');
        renderPage('strategies');
    } catch (e) {
        showToast('Failed to toggle strategy: ' + e.message, 'error');
    }
}

async function renderAnalyticsPage() {
    const [correlations, attribution] = await Promise.all([
        Api.getCorrelations().catch(() => ({top_correlations: [], cointegrated_pairs: []})),
        Api.getAttribution().catch(() => ({}))
    ]);
    
    return `
        <div class="grid grid-cols-2 gap-6">
            ${renderCard('Top Correlations', `
                <table>
                    <thead>
                        <tr><th>Market 1</th><th>Market 2</th><th>Correlation</th></tr>
                    </thead>
                    <tbody>
                        ${(correlations.top_correlations || []).slice(0, 10).map(c => `
                            <tr>
                                <td>${c.market_1}</td>
                                <td>${c.market_2}</td>
                                <td class="${c.correlation > 0 ? 'text-positive' : 'text-negative'}">${c.correlation?.toFixed(3)}</td>
                            </tr>
                        `).join('') || '<tr><td colspan="3" class="text-center text-muted">No correlations found</td></tr>'}
                    </tbody>
                </table>
            `)}
            
            ${renderCard('Cointegrated Pairs', `
                <table>
                    <thead>
                        <tr><th>Market 1</th><th>Market 2</th><th>Hedge Ratio</th></tr>
                    </thead>
                    <tbody>
                        ${(correlations.cointegrated_pairs || []).slice(0, 10).map(c => `
                            <tr>
                                <td>${c.market_1}</td>
                                <td>${c.market_2}</td>
                                <td>${c.hedge_ratio?.toFixed(3)}</td>
                            </tr>
                        `).join('') || '<tr><td colspan="3" class="text-center text-muted">No pairs found</td></tr>'}
                    </tbody>
                </table>
            `)}
        </div>
    `;
}

async function renderBacktestingPage() {
    return `
        <div class="grid grid-cols-2 gap-6">
            ${renderCard('Run Backtest', `
                <form id="backtestForm" onsubmit="runBacktest(event)">
                    <div class="form-group">
                        <label class="form-label">Strategy Type</label>
                        <select class="form-select" id="strategyType">
                            <option value="arbitrage">Arbitrage</option>
                            <option value="pairs">Pairs Trading</option>
                            <option value="momentum">Momentum</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">Start Date</label>
                            <input type="date" class="form-input" id="startDate" value="${new Date(Date.now() - 30*24*60*60*1000).toISOString().split('T')[0]}">
                        </div>
                        <div class="form-group">
                            <label class="form-label">End Date</label>
                            <input type="date" class="form-input" id="endDate" value="${new Date().toISOString().split('T')[0]}">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Initial Capital ($)</label>
                        <input type="number" class="form-input" id="initialCapital" value="10000">
                    </div>
                    <button type="submit" class="btn btn-primary">Run Backtest</button>
                </form>
            `)}
            
            ${renderCard('Results', `
                <div id="backtestResults">
                    <p class="text-muted text-center">Run a backtest to see results</p>
                </div>
            `)}
        </div>
    `;
}

async function runBacktest(e) {
    e.preventDefault();
    
    const config = {
        strategy_config: {
            strategy_type: document.getElementById('strategyType').value,
            parameters: {},
            enabled: true
        },
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value,
        initial_capital: parseFloat(document.getElementById('initialCapital').value)
    };
    
    showToast('Running backtest...', 'info');
    
    try {
        const result = await Api.runBacktest(config);
        
        if (result.task_id) {
            showToast('Backtest started', 'success');
            pollBacktestResult(result.task_id);
        } else {
            showToast('Backtest completed', 'success');
            renderBacktestResults(result);
        }
    } catch (e) {
        showToast('Backtest failed: ' + e.message, 'error');
    }
}

function pollBacktestResult(taskId) {
    const interval = setInterval(async () => {
        try {
            const result = await Api.getBacktestResults(taskId);
            if (result.status === 'completed') {
                clearInterval(interval);
                renderBacktestResults(result);
                showToast('Backtest completed', 'success');
            } else if (result.status === 'failed') {
                clearInterval(interval);
                showToast('Backtest failed', 'error');
            }
        } catch (e) {
            clearInterval(interval);
            showToast('Failed to get results', 'error');
        }
    }, 2000);
}

function renderBacktestResults(results) {
    const container = document.getElementById('backtestResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="grid grid-cols-2 gap-4 mb-4">
            ${renderStatCard('Total Return', formatPercent(results.total_return || 0))}
            ${renderStatCard('Sharpe Ratio', results.sharpe_ratio || 0)}
            ${renderStatCard('Max Drawdown', formatPercent(results.max_drawdown || 0))}
            ${renderStatCard('Win Rate', formatPercent(results.win_rate || 0))}
        </div>
        <div class="flex gap-2">
            <button class="btn btn-secondary" onclick="downloadBacktest('${results.task_id}')">
                ${Icons.download} Download CSV
            </button>
        </div>
    `;
}

async function downloadBacktest(taskId) {
    try {
        const csv = await Api.downloadBacktestResults(taskId);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `backtest_${taskId}.csv`;
        a.click();
    } catch (e) {
        showToast('Download failed: ' + e.message, 'error');
    }
}

async function renderMLPage() {
    const models = await Api.getMLModels().catch(() => []);
    
    return `
        <div class="grid grid-cols-3 gap-4 mb-6">
            ${models.map(model => `
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">${model.name || model.model_id}</h3>
                        ${renderBadge(model.status || 'ready', model.status === 'training' ? 'warning' : 'success')}
                    </div>
                    <div class="card-body">
                        <p class="text-sm text-muted mb-2">Type: ${model.model_type || 'N/A'}</p>
                        ${model.accuracy ? `<p class="text-sm">Accuracy: ${formatPercent(model.accuracy)}</p>` : ''}
                        <div class="flex gap-2 mt-4">
                            ${model.status === 'ready' ? `
                                <button class="btn btn-sm btn-primary" onclick="deployModel('${model.model_id}')">Deploy</button>
                            ` : ''}
                            ${model.status === 'deployed' ? `
                                <button class="btn btn-sm btn-secondary" onclick="undeployModel('${model.model_id}')">Undeploy</button>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        ${renderCard('Train New Model', `
            <form onsubmit="trainModel(event)">
                <div class="form-group">
                    <label class="form-label">Model Type</label>
                    <select class="form-select" id="modelType">
                        <option value="regressor">Price Predictor</option>
                        <option value="classifier">Trend Classifier</option>
                    </select>
                </div>
                <button type="submit" class="btn btn-primary">Train Model</button>
            </form>
        `)}
    `;
}

async function trainModel(e) {
    e.preventDefault();
    showToast('Training model...', 'info');
    
    try {
        const result = await Api.createMLModel({
            model_type: document.getElementById('modelType').value,
            parameters: {},
            features: ['price', 'volume', 'volatility'],
            target: 'next_price'
        });
        
        showToast('Model training started', 'success');
        renderPage('ml');
    } catch (e) {
        showToast('Training failed: ' + e.message, 'error');
    }
}

async function deployModel(modelId) {
    try {
        await Api.deployMLModel(modelId);
        showToast('Model deployed', 'success');
        renderPage('ml');
    } catch (e) {
        showToast('Deploy failed: ' + e.message, 'error');
    }
}

async function renderOrdersPage() {
    const orders = await Api.getOrders().catch(() => []);
    
    const rows = orders.slice(0, 50).map(order => [
        order.order_id?.substring(0, 8) || '-',
        order.market_id || '-',
        order.side || '-',
        order.quantity || 0,
        formatCurrency(order.price || 0),
        order.status || 'pending',
        formatTimestamp(order.created_at || order.timestamp)
    ]);
    
    return `
        ${renderCard('Submit Order', `
            <form id="orderForm" onsubmit="submitOrder(event)" class="mb-4">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Symbol</label>
                        <input type="text" class="form-input" id="orderSymbol" placeholder="e.g., KXHCOV-23" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Side</label>
                        <select class="form-select" id="orderSide">
                            <option value="buy">Buy</option>
                            <option value="sell">Sell</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Quantity</label>
                        <input type="number" class="form-input" id="orderQuantity" value="1" min="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Price</label>
                        <input type="number" class="form-input" id="orderPrice" placeholder="In cents" step="1">
                    </div>
                </div>
                <button type="submit" class="btn btn-primary">Submit Order</button>
            </form>
        `)}
        ${renderCard('Order History', renderTable(
            ['ID', 'Market', 'Side', 'Qty', 'Price', 'Status', 'Time'],
            rows,
            'No orders yet'
        ))}
    `;
}

async function submitOrder(e) {
    e.preventDefault();
    
    const order = {
        exchange: 'kalshi',
        symbol: document.getElementById('orderSymbol').value,
        side: document.getElementById('orderSide').value,
        quantity: parseInt(document.getElementById('orderQuantity').value),
        order_type: document.getElementById('orderPrice').value ? 'limit' : 'market',
        price: document.getElementById('orderPrice').value ? parseInt(document.getElementById('orderPrice').value) : null
    };
    
    try {
        await Api.submitOrder(order);
        showToast('Order submitted', 'success');
        renderPage('orders');
    } catch (e) {
        showToast('Order failed: ' + e.message, 'error');
    }
}

async function renderJournalPage() {
    const journalData = await Api.getJournalEntries().catch(() => ({entries: []}));
    const entries = journalData.entries || [];
    
    const stats = {
        total: entries.length,
        open: entries.filter(e => e.status === 'open').length,
        closed: entries.filter(e => e.status === 'closed').length,
        totalPnl: entries.reduce((sum, e) => sum + (e.pnl || 0), 0)
    };
    
    const rows = entries.map(entry => [
        entry.date ? new Date(entry.date).toLocaleDateString() : '-',
        entry.market_id || '-',
        entry.direction || '-',
        entry.size || 0,
        entry.entry_price ? formatCurrency(entry.entry_price) : '-',
        entry.exit_price ? formatCurrency(entry.exit_price) : '-',
        formatCurrency(entry.pnl || 0),
        renderBadge(entry.status || 'open', entry.status === 'closed' ? 'success' : 'warning'),
        `<div class="flex gap-2">
            <button class="btn btn-sm btn-secondary" onclick="editJournalEntry(${entry.id})">${Icons.edit}</button>
            <button class="btn btn-sm btn-danger" onclick="deleteJournalEntry(${entry.id})">${Icons.trash}</button>
        </div>`
    ]);
    
    return `
        <div class="grid grid-cols-4 gap-4 mb-6">
            ${renderStatCard('Total Entries', stats.total)}
            ${renderStatCard('Open Positions', stats.open)}
            ${renderStatCard('Closed Trades', stats.closed)}
            ${renderStatCard('Total P&L', formatCurrency(stats.totalPnl), null, '$')}
        </div>
        
        <div class="flex justify-between items-center mb-4">
            <div class="flex gap-2">
                <button class="btn btn-primary" onclick="showJournalModal()">
                    ${Icons.plus} New Entry
                </button>
            </div>
            <div class="search-input">
                ${Icons.search}
                <input type="text" class="form-input" placeholder="Search entries..." oninput="filterJournal(this.value)">
            </div>
        </div>
        
        ${renderCard('Journal Entries', renderTable(
            ['Date', 'Market', 'Dir', 'Size', 'Entry', 'Exit', 'P&L', 'Status', 'Actions'],
            rows,
            'No journal entries yet. Click "New Entry" to add one.'
        ))}
    `;
}

function showJournalModal(entry = null) {
    const modal = document.createElement('div');
    modal.innerHTML = renderJournalModal(entry);
    document.body.appendChild(modal);
    
    document.getElementById('journalEntryForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await saveJournalEntry(entry?.id);
    });
}

async function saveJournalEntry(id = null) {
    const entry = {
        date: document.getElementById('entryDate').value,
        market_id: document.getElementById('entryMarket').value,
        direction: document.getElementById('entryDirection').value,
        entry_price: parseInt(document.getElementById('entryPrice').value) || null,
        exit_price: parseInt(document.getElementById('entryExitPrice').value) || null,
        size: parseInt(document.getElementById('entrySize').value) || 0,
        pnl: parseInt(document.getElementById('entryPnl').value) || 0,
        status: document.getElementById('entryStatus').value,
        notes: document.getElementById('entryNotes').value,
        tags: document.getElementById('entryTags').value.split(',').map(t => t.trim()).filter(t => t)
    };
    
    try {
        if (id) {
            await Api.updateJournalEntry(id, entry);
            showToast('Entry updated', 'success');
        } else {
            await Api.createJournalEntry(entry);
            showToast('Entry created', 'success');
        }
        closeJournalModal();
        renderPage('journal');
    } catch (e) {
        showToast('Failed to save: ' + e.message, 'error');
    }
}

async function editJournalEntry(id) {
    const data = await Api.getJournalEntries().catch(() => ({entries: []}));
    const entry = data.entries?.find(e => e.id === id);
    if (entry) {
        showJournalModal(entry);
    }
}

async function deleteJournalEntry(id) {
    if (!confirm('Delete this journal entry?')) return;
    
    try {
        await Api.deleteJournalEntry(id);
        showToast('Entry deleted', 'success');
        renderPage('journal');
    } catch (e) {
        showToast('Failed to delete: ' + e.message, 'error');
    }
}

function filterJournal(query) {
    const rows = document.querySelectorAll('#journalTable tbody tr');
    const q = query.toLowerCase();
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(q) ? '' : 'none';
    });
}

async function renderSettingsPage() {
    const [balance] = await Promise.all([
        Api.getPaperBalance().catch(() => ({balance: 10000}))
    ]);
    
    return `
        <div class="grid grid-cols-2 gap-6">
            ${renderCard('Paper Trading', `
                <div class="mb-4">
                    <p class="text-muted mb-2">Current Balance</p>
                    <p class="text-lg font-semibold">${formatCurrency(balance.balance || balance.balance_dollars * 100 || 10000)}</p>
                </div>
                <button class="btn btn-danger" onclick="resetPaper()">Reset Balance</button>
            `)}
            
            ${renderCard('Notifications', `
                <div class="flex justify-between items-center mb-4">
                    <span>Slack</span>
                    <div class="toggle" onclick="this.classList.toggle('active')"></div>
                </div>
                <div class="flex justify-between items-center mb-4">
                    <span>Discord</span>
                    <div class="toggle" onclick="this.classList.toggle('active')"></div>
                </div>
                <div class="flex justify-between items-center">
                    <span>Telegram</span>
                    <div class="toggle" onclick="this.classList.toggle('active')"></div>
                </div>
            `)}
        </div>
    `;
}

async function resetPaper() {
    if (!confirm('Reset paper trading balance to $10,000?')) return;
    
    try {
        await Api.resetPaper();
        showToast('Paper balance reset', 'success');
        renderPage('settings');
    } catch (e) {
        showToast('Reset failed: ' + e.message, 'error');
    }
}

// Calculator Page
let calcDisplay = '0';
let calcHistory = [];

function renderCalculatorPage() {
    return `
        <div class="grid grid-cols-2 gap-6">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Trade Calculator</h3>
                </div>
                <div class="card-body">
                    <div class="calculator-result" id="calcDisplay">${calcDisplay}</div>
                    <div class="calculator-grid">
                        <button onclick="calcInput('7')">7</button>
                        <button onclick="calcInput('8')">8</button>
                        <button onclick="calcInput('9')">9</button>
                        <button onclick="calcInput('/')">÷</button>
                        <button onclick="calcInput('4')">4</button>
                        <button onclick="calcInput('5')">5</button>
                        <button onclick="calcInput('6')">6</button>
                        <button onclick="calcInput('*')">×</button>
                        <button onclick="calcInput('1')">1</button>
                        <button onclick="calcInput('2')">2</button>
                        <button onclick="calcInput('3')">3</button>
                        <button onclick="calcInput('-')">−</button>
                        <button onclick="calcInput('0')">0</button>
                        <button onclick="calcInput('.')">.</button>
                        <button onclick="calcEquals()" class="equals">=</button>
                        <button onclick="calcInput('+')">+</button>
                    </div>
                    <div class="flex gap-2">
                        <button class="btn btn-secondary" onclick="calcClear()">Clear</button>
                        <button class="btn btn-danger" onclick="calcClearAll()">Clear All</button>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Trade Calculator</h3>
                </div>
                <div class="card-body">
                    <div class="form-group">
                        <label class="form-label">Entry Price (cents)</label>
                        <input type="number" class="form-input" id="tradeEntry" placeholder="e.g., 50" oninput="calculateTrade()">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Exit Price (cents)</label>
                        <input type="number" class="form-input" id="tradeExit" placeholder="e.g., 55" oninput="calculateTrade()">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Contracts</label>
                        <input type="number" class="form-input" id="tradeContracts" placeholder="e.g., 100" oninput="calculateTrade()">
                    </div>
                    <div class="calculator-summary">
                        <div class="calc-summary-item">
                            <span>Profit/Contract</span>
                            <span id="profitPerContract">$0.00</span>
                        </div>
                        <div class="calc-summary-item">
                            <span>Total P&L</span>
                            <span id="totalPnl">$0.00</span>
                        </div>
                        <div class="calc-summary-item">
                            <span>ROI</span>
                            <span id="tradeRoi">0%</span>
                        </div>
                        <div class="calc-summary-item">
                            <span>Fees (1%)</span>
                            <span id="tradeFees">$0.00</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function calcInput(val) {
    if (calcDisplay === '0' && val !== '.') {
        calcDisplay = val;
    } else {
        calcDisplay += val;
    }
    document.getElementById('calcDisplay').textContent = calcDisplay;
}

function calcEquals() {
    try {
        const result = eval(calcDisplay.replace(/×/g, '*').replace(/−/g, '-').replace(/÷/g, '/'));
        calcHistory.push(`${calcDisplay} = ${result}`);
        calcDisplay = result.toString();
        document.getElementById('calcDisplay').textContent = calcDisplay;
    } catch (e) {
        calcDisplay = 'Error';
        document.getElementById('calcDisplay').textContent = calcDisplay;
        setTimeout(() => { calcDisplay = '0'; }, 1500);
    }
}

function calcClear() {
    calcDisplay = calcDisplay.slice(0, -1) || '0';
    document.getElementById('calcDisplay').textContent = calcDisplay;
}

function calcClearAll() {
    calcDisplay = '0';
    document.getElementById('calcDisplay').textContent = calcDisplay;
}

function calculateTrade() {
    const entry = parseFloat(document.getElementById('tradeEntry').value) || 0;
    const exit = parseFloat(document.getElementById('tradeExit').value) || 0;
    const contracts = parseInt(document.getElementById('tradeContracts').value) || 0;
    
    const profitPerContract = (exit - entry);
    const totalPnl = profitPerContract * contracts;
    const fees = (entry * contracts * 0.01) + (exit * contracts * 0.01);
    const roi = entry > 0 ? ((exit - entry) / entry) * 100 : 0;
    
    document.getElementById('profitPerContract').textContent = formatCurrency(profitPerContract);
    document.getElementById('totalPnl').textContent = formatCurrency(totalPnl);
    document.getElementById('tradeRoi').textContent = roi.toFixed(2) + '%';
    document.getElementById('tradeRoi').className = roi >= 0 ? 'text-positive' : 'text-negative';
    document.getElementById('tradeFees').textContent = formatCurrency(fees);
}

// Alerts Page
let alerts = JSON.parse(localStorage.getItem('marketAlerts') || '[]');

async function renderAlertsPage() {
    const stats = {
        total: alerts.length,
        active: alerts.filter(a => a.status === 'active').length,
        triggered: alerts.filter(a => a.status === 'triggered').length
    };
    
    const rows = alerts.map(alert => `
        <div class="alert-item ${alert.status === 'triggered' ? 'triggered' : ''}">
            <div class="alert-condition">
                <div class="alert-market">${alert.market}</div>
                <div class="alert-rule">${alert.condition} ${alert.target}</div>
            </div>
            <span class="alert-status ${alert.status}">${alert.status}</span>
            <button class="btn btn-sm btn-danger" onclick="deleteAlert(${alert.id})">${Icons.trash}</button>
        </div>
    `).join('');
    
    return `
        <div class="grid grid-cols-3 gap-4 mb-6">
            ${renderStatCard('Total Alerts', stats.total)}
            ${renderStatCard('Active', stats.active)}
            ${renderStatCard('Triggered', stats.triggered)}
        </div>
        
        <div class="card mb-6">
            <div class="card-header">
                <h3 class="card-title">Create Alert</h3>
            </div>
            <div class="card-body">
                <form onsubmit="createAlert(event)" class="grid grid-cols-4 gap-4">
                    <div class="form-group">
                        <label class="form-label">Market</label>
                        <input type="text" class="form-input" id="alertMarket" placeholder="e.g., KXHCOV-23" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Condition</label>
                        <select class="form-select" id="alertCondition">
                            <option value="above">Price Above</option>
                            <option value="below">Price Below</option>
                            <option value="spread_above">Spread Above</option>
                            <option value="spread_below">Spread Below</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Target</label>
                        <input type="number" class="form-input" id="alertTarget" placeholder="cents" required>
                    </div>
                    <div class="form-group flex items-end">
                        <button type="submit" class="btn btn-primary w-full">Create Alert</button>
                    </div>
                </form>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Active Alerts</h3>
            </div>
            <div class="card-body">
                <div class="alerts-list">
                    ${rows || '<p class="text-muted text-center py-4">No alerts configured</p>'}
                </div>
            </div>
        </div>
    `;
}

function createAlert(e) {
    e.preventDefault();
    const alert = {
        id: Date.now(),
        market: document.getElementById('alertMarket').value,
        condition: document.getElementById('alertCondition').value,
        target: parseInt(document.getElementById('alertTarget').value),
        status: 'active',
        created: new Date().toISOString()
    };
    alerts.push(alert);
    localStorage.setItem('marketAlerts', JSON.stringify(alerts));
    showToast('Alert created', 'success');
    renderPage('alerts');
}

function deleteAlert(id) {
    alerts = alerts.filter(a => a.id !== id);
    localStorage.setItem('marketAlerts', JSON.stringify(alerts));
    showToast('Alert deleted', 'success');
    renderPage('alerts');
}

// Advanced Orders Page
let selectedOrderType = 'stop_loss';

async function renderAdvancedOrdersPage() {
    const ordersData = await Api.getAdvancedOrders().catch(() => ({orders: []}));
    const orders = ordersData.orders || [];
    
    const stats = {
        total: orders.length,
        active: orders.filter(o => o.status === 'active').length,
        executed: orders.filter(o => o.status === 'executed').length,
        cancelled: orders.filter(o => o.status === 'cancelled').length
    };
    
    const orderTypeCards = [
        { type: 'stop_loss', name: 'Stop Loss', icon: '↓', desc: 'Trigger when price falls below threshold' },
        { type: 'take_profit', name: 'Take Profit', icon: '↑', desc: 'Trigger when price rises above threshold' },
        { type: 'oco', name: 'OCO', icon: '⇅', desc: 'One order triggers, other cancels' },
        { type: 'trailing_stop', name: 'Trailing Stop', icon: '↗', desc: 'Stop moves with price' },
        { type: 'twap', name: 'TWAP', icon: '≡', desc: 'Split order over time' }
    ];
    
    const rows = orders.map(order => [
        order.type.replace('_', ' ').toUpperCase(),
        order.market_id || '-',
        order.side || '-',
        order.quantity || order.total_quantity || '-',
        order.stop_price ? formatCurrency(order.stop_price) : (order.target_price ? formatCurrency(order.target_price) : '-'),
        renderBadge(order.status, order.status === 'active' ? 'warning' : order.status === 'executed' ? 'success' : 'neutral'),
        `<button class="btn btn-sm btn-danger" onclick="cancelAdvancedOrder('${order.order_id}')" ${order.status !== 'active' ? 'disabled' : ''}>Cancel</button>`
    ]);
    
    return `
        <div class="grid grid-cols-4 gap-4 mb-6">
            ${renderStatCard('Total Orders', stats.total)}
            ${renderStatCard('Active', stats.active)}
            ${renderStatCard('Executed', stats.executed)}
            ${renderStatCard('Cancelled', stats.cancelled)}
        </div>
        
        <div class="grid grid-cols-5 gap-2 mb-6">
            ${orderTypeCards.map(t => `
                <div class="card cursor-pointer order-type-card ${selectedOrderType === t.type ? 'selected' : ''}" 
                     onclick="selectOrderType('${t.type}')">
                    <div class="card-body text-center p-3">
                        <div class="text-2xl mb-1">${t.icon}</div>
                        <div class="font-medium text-sm">${t.name}</div>
                        <div class="text-xs text-muted">${t.desc}</div>
                    </div>
                </div>
            `).join('')}
        </div>
        
        <div class="card mb-6">
            <div class="card-header">
                <h3 class="card-title">Create ${orderTypeCards.find(t => t.type === selectedOrderType)?.name || 'Order'}</h3>
            </div>
            <div class="card-body">
                <form onsubmit="createAdvancedOrder(event)" class="grid grid-cols-4 gap-4">
                    <div class="form-group">
                        <label class="form-label">Market ID</label>
                        <input type="text" class="form-input" id="advMarket" placeholder="e.g., KXHCOV-23" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Side</label>
                        <select class="form-select" id="advSide">
                            <option value="buy">Buy</option>
                            <option value="sell">Sell</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Quantity</label>
                        <input type="number" class="form-input" id="advQuantity" placeholder="Contracts" required>
                    </div>
                    ${selectedOrderType !== 'twap' ? `
                    <div class="form-group">
                        <label class="form-label">${selectedOrderType === 'oco' ? 'Stop Price' : selectedOrderType === 'trailing_stop' ? 'Activation Price' : 'Stop Price'}</label>
                        <input type="number" class="form-input" id="advStopPrice" placeholder="cents" required>
                    </div>
                    ` : ''}
                    ${selectedOrderType === 'oco' ? `
                    <div class="form-group">
                        <label class="form-label">Target Price</label>
                        <input type="number" class="form-input" id="advTargetPrice" placeholder="cents" required>
                    </div>
                    ` : ''}
                    ${selectedOrderType === 'trailing_stop' ? `
                    <div class="form-group">
                        <label class="form-label">Trail Amount</label>
                        <input type="number" class="form-input" id="advTrailAmount" placeholder="cents" required>
                    </div>
                    ` : ''}
                    ${selectedOrderType === 'twap' ? `
                    <div class="form-group">
                        <label class="form-label">Duration (seconds)</label>
                        <input type="number" class="form-input" id="advDuration" value="3600">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Interval (seconds)</label>
                        <input type="number" class="form-input" id="advInterval" value="60">
                    </div>
                    ` : ''}
                    <div class="form-group flex items-end">
                        <button type="submit" class="btn btn-primary w-full">Create Order</button>
                    </div>
                </form>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Advanced Orders</h3>
            </div>
            <div class="card-body">
                ${renderTable(
                    ['Type', 'Market', 'Side', 'Qty', 'Price', 'Status', 'Actions'],
                    rows,
                    'No advanced orders yet'
                )}
            </div>
        </div>
        
        <style>
            .order-type-card { transition: all 0.2s; }
            .order-type-card:hover { border-color: var(--primary); }
            .order-type-card.selected { border-color: var(--primary); background-color: var(--primary-light); }
        </style>
    `;
}

function selectOrderType(type) {
    selectedOrderType = type;
    renderPage('advanced-orders');
}

async function createAdvancedOrder(e) {
    e.preventDefault();
    
    const baseOrder = {
        market_id: document.getElementById('advMarket').value,
        side: document.getElementById('advSide').value,
        quantity: parseInt(document.getElementById('advQuantity').value)
    };
    
    try {
        let result;
        switch(selectedOrderType) {
            case 'stop_loss':
                result = await Api.createStopLossOrder({
                    ...baseOrder,
                    stop_price: parseInt(document.getElementById('advStopPrice').value)
                });
                break;
            case 'take_profit':
                result = await Api.createTakeProfitOrder({
                    ...baseOrder,
                    target_price: parseInt(document.getElementById('advStopPrice').value)
                });
                break;
            case 'oco':
                result = await Api.createOCOOrder({
                    ...baseOrder,
                    stop_price: parseInt(document.getElementById('advStopPrice').value),
                    target_price: parseInt(document.getElementById('advTargetPrice').value)
                });
                break;
            case 'trailing_stop':
                result = await Api.createTrailingStopOrder({
                    ...baseOrder,
                    activation_price: parseInt(document.getElementById('advStopPrice').value),
                    trail_amount: parseInt(document.getElementById('advTrailAmount').value)
                });
                break;
            case 'twap':
                result = await Api.createTWAPOrder({
                    ...baseOrder,
                    total_quantity: baseOrder.quantity,
                    duration_seconds: parseInt(document.getElementById('advDuration').value),
                    interval_seconds: parseInt(document.getElementById('advInterval').value)
                });
                break;
        }
        
        showToast('Order created successfully', 'success');
        renderPage('advanced-orders');
    } catch (err) {
        showToast('Failed to create order: ' + err.message, 'error');
    }
}

async function cancelAdvancedOrder(orderId) {
    if (!confirm('Cancel this order?')) return;
    
    try {
        await Api.cancelAdvancedOrder(orderId);
        showToast('Order cancelled', 'success');
        renderPage('advanced-orders');
    } catch (err) {
        showToast('Failed to cancel: ' + err.message, 'error');
    }
}

// Charts Page
let chart = null;
let selectedMarket = '';
let selectedTimeframe = '15';
let selectedChartType = 'candlestick';

const timeframes = [
    { value: '1', label: '1m' },
    { value: '5', label: '5m' },
    { value: '15', label: '15m' },
    { value: '60', label: '1h' },
    { value: '240', label: '4h' },
    { value: '1440', label: '1D' }
];

async function renderChartsPage() {
    const orderbooks = await Api.getOrderbooks().catch(() => []);
    const markets = orderbooks.map(ob => ob.market_id).slice(0, 50);
    
    if (!selectedMarket && markets.length > 0) {
        selectedMarket = markets[0];
    }
    
    return `
        <div class="charts-controls mb-4 flex flex-wrap gap-4 items-center">
            <div class="form-group mb-0">
                <label class="form-label">Market</label>
                <select class="form-select" id="chartMarket" onchange="changeChartMarket(this.value)">
                    ${markets.map(m => `<option value="${m}" ${m === selectedMarket ? 'selected' : ''}>${m}</option>`).join('')}
                </select>
            </div>
            <div class="form-group mb-0">
                <label class="form-label">Timeframe</label>
                <select class="form-select" id="chartTimeframe" onchange="changeTimeframe(this.value)">
                    ${timeframes.map(t => `<option value="${t.value}" ${t.value === selectedTimeframe ? 'selected' : ''}>${t.label}</option>`).join('')}
                </select>
            </div>
            <div class="form-group mb-0">
                <label class="form-label">Chart Type</label>
                <select class="form-select" id="chartType" onchange="changeChartType(this.value)">
                    <option value="candlestick" ${selectedChartType === 'candlestick' ? 'selected' : ''}>Candlestick</option>
                    <option value="line" ${selectedChartType === 'line' ? 'selected' : ''}>Line</option>
                    <option value="area" ${selectedChartType === 'area' ? 'selected' : ''}>Area</option>
                </select>
            </div>
            <div class="flex gap-2 items-end">
                <button class="btn btn-secondary btn-sm" onclick="toggleIndicator('sma')">SMA</button>
                <button class="btn btn-secondary btn-sm" onclick="toggleIndicator('ema')">EMA</button>
                <button class="btn btn-secondary btn-sm" onclick="toggleIndicator('rsi')">RSI</button>
                <button class="btn btn-secondary btn-sm" onclick="toggleIndicator('bb')">BB</button>
            </div>
        </div>
        
        <div class="card">
            <div class="card-body" style="height: 500px;">
                <div id="chartContainer" style="width: 100%; height: 100%;"></div>
            </div>
        </div>
        
        <div class="grid grid-cols-3 gap-4 mt-4">
            <div class="card">
                <div class="card-header"><h3 class="card-title">Statistics</h3></div>
                <div class="card-body" id="chartStats">
                    <p class="text-muted">Select a market to view stats</p>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><h3 class="card-title">Order Book Depth</h3></div>
                <div class="card-body" style="height: 200px;">
                    <canvas id="depthChart"></canvas>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><h3 class="card-title">Recent Trades</h3></div>
                <div class="card-body" id="recentTrades">
                    <p class="text-muted">No recent trades</p>
                </div>
            </div>
        </div>
    `;
}

function changeChartMarket(market) {
    selectedMarket = market;
    renderChart();
}

function changeTimeframe(tf) {
    selectedTimeframe = tf;
    renderChart();
}

function changeChartType(type) {
    selectedChartType = type;
    renderChart();
}

function renderChart() {
    if (!selectedMarket) return;
    
    const container = document.getElementById('chartContainer');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Generate mock OHLC data
    const data = generateMockOHLCData(100);
    
    if (typeof LightweightCharts !== 'undefined') {
        const chartOptions = {
            layout: {
                textColor: document.body.classList.contains('dark') ? '#d1d5db' : '#374151',
                background: { type: 'solid', color: getComputedStyle(document.body).getPropertyValue('--bg-card') }
            },
            grid: {
                vertLines: { color: document.body.classList.contains('dark') ? '#374151' : '#e5e7eb' },
                horzLines: { color: document.body.classList.contains('dark') ? '#374151' : '#e5e7eb' }
            },
            width: container.clientWidth,
            height: container.clientHeight || 450
        };
        
        chart = LightweightCharts.createChart(container, chartOptions);
        
        let series;
        const seriesOptions = {
            upColor: '#10b981',
            downColor: '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444'
        };
        
        if (selectedChartType === 'candlestick') {
            series = chart.addCandlestickSeries(seriesOptions);
            series.setData(data);
        } else if (selectedChartType === 'line') {
            series = chart.addLineSeries({ color: '#10b981', lineWidth: 2 });
            series.setData(data.map(d => ({ time: d.time, value: d.close })));
        } else {
            series = chart.addAreaSeries({
                topColor: 'rgba(16, 185, 129, 0.4)',
                bottomColor: 'rgba(16, 185, 129, 0.0)',
                lineColor: '#10b981',
                lineWidth: 2
            });
            series.setData(data.map(d => ({ time: d.time, value: d.close })));
        }
        
        chart.timeScale().fitContent();
    } else {
        container.innerHTML = '<p class="text-center text-muted p-4">Chart library not loaded. Using fallback chart.</p>';
        // Fallback to Chart.js
        const canvas = document.createElement('canvas');
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d');
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => d.time),
                datasets: [{
                    label: selectedMarket,
                    data: data.map(d => d.close),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
}

function generateMockOHLCData(count) {
    const data = [];
    let price = 50 + Math.random() * 20;
    const now = Math.floor(Date.now() / 1000);
    const interval = parseInt(selectedTimeframe) * 60;
    
    for (let i = count; i >= 0; i--) {
        const time = now - (i * interval);
        const volatility = 2;
        const open = price;
        const close = price + (Math.random() - 0.5) * volatility * 2;
        const high = Math.max(open, close) + Math.random() * volatility;
        const low = Math.min(open, close) - Math.random() * volatility;
        
        data.push({
            time: time,
            open: open,
            high: high,
            low: low,
            close: close
        });
        
        price = close;
    }
    
    return data;
}

let activeIndicators = new Set();

function toggleIndicator(indicator) {
    if (activeIndicators.has(indicator)) {
        activeIndicators.delete(indicator);
    } else {
        activeIndicators.add(indicator);
    }
    showToast(`${indicator.toUpperCase()} ${activeIndicators.has(indicator) ? 'added' : 'removed'}`, 'info');
    renderChart();
}

// Quick Actions
function toggleQuickActions() {
    AppState.quickActionsOpen = !AppState.quickActionsOpen;
    const existing = document.querySelector('.quick-actions-panel');
    if (existing) {
        existing.remove();
    }
    
    if (AppState.quickActionsOpen) {
        const panel = document.createElement('div');
        panel.className = 'quick-actions-panel';
        panel.innerHTML = `
            <div class="quick-actions-header">
                <h3>Quick Actions</h3>
                <button class="btn btn-icon btn-sm" onclick="toggleQuickActions()">${Icons.close}</button>
            </div>
            <div class="quick-actions-list">
                <div class="quick-action-item" onclick="navigateTo('dashboard'); toggleQuickActions();">
                    <div class="quick-action-icon">${Icons.dashboard}</div>
                    <div class="quick-action-text">
                        <div class="quick-action-title">Dashboard</div>
                        <div class="quick-action-desc">View overview</div>
                    </div>
                </div>
                <div class="quick-action-item" onclick="navigateTo('orders'); toggleQuickActions();">
                    <div class="quick-action-icon">${Icons.orders}</div>
                    <div class="quick-action-text">
                        <div class="quick-action-title">New Order</div>
                        <div class="quick-action-desc">Place an order</div>
                    </div>
                </div>
                <div class="quick-action-item" onclick="navigateTo('calculator'); toggleQuickActions();">
                    <div class="quick-action-icon">${Icons.trade}</div>
                    <div class="quick-action-text">
                        <div class="quick-action-title">Calculator</div>
                        <div class="quick-action-desc">Calculate trades</div>
                    </div>
                </div>
                <div class="quick-action-item" onclick="toggleBot(); toggleQuickActions();">
                    <div class="quick-action-icon">${AppState.botRunning ? Icons.stop : Icons.play}</div>
                    <div class="quick-action-text">
                        <div class="quick-action-title">${AppState.botRunning ? 'Stop Bot' : 'Start Bot'}</div>
                        <div class="quick-action-desc">${AppState.botRunning ? 'Stop trading' : 'Start trading'}</div>
                    </div>
                </div>
                <div class="quick-action-item" onclick="navigateTo('alerts'); toggleQuickActions();">
                    <div class="quick-action-icon">${Icons.bell}</div>
                    <div class="quick-action-text">
                        <div class="quick-action-title">Market Alerts</div>
                        <div class="quick-action-desc">Set price alerts</div>
                    </div>
                </div>
                <div class="quick-action-item" onclick="showKeyboardShortcuts(); toggleQuickActions();">
                    <div class="quick-action-icon">${Icons.info}</div>
                    <div class="quick-action-text">
                        <div class="quick-action-title">Keyboard Shortcuts</div>
                        <div class="quick-action-desc">View all shortcuts</div>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(panel);
    }
}

async function renderSystemPage() {
    const [health, status] = await Promise.all([
        Api.getHealth().catch(() => ({})),
        Api.getBotStatus().catch(() => ({}))
    ]);
    
    const components = [
        { name: 'Exchange', status: status.exchange_active !== false ? 'healthy' : 'down' },
        { name: 'WebSocket', status: status.ws_connected !== false ? 'healthy' : 'degraded' },
        { name: 'Database', status: health.database !== false ? 'healthy' : 'down' },
        { name: 'Rate Limiter', status: health.rate_limiter !== false ? 'healthy' : 'degraded' }
    ];
    
    const circuitBreaker = status.circuit_breaker || {};
    
    return `
        <div class="grid grid-cols-4 gap-4 mb-6">
            ${components.map(c => `
                <div class="stat-card">
                    <div class="stat-label">${c.name}</div>
                    <div class="flex items-center gap-2 mt-2">
                        <span class="status-dot" style="color: ${c.status === 'healthy' ? '#10b981' : (c.status === 'degraded' ? '#f59e0b' : '#ef4444')}"></span>
                        <span class="font-medium">${c.status.charAt(0).toUpperCase() + c.status.slice(1)}</span>
                    </div>
                </div>
            `).join('')}
        </div>
        
        <div class="grid grid-cols-2 gap-6">
            ${renderCard('Circuit Breaker', `
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="text-muted text-sm">State</div>
                        <div class="font-medium">${circuitBreaker.state || 'closed'}</div>
                    </div>
                    <div>
                        <div class="text-muted text-sm">Consecutive Failures</div>
                        <div class="font-medium">${circuitBreaker.failures || 0}</div>
                    </div>
                    <div>
                        <div class="text-muted text-sm">Last Failure</div>
                        <div class="font-medium">${formatTimeAgo(circuitBreaker.last_failure)}</div>
                    </div>
                    <div>
                        <div class="text-muted text-sm">Uptime</div>
                        <div class="font-medium">${health.uptime || 'N/A'}</div>
                    </div>
                </div>
            `)}
            
            ${renderCard('API Health', `
                <div class="text-sm">
                    <div class="flex justify-between py-2 border-b">
                        <span>Status</span>
                        <span class="text-positive">${health.status || 'ok'}</span>
                    </div>
                    <div class="flex justify-between py-2 border-b">
                        <span>Version</span>
                        <span>${health.version || '2.0.0'}</span>
                    </div>
                    <div class="flex justify-between py-2">
                        <span>Errors (24h)</span>
                        <span>${health.error_count || 0}</span>
                    </div>
                </div>
            `)}
        </div>
    `;
}

async function renderAutoTradingPage() {
    const status = await Api.getAutoTradingStatus().catch(() => ({ enabled: false, config: {}, history: [] }));
    
    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            ${renderCard('Auto-Trading Status', `
                <div class="flex items-center justify-between">
                    <div class="text-lg font-medium">${status.enabled ? 'Running' : 'Stopped'}</div>
                    <button class="btn ${status.enabled ? 'btn-danger' : 'btn-primary'}" 
                        onclick="toggleAutoTrading(${!status.enabled})">
                        ${status.enabled ? 'Stop' : 'Start'}
                    </button>
                </div>
            `)}
            
            ${renderCard('Configuration', `
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span>Max Position Size</span>
                        <span>${status.config?.max_position_size || 100}</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Auto-Close on Risk</span>
                        <span>${status.config?.auto_close_on_risk ? 'Yes' : 'No'}</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Min Profit Threshold</span>
                        <span>${((status.config?.min_profit_threshold || 0.01) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Max Slippage</span>
                        <span>${((status.config?.max_slippage || 0.02) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `)}
        </div>
        
        ${renderCard('Recent Activity', `
            <div class="space-y-2">
                ${(status.history || []).length === 0 ? '<div class="text-muted">No recent activity</div>' : ''}
                ${(status.history || []).map(h => `
                    <div class="flex justify-between py-2 border-b">
                        <span>${h.action}</span>
                        <span class="text-muted">${new Date(h.timestamp).toLocaleString()}</span>
                    </div>
                `).join('')}
            </div>
        `)}
    `;
}

async function toggleAutoTrading(enable) {
    try {
        if (enable) {
            await Api.enableAutoTrading({ enabled: true, max_position_size: 100, auto_close_on_risk: true, min_profit_threshold: 0.01, max_slippage: 0.02, strategies: ['statistical_arbitrage'] });
        } else {
            await Api.disableAutoTrading();
        }
        renderPage(AppState.currentPage);
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function renderPositionSizingPage() {
    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            ${renderCard('Kelly Criterion Calculator', `
                <form id="kellyForm" class="space-y-4">
                    <div>
                        <label class="block text-sm mb-1">Account Balance ($)</label>
                        <input type="number" name="account_balance" class="input" value="10000" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Win Rate (%)</label>
                        <input type="number" name="win_rate" class="input" value="60" min="0" max="100" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Average Win ($)</label>
                        <input type="number" name="avg_win" class="input" value="100" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Average Loss ($)</label>
                        <input type="number" name="avg_loss" class="input" value="50" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Kelly Fraction</label>
                        <select name="kelly_fraction" class="input">
                            <option value="0.25">25% (Conservative)</option>
                            <option value="0.5" selected>50% (Moderate)</option>
                            <option value="0.75">75% (Aggressive)</option>
                            <option value="1">100% (Full Kelly)</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary w-full">Calculate</button>
                </form>
            `)}
            
            ${renderCard('Results', `
                <div id="kellyResults" class="space-y-3">
                    <div class="text-muted">Enter your trading statistics and click Calculate</div>
                </div>
            `)}
        </div>
        
        <script>
            document.getElementById('kellyForm')?.addEventListener('submit', async (e) => {
                e.preventDefault();
                const form = e.target;
                const data = {
                    account_balance: parseFloat(form.account_balance.value),
                    win_rate: parseFloat(form.win_rate.value) / 100,
                    avg_win: parseFloat(form.avg_win.value),
                    avg_loss: parseFloat(form.avg_loss.value),
                    kelly_fraction: parseFloat(form.kelly_fraction.value),
                    risk_percent: 0.02
                };
                try {
                    const result = await Api.calculatePositionSize(data);
                    document.getElementById('kellyResults').innerHTML = \`
                        <div class="flex justify-between py-2 border-b">
                            <span>Kelly %</span>
                            <span class="font-medium">\${result.kelly_percent}%</span>
                        </div>
                        <div class="flex justify-between py-2 border-b">
                            <span>Fractional Kelly</span>
                            <span class="font-medium">\${result.fractional_kelly_percent}%</span>
                        </div>
                        <div class="flex justify-between py-2 border-b">
                            <span>Recommended Position</span>
                            <span class="font-medium">\${result.recommended_position_size.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between py-2 border-b">
                            <span>Risk-Adjusted Position</span>
                            <span class="font-medium">\${result.risk_adjusted_position_size.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between py-2 border-b">
                            <span>Risk Amount (2%)</span>
                            <span class="font-medium">\${result.risk_amount.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between py-2">
                            <span>Recommendation</span>
                            <span class="badge badge-\${result.kelly_recommendation === 'no_trade' ? 'danger' : 'success'}">\${result.kelly_recommendation}</span>
                        </div>
                    \`;
                } catch (err) {
                    document.getElementById('kellyResults').innerHTML = '<div class="text-negative">Error: ' + err.message + '</div>';
                }
            });
        </script>
    `;
}

async function renderSpreadAlertsPage() {
    const alerts = await Api.getSpreadAlerts().catch(() => ({ alerts: [] }));
    
    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            ${renderCard('Create Spread Alert', `
                <form id="spreadAlertForm" class="space-y-4">
                    <div>
                        <label class="block text-sm mb-1">Market A</label>
                        <input type="text" name="market_a" class="input" placeholder="e.g., KALSHI:BTC-25DEC" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Market B</label>
                        <input type="text" name="market_b" class="input" placeholder="e.g., KALSHI:ETH-25DEC" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Threshold (%)</label>
                        <input type="number" name="threshold" class="input" value="1" step="0.1" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Direction</label>
                        <select name="direction" class="input">
                            <option value="both">Both</option>
                            <option value="above">Above Threshold</option>
                            <option value="below">Below Threshold</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary w-full">Create Alert</button>
                </form>
            `)}
            
            ${renderCard('Active Alerts', `
                <div class="space-y-2" id="alertsList">
                    ${alerts.alerts.length === 0 ? '<div class="text-muted">No spread alerts configured</div>' : ''}
                    ${alerts.alerts.map(a => \`
                        <div class="flex justify-between items-center py-2 border-b">
                            <div>
                                <div class="font-medium">\${a.market_a} / \${a.market_b}</div>
                                <div class="text-sm text-muted">Threshold: \${a.threshold}% | \${a.direction}</div>
                            </div>
                            <button class="btn btn-sm btn-danger" onclick="deleteSpreadAlert(\${a.id})">Delete</button>
                        </div>
                    \`).join('')}
                </div>
            `)}
        </div>
        
        <script>
            document.getElementById('spreadAlertForm')?.addEventListener('submit', async (e) => {
                e.preventDefault();
                const form = e.target;
                const data = {
                    market_a: form.market_a.value,
                    market_b: form.market_b.value,
                    threshold: parseFloat(form.threshold.value),
                    direction: form.direction.value,
                    enabled: true
                };
                try {
                    await Api.createSpreadAlert(data);
                    form.reset();
                    renderPage(AppState.currentPage);
                } catch (err) {
                    alert('Error: ' + err.message);
                }
            });
            
            async function deleteSpreadAlert(id) {
                try {
                    await Api.deleteSpreadAlert(id);
                    renderPage(AppState.currentPage);
                } catch (err) {
                    alert('Error: ' + err.message);
                }
            }
        </script>
    `;
}

async function renderOneClickPage() {
    const status = await Api.getOneClickStatus ? await Api.getOneClickStatus().catch(() => ({ enabled: false })) : { enabled: false };
    
    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            ${renderCard('One-Click Trading', `
                <div class="flex items-center justify-between">
                    <div class="text-lg font-medium">${status.enabled ? 'Enabled' : 'Disabled'}</div>
                    <button class="btn ${status.enabled ? 'btn-danger' : 'btn-primary'}" 
                        onclick="toggleOneClick(${!status.enabled})">
                        ${status.enabled ? 'Disable' : 'Enable'}
                    </button>
                </div>
                <p class="text-sm text-muted mt-4">One-click trading allows rapid trade execution with predefined parameters.</p>
            `)}
            
            ${renderCard('Quick Trade', `
                <form id="quickTradeForm" class="space-y-4">
                    <div>
                        <label class="block text-sm mb-1">Market ID</label>
                        <input type="text" name="market_id" class="input" placeholder="e.g., KALSHI:SPX-25DEC" required>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Side</label>
                        <select name="side" class="input">
                            <option value="buy">Buy</option>
                            <option value="sell">Sell</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm mb-1">Quantity</label>
                        <input type="number" name="quantity" class="input" value="10" required>
                    </div>
                    <div class="flex items-center gap-2">
                        <input type="checkbox" name="use_kelly" id="useKelly" class="checkbox">
                        <label for="useKelly">Use Kelly Criterion for sizing</label>
                    </div>
                    <div id="kellyFields" style="display: none;">
                        <div class="grid grid-cols-3 gap-2">
                            <div>
                                <label class="block text-xs mb-1">Balance ($)</label>
                                <input type="number" name="account_balance" class="input" value="10000">
                            </div>
                            <div>
                                <label class="block text-xs mb-1">Win Rate (%)</label>
                                <input type="number" name="win_rate" class="input" value="60">
                            </div>
                            <div>
                                <label class="block text-xs mb-1">Avg Win ($)</label>
                                <input type="number" name="avg_win" class="input" value="100">
                            </div>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-success w-full" ${!status.enabled ? 'disabled' : ''}>
                        ${status.enabled ? 'Execute Trade' : 'Enable First'}
                    </button>
                </form>
            `)}
        </div>
        
        <script>
            document.getElementById('useKelly')?.addEventListener('change', (e) => {
                document.getElementById('kellyFields').style.display = e.target.checked ? 'block' : 'none';
            });
            
            document.getElementById('quickTradeForm')?.addEventListener('submit', async (e) => {
                e.preventDefault();
                const form = e.target;
                const useKelly = form.use_kelly.checked;
                const data = {
                    market_id: form.market_id.value,
                    side: form.side.value,
                    quantity: parseInt(form.quantity.value),
                    order_type: 'market',
                    use_kelly_sizing: useKelly,
                    account_balance: useKelly ? parseFloat(form.account_balance.value) : null,
                    win_rate: useKelly ? parseFloat(form.win_rate.value) / 100 : null,
                    avg_win: useKelly ? parseFloat(form.avg_win.value) : null,
                    avg_loss: 50
                };
                try {
                    const result = await Api.executeQuickTrade(data);
                    alert('Trade executed: ' + result.order?.order_id);
                    form.reset();
                } catch (err) {
                    alert('Error: ' + err.message);
                }
            });
            
            async function toggleOneClick(enable) {
                try {
                    if (enable) {
                        await Api.enableOneClick();
                    } else {
                        await Api.disableOneClick();
                    }
                    renderPage(AppState.currentPage);
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }
        </script>
    `;
}

async function renderDatabasePage() {
    const stats = await Api.getDatabaseStats().catch(() => ({}));
    
    return `
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            ${renderStatCard('Trades', stats.trades || 0)}
            ${renderStatCard('Positions', stats.positions || 0)}
            ${renderStatCard('Journal Entries', stats.journal_entries || 0)}
            ${renderStatCard('Alerts', stats.alerts || 0)}
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            ${renderCard('Database Info', `
                <div class="space-y-3">
                    <div class="flex justify-between py-2 border-b">
                        <span>Database Path</span>
                        <span class="text-sm">${stats.database_path || 'data/arbitrage.db'}</span>
                    </div>
                    <div class="flex justify-between py-2 border-b">
                        <span>Status</span>
                        <span class="text-positive">Connected</span>
                    </div>
                </div>
            `)}
            
            ${renderCard('Recent Trades', `
                <div id="dbTrades">
                    <div class="text-muted">Loading...</div>
                </div>
            `)}
        </div>
        
        <script>
            Api.getTradesFromDb(10, 0).then(result => {
                const container = document.getElementById('dbTrades');
                if (!result.trades || result.trades.length === 0) {
                    container.innerHTML = '<div class="text-muted">No trades in database</div>';
                    return;
                }
                container.innerHTML = result.trades.map(t => \`
                    <div class="flex justify-between py-2 border-b text-sm">
                        <span>\${t.market_id} (\${t.side})</span>
                        <span>\${t.quantity} @ \${t.price || 'N/A'}</span>
                    </div>
                \`).join('');
            }).catch(() => {
                document.getElementById('dbTrades').innerHTML = '<div class="text-muted">Error loading trades</div>';
            });
        </script>
    `;
}

document.addEventListener('DOMContentLoaded', initApp);

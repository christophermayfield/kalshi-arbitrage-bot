const Icons = {
    dashboard: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>`,
    markets: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>`,
    arbitrage: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v12"/><path d="M6 12h12"/></svg>`,
    portfolio: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.27 6.96L12 12.01l8.73-5.05"/><path d="M12 22.08V12"/></svg>`,
    risk: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
    strategies: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`,
    analytics: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>`,
    backtesting: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
    ml: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5"/><path d="M8.5 8.5v.01"/><path d="M16 15.5v.01"/><path d="M12 12v.01"/><path d="M11 17v.01"/><path d="M7 14v.01"/></svg>`,
    orders: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/><path d="M9 12h6"/><path d="M9 16h6"/></svg>`,
    journal: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><path d="M8 7h8"/><path d="M8 11h6"/></svg>`,
    settings: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
    system: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/><path d="M7 8h2"/><path d="M7 11h4"/></svg>`,
    sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2"/><path d="M12 21v2"/><path d="M4.22 4.22l1.42 1.42"/><path d="M18.36 18.36l1.42 1.42"/><path d="M1 12h2"/><path d="M21 12h2"/><path d="M4.22 19.78l1.42-1.42"/><path d="M18.36 5.64l1.42-1.42"/></svg>`,
    moon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`,
    menu: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>`,
    close: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
    search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>`,
    refresh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>`,
    play: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>`,
    stop: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>`,
    check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    x: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
    chevronDown: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>`,
    chevronUp: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>`,
    alertTriangle: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
    info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
    download: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
    clock: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
    trade: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>`,
    trendingUp: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>`,
    trendingDown: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>`,
    activity: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
    wifi: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>`,
    wifiOff: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/><path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/><path d="M10.71 5.05A16 16 0 0 1 22.58 9"/><path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>`,
    reload: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>`,
    save: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>`,
    edit: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`,
    trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
    plus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
    filter: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>`,
    columns: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/></svg>`,
    chart: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`,
    bell: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>`,
    bellOff: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13.73 21a2 2 0 0 1-3.46 0"/><path d="M18.63 13A17.89 17.89 0 0 1 18 8"/><path d="M6.26 6.26A5.86 5.86 0 0 0 6 8c0 7-3 9-3 9h14"/><path d="M18 8a6 6 0 0 0-9.33-5"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`,
    export: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>`,
    zap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
    database: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
    target: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>`
};

const Pages = {
    dashboard: { title: 'Dashboard', icon: Icons.dashboard, path: '#/' },
    markets: { title: 'Markets', icon: Icons.markets, path: '#/markets' },
    arbitrage: { title: 'Arbitrage', icon: Icons.arbitrage, path: '#/arbitrage' },
    portfolio: { title: 'Portfolio', icon: Icons.portfolio, path: '#/portfolio' },
    risk: { title: 'Risk Analytics', icon: Icons.risk, path: '#/risk' },
    strategies: { title: 'Strategies', icon: Icons.strategies, path: '#/strategies' },
    analytics: { title: 'Analytics', icon: Icons.analytics, path: '#/analytics' },
    backtesting: { title: 'Backtesting', icon: Icons.backtesting, path: '#/backtesting' },
    ml: { title: 'ML Models', icon: Icons.ml, path: '#/ml' },
    orders: { title: 'Orders', icon: Icons.orders, path: '#/orders' },
    calculator: { title: 'Calculator', icon: Icons.trade, path: '#/calculator' },
    alerts: { title: 'Alerts', icon: Icons.bell, path: '#/alerts' },
    'advanced-orders': { title: 'Advanced Orders', icon: Icons.orders, path: '#/advanced-orders' },
    charts: { title: 'Charts', icon: Icons.chart, path: '#/charts' },
    journal: { title: 'Trade Journal', icon: Icons.journal, path: '#/journal' },
    'auto-trading': { title: 'Auto Trading', icon: Icons.play, path: '#/auto-trading' },
    'position-sizing': { title: 'Position Sizing', icon: Icons.trade, path: '#/position-sizing' },
    'spread-alerts': { title: 'Spread Alerts', icon: Icons.bell, path: '#/spread-alerts' },
    'one-click': { title: 'One-Click Trade', icon: Icons.zap, path: '#/one-click' },
    database: { title: 'Database', icon: Icons.database, path: '#/database' },
    settings: { title: 'Settings', icon: Icons.settings, path: '#/settings' },
    system: { title: 'System', icon: Icons.system, path: '#/system' }
};

function renderNav() {
    const nav = document.getElementById('sidebarNav');
    if (!nav) return '';
    
    const sections = [
        { title: 'Main', items: ['dashboard', 'markets', 'arbitrage', 'portfolio'] },
        { title: 'Trading', items: ['auto-trading', 'one-click', 'position-sizing', 'spread-alerts'] },
        { title: 'Analysis', items: ['risk', 'strategies', 'analytics'] },
        { title: 'Tools', items: ['backtesting', 'ml', 'orders'] },
        { title: 'Data', items: ['database', 'journal'] },
        { title: 'System', items: ['settings', 'system'] }
    ];
    
    let html = '';
    sections.forEach(section => {
        html += `<div class="nav-section"><div class="nav-section-title">${section.title}</div>`;
        section.items.forEach(item => {
            const page = Pages[item];
            html += `<a href="${page.path}" class="nav-link" data-page="${item}">
                ${page.icon}
                <span>${page.title}</span>
            </a>`;
        });
        html += '</div>';
    });
    
    return html;
}

function renderPageTitle(title) {
    return `<h1 class="page-title">${title}</h1>`;
}

function renderStatCard(label, value, change = null, prefix = '', suffix = '') {
    const changeClass = change && change > 0 ? 'positive' : (change && change < 0 ? 'negative' : '');
    const changeIcon = change && change > 0 ? Icons.trendingUp : (change && change < 0 ? Icons.trendingDown : '');
    
    return `
        <div class="stat-card">
            <div class="stat-label">${label}</div>
            <div class="stat-value ${changeClass}">${prefix}${typeof value === 'number' ? value.toLocaleString() : value}${suffix}</div>
            ${change !== null ? `
                <div class="stat-change ${changeClass}">
                    ${changeIcon}
                    ${change > 0 ? '+' : ''}${change.toFixed(2)}%
                </div>
            ` : ''}
        </div>
    `;
}

function renderCard(title, content, footer = '') {
    return `
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">${title}</h3>
            </div>
            <div class="card-body">${content}</div>
            ${footer ? `<div class="card-footer">${footer}</div>` : ''}
        </div>
    `;
}

function renderTable(headers, rows, emptyMessage = 'No data available') {
    if (!rows || rows.length === 0) {
        return `<div class="empty-state">${emptyMessage}</div>`;
    }
    
    let html = '<div class="table-container overflow-auto"><table><thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    
    rows.forEach(row => {
        html += '<tr>';
        row.forEach(cell => html += `<td>${cell}</td>`);
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

function renderBadge(text, type = 'neutral') {
    const typeMap = {
        success: 'badge-success',
        warning: 'badge-warning',
        danger: 'badge-danger',
        info: 'badge-info',
        neutral: 'badge-neutral',
        low: 'badge-low',
        medium: 'badge-medium',
        high: 'badge-high'
    };
    return `<span class="badge ${typeMap[type] || 'badge-neutral'}">${text}</span>`;
}

function renderToggle(checked, onChange) {
    return `<div class="toggle ${checked ? 'active' : ''}" onclick="${onChange}"></div>`;
}

function renderTabs(tabs, activeTab, onChange) {
    let html = '<div class="tabs">';
    tabs.forEach((tab, i) => {
        html += `<div class="tab ${activeTab === tab.id ? 'active' : ''}" data-tab="${tab.id}">${tab.label}</div>`;
    });
    html += '</div>';
    return html;
}

function renderLoading() {
    return `<div class="loading"><div class="spinner"></div></div>`;
}

function renderEmptyState(message, icon = '') {
    return `
        <div class="empty-state">
            ${icon || Icons.analytics}
            <p>${message}</p>
        </div>
    `;
}

function formatCurrency(cents, decimals = 2) {
    return '$' + (cents / 100).toFixed(decimals);
}

function formatPercent(value, decimals = 2) {
    return (value * 100).toFixed(decimals) + '%';
}

function formatNumber(value, decimals = 2) {
    return typeof value === 'number' ? value.toFixed(decimals) : value;
}

function formatTimestamp(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return '-';
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now - date) / 1000);
    
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
    if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
    return Math.floor(seconds / 86400) + 'd ago';
}

function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="background:none;border:none;color:inherit;cursor:pointer;margin-left:auto;">${Icons.close}</button>
    `;
    
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Error handling components
function renderError(message, retryCallback = null) {
    return `
        <div class="error-state">
            <div class="error-icon">${Icons.alertTriangle}</div>
            <p class="error-message">${message}</p>
            ${retryCallback ? `
                <button class="btn btn-primary" onclick="${retryCallback}">
                    ${Icons.reload} Retry
                </button>
            ` : ''}
        </div>
    `;
}

function renderLoadingState(message = 'Loading...') {
    return `
        <div class="loading-state">
            <div class="spinner"></div>
            <p>${message}</p>
        </div>
    `;
}

function renderConnectionStatus(state) {
    const states = {
        connected: { icon: Icons.wifi, text: 'Connected', class: 'text-positive' },
        connecting: { icon: Icons.activity, text: 'Connecting...', class: 'text-warning' },
        disconnected: { icon: Icons.wifiOff, text: 'Disconnected', class: 'text-negative' },
        reconnecting: { icon: Icons.reload, text: 'Reconnecting...', class: 'text-warning' },
        failed: { icon: Icons.alertTriangle, text: 'Connection Failed', class: 'text-negative' }
    };
    
    const s = states[state] || states.disconnected;
    return `
        <div class="connection-status ${s.class}">
            ${s.icon}
            <span>${s.text}</span>
        </div>
    `;
}

function showErrorToast(message) {
    showToast(message, 'error');
}

function showSuccessToast(message) {
    showToast(message, 'success');
}

function showWarningToast(message) {
    showToast(message, 'warning');
}

// Orderbook depth visualization
function renderOrderbookDepth(orderbook) {
    const bids = orderbook.bids || [];
    const asks = orderbook.asks || [];
    
    const maxDepth = Math.max(
        ...bids.map(b => b.total || 0),
        ...asks.map(a => a.total || 0)
    );
    
    return `
        <div class="orderbook-depth">
            <div class="depth-row header">
                <span>Price</span>
                <span>Size</span>
                <span>Total</span>
            </div>
            <div class="depth-asks">
                ${asks.slice(0, 10).reverse().map(ask => `
                    <div class="depth-row ask" style="--depth: ${(ask.total || 0) / maxDepth}">
                        <span class="price">${formatCurrency(ask.price)}</span>
                        <span class="size">${ask.count || 0}</span>
                        <span class="total">${ask.total || 0}</span>
                    </div>
                `).join('')}
            </div>
            <div class="depth-spread">
                Spread: ${orderbook.spread || 0}¢ (${((orderbook.spread_percent || 0) * 100).toFixed(2)}%)
            </div>
            <div class="depth-bids">
                ${bids.slice(0, 10).map(bid => `
                    <div class="depth-row bid" style="--depth: ${(bid.total || 0) / maxDepth}">
                        <span class="price">${formatCurrency(bid.price)}</span>
                        <span class="size">${bid.count || 0}</span>
                        <span class="total">${bid.total || 0}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// Trade Journal Entry form
function renderJournalEntryForm(entry = null) {
    return `
        <form id="journalEntryForm" class="journal-form">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Date</label>
                    <input type="datetime-local" class="form-input" id="entryDate" 
                           value="${entry?.date || new Date().toISOString().slice(0, 16)}">
                </div>
                <div class="form-group">
                    <label class="form-label">Market</label>
                    <input type="text" class="form-input" id="entryMarket" 
                           placeholder="e.g., KXHCOV-23" value="${entry?.market_id || ''}">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Direction</label>
                    <select class="form-select" id="entryDirection">
                        <option value="long" ${entry?.direction === 'long' ? 'selected' : ''}>Long</option>
                        <option value="short" ${entry?.direction === 'short' ? 'selected' : ''}>Short</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Entry Price</label>
                    <input type="number" class="form-input" id="entryPrice" 
                           placeholder="In cents" step="1" value="${entry?.entry_price || ''}">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Position Size</label>
                    <input type="number" class="form-input" id="entrySize" 
                           placeholder="Contracts" value="${entry?.size || ''}">
                </div>
                <div class="form-group">
                    <label class="form-label">Exit Price</label>
                    <input type="number" class="form-input" id="entryExitPrice" 
                           placeholder="In cents" step="1" value="${entry?.exit_price || ''}">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">P&L (cents)</label>
                    <input type="number" class="form-input" id="entryPnl" 
                           placeholder="Profit/Loss" value="${entry?.pnl || ''}">
                </div>
                <div class="form-group">
                    <label class="form-label">Status</label>
                    <select class="form-select" id="entryStatus">
                        <option value="open" ${entry?.status === 'open' ? 'selected' : ''}>Open</option>
                        <option value="closed" ${entry?.status === 'closed' ? 'selected' ''}>Closed</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">Notes</label>
                <textarea class="form-input" id="entryNotes" rows="3" 
                          placeholder="Trade rationale, lessons learned, market context...">${entry?.notes || ''}</textarea>
            </div>
            <div class="form-group">
                <label class="form-label">Tags</label>
                <input type="text" class="form-input" id="entryTags" 
                       placeholder="Comma-separated tags" value="${entry?.tags?.join(', ') || ''}">
            </div>
            <div class="form-actions">
                <button type="button" class="btn btn-secondary" onclick="closeJournalModal()">Cancel</button>
                <button type="submit" class="btn btn-primary">${Icons.save} Save Entry</button>
            </div>
        </form>
    `;
}

function renderJournalModal(entry = null) {
    return `
        <div class="modal-overlay" onclick="closeJournalModal()">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3>${entry ? 'Edit' : 'New'} Journal Entry</h3>
                    <button class="btn btn-icon" onclick="closeJournalModal()">${Icons.close}</button>
                </div>
                <div class="modal-body">
                    ${renderJournalEntryForm(entry)}
                </div>
            </div>
        </div>
    `;
}

function closeJournalModal() {
    const modal = document.querySelector('.modal-overlay');
    if (modal) modal.remove();
}

// Notifications Panel
function renderNotificationsPanel(notifications = []) {
    if (!notifications.length) {
        return `
            <div class="notifications-empty">
                ${Icons.bell}
                <p>No notifications</p>
            </div>
        `;
    }
    
    return notifications.map(n => `
        <div class="notification-item ${n.read ? '' : 'unread'}" data-id="${n.id}">
            <div class="notification-icon ${n.type}">
                ${n.type === 'success' ? Icons.check : (n.type === 'error' ? Icons.alertTriangle : Icons.info)}
            </div>
            <div class="notification-content">
                <div class="notification-title">${n.title}</div>
                <div class="notification-message">${n.message}</div>
                <div class="notification-time">${formatTimeAgo(n.timestamp)}</div>
            </div>
            <button class="notification-dismiss" onclick="dismissNotification(${n.id})">${Icons.close}</button>
        </div>
    `).join('');
}

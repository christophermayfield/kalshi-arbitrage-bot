const ChartColors = {
    primary: '#10b981',
    primaryLight: '#d1fae5',
    danger: '#ef4444',
    dangerLight: '#fee2e2',
    warning: '#f59e0b',
    warningLight: '#fef3c7',
    info: '#3b82f6',
    infoLight: '#dbeafe',
    purple: '#8b5cf6',
    purpleLight: '#ede9fe',
    gray: '#6b7280',
    grayLight: '#f3f4f6'
};

const chartInstances = {};

function getChartColors() {
    const isDark = document.body.classList.contains('dark');
    return {
        text: isDark ? '#f9fafb' : '#111827',
        textMuted: isDark ? '#9ca3af' : '#6b7280',
        grid: isDark ? '#374151' : '#e5e7eb',
        bg: isDark ? '#1f2937' : '#ffffff'
    };
}

function createLineChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }
    
    const colors = getChartColors();
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: { color: colors.text }
            }
        },
        scales: {
            x: {
                ticks: { color: colors.textMuted },
                grid: { color: colors.grid }
            },
            y: {
                ticks: { color: colors.textMuted },
                grid: { color: colors.grid }
            }
        },
        ...options
    };
    
    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: defaultOptions
    });
    
    return chartInstances[canvasId];
}

function createDoughnutChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }
    
    const colors = getChartColors();
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'right',
                labels: { color: colors.text }
            }
        },
        ...options
    };
    
    const backgroundColors = [
        ChartColors.primary,
        ChartColors.danger,
        ChartColors.warning,
        ChartColors.info,
        ChartColors.purple
    ];
    
    chartInstances[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: backgroundColors.slice(0, data.length),
                borderWidth: 0
            }]
        },
        options: defaultOptions
    });
    
    return chartInstances[canvasId];
}

function createBarChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }
    
    const colors = getChartColors();
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: { color: colors.text }
            }
        },
        scales: {
            x: {
                ticks: { color: colors.textMuted },
                grid: { display: false }
            },
            y: {
                ticks: { color: colors.textMuted },
                grid: { color: colors.grid }
            }
        },
        ...options
    };
    
    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: defaultOptions
    });
    
    return chartInstances[canvasId];
}

function createAreaChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }
    
    const colors = getChartColors();
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        fill: true,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                ticks: { color: colors.textMuted },
                grid: { display: false }
            },
            y: {
                ticks: { color: colors.textMuted },
                grid: { color: colors.grid }
            }
        },
        ...options
    };
    
    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data,
                borderColor: ChartColors.primary,
                backgroundColor: ChartColors.primaryLight,
                fill: true,
                tension: 0.4
            }]
        },
        options: defaultOptions
    });
    
    return chartInstances[canvasId];
}

function createCorrelationHeatmap(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }
    
    const isDark = document.body.classList.contains('dark');
    const textColor = isDark ? '#f9fafb' : '#111827';
    
    chartInstances[canvasId] = new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Correlation Matrix',
                data: data.values,
                backgroundColor(ctx) {
                    const value = ctx.dataset.data[ctx.dataIndex].v;
                    const alpha = Math.abs(value);
                    if (value > 0) {
                        return `rgba(16, 185, 129, ${alpha})`;
                    } else {
                        return `rgba(239, 68, 68, ${alpha})`;
                    }
                },
                borderColor: isDark ? '#374151' : '#e5e7eb',
                borderWidth: 1,
                width: ({ chart }) => (chart.chartArea || {}).width / data.labels.length - 2,
                height: ({ chart }) => (chart.chartArea || {}).height / data.labels.length - 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title() { return ''; },
                        label(ctx) {
                            const v = ctx.dataset.data[ctx.dataIndex];
                            return `${v.x} vs ${v.y}: ${v.v.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: data.labels,
                    ticks: { color: textColor },
                    grid: { display: false }
                },
                y: {
                    type: 'category',
                    labels: data.labels,
                    ticks: { color: textColor },
                    grid: { display: false }
                }
            }
        }
    });
    
    return chartInstances[canvasId];
}

function updateProfitChart(canvasId, profitData) {
    const labels = profitData.map(d => d.date);
    const profits = profitData.map(d => d.profit);
    
    return createLineChart(canvasId, labels, [{
        label: 'Profit ($)',
        data: profits,
        borderColor: ChartColors.primary,
        backgroundColor: ChartColors.primaryLight,
        fill: true,
        tension: 0.1
    }]);
}

function updateWinLossChart(canvasId, wins, losses) {
    return createDoughnutChart(canvasId, ['Wins', 'Losses'], [wins, losses]);
}

function updatePerformanceChart(canvasId, performanceData) {
    const labels = performanceData.map(d => d.period);
    const returns = performanceData.map(d => d.return);
    
    return createBarChart(canvasId, labels, [{
        label: 'Return (%)',
        data: returns,
        backgroundColor: returns.map(r => r >= 0 ? ChartColors.primary : ChartColors.danger),
        borderRadius: 4
    }]);
}

function updateDrawdownChart(canvasId, drawdownData) {
    const labels = drawdownData.map(d => d.date);
    const drawdowns = drawdownData.map(d => d.drawdown);
    
    return createAreaChart(canvasId, labels, drawdowns);
}

function destroyChart(canvasId) {
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
        delete chartInstances[canvasId];
    }
}

function destroyAllCharts() {
    Object.keys(chartInstances).forEach(id => destroyChart(id));
}

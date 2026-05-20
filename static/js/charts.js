/**
 * Plotly chart helpers for Forecast Engine.
 */

const GDMS_BLUE = '#1B3A5C';
const ACCENT_GREEN = '#548235';
const ACCENT_RED = '#C00000';
const ACCENT_AMBER = '#BF8F00';
const LIGHT_GRAY = '#F2F2F2';

const CHART_LAYOUT_DEFAULTS = {
    font: { family: 'system-ui, -apple-system, sans-serif', size: 12 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    margin: { t: 40, r: 20, b: 40, l: 60 },
};

const CHART_CONFIG = {
    responsive: true,
    displayModeBar: false,
};

/**
 * Render a SHAP waterfall chart.
 */
function renderSHAPWaterfall(elementId, shapData) {
    if (!shapData || !shapData.features) return;

    const features = shapData.features.slice(0, 10);
    const labels = features.map(f => f.name);
    const values = features.map(f => f.shap);
    const colors = values.map(v => v >= 0 ? ACCENT_GREEN : ACCENT_RED);

    const trace = {
        type: 'waterfall',
        orientation: 'h',
        y: labels.reverse(),
        x: values.reverse(),
        connector: { line: { color: '#E0E0E0' } },
        increasing: { marker: { color: ACCENT_GREEN } },
        decreasing: { marker: { color: ACCENT_RED } },
        totals: { marker: { color: GDMS_BLUE } },
    };

    const layout = {
        ...CHART_LAYOUT_DEFAULTS,
        title: { text: 'SHAP Feature Contributions', font: { size: 14 } },
        xaxis: { title: 'Impact (hours)' },
        height: 350,
    };

    Plotly.newPlot(elementId, [trace], layout, CHART_CONFIG);
}

/**
 * Render a forecast trend line chart.
 */
function renderTrendChart(elementId, periods, actuals, gbmPreds, pmFinals) {
    const traces = [
        {
            x: periods, y: actuals, name: 'Actual',
            line: { color: GDMS_BLUE, width: 2.5 },
            mode: 'lines+markers',
        },
        {
            x: periods, y: gbmPreds, name: 'GBM Forecast',
            line: { color: ACCENT_GREEN, width: 2, dash: 'dash' },
            mode: 'lines+markers',
        },
        {
            x: periods, y: pmFinals, name: 'PM Final',
            line: { color: ACCENT_AMBER, width: 2 },
            mode: 'lines+markers',
        },
    ];

    const layout = {
        ...CHART_LAYOUT_DEFAULTS,
        title: { text: 'Support Hours: Forecast vs Actual', font: { size: 14 } },
        yaxis: { title: 'Hours' },
        legend: { orientation: 'h', y: -0.15 },
        height: 300,
    };

    Plotly.newPlot(elementId, traces, layout, CHART_CONFIG);
}

/**
 * Render a variance bar chart.
 */
function renderVarianceChart(elementId, programs, modelVar, pmVar) {
    const traces = [
        {
            x: programs, y: modelVar, name: 'Model Variance',
            type: 'bar', marker: { color: GDMS_BLUE },
        },
        {
            x: programs, y: pmVar, name: 'PM Variance',
            type: 'bar', marker: { color: ACCENT_GREEN },
        },
    ];

    const layout = {
        ...CHART_LAYOUT_DEFAULTS,
        barmode: 'group',
        yaxis: { title: 'Variance %', tickformat: '.0%' },
        legend: { orientation: 'h', y: -0.2 },
        height: 300,
    };

    Plotly.newPlot(elementId, traces, layout, CHART_CONFIG);
}

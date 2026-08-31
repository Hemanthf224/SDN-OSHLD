// Global Chart Instances
let qosChart;
let attackChart;
let energyChart;

// Data Arrays for Charts
const maxDataPoints = 30;
const timeLabels = [];
const latencyData = [];
const throughputData = [];

// Attack Distribution Tracker
const attackCounts = {
    'Normal': 0,
    'DoS': 0,
    'DDoS': 0,
    'PortScan': 0,
    'Botnet': 0,
    'BruteForce': 0
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initCharts();
    
    // Start generating traffic
    setInterval(simulateTraffic, 1000);
});

// Clock
function initClock() {
    const clockEl = document.getElementById("clock");
    setInterval(() => {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString();
    }, 1000);
}

// Chart.js Default styling for Dark Theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Inter, sans-serif';

function initCharts() {
    // 1. QoS Chart (Line)
    const qosCtx = document.getElementById('qosChart').getContext('2d');
    qosChart = new Chart(qosCtx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: 'Latency (ms)',
                    data: latencyData,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Throughput (Mbps)',
                    data: throughputData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { 
                    type: 'linear', display: true, position: 'left',
                    title: { display: true, text: 'Latency (ms)' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y1: { 
                    type: 'linear', display: true, position: 'right',
                    title: { display: true, text: 'Throughput (Mbps)' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });

    // 2. Attack Distribution Chart (Bar)
    const attackCtx = document.getElementById('attackChart').getContext('2d');
    attackChart = new Chart(attackCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(attackCounts),
            datasets: [{
                label: 'Detections',
                data: Object.values(attackCounts),
                backgroundColor: [
                    '#10b981', // Normal
                    '#f59e0b', // DoS
                    '#ef4444', // DDoS
                    '#8b5cf6', // PortScan
                    '#ec4899', // Botnet
                    '#3b82f6'  // BruteForce
                ],
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 3. Energy Donut Chart
    const energyCtx = document.getElementById('energyChart').getContext('2d');
    energyChart = new Chart(energyCtx, {
        type: 'doughnut',
        data: {
            labels: ['Used', 'Saved'],
            datasets: [{
                data: [100, 0],
                backgroundColor: ['#3b82f6', 'rgba(255, 255, 255, 0.1)'],
                borderWidth: 0,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });
}

// Simulated Data Generator to ping the API
function generateRandomFlow() {
    // 80% chance normal, 20% attack
    const isNormal = Math.random() < 0.8;
    let packet_count, bytes, duration, bandwidth;

    if (isNormal) {
        packet_count = 50 + Math.random() * 20;
        bytes = packet_count * 500;
        duration = 1.5 + Math.random() * 0.5;
    } else {
        // Attack
        packet_count = 5000 + Math.random() * 1000;
        bytes = packet_count * 64;
        duration = 0.2 + Math.random() * 0.1;
    }
    bandwidth = (bytes * 8) / duration / 1e6;

    return { packet_count, bytes, duration, bandwidth };
}

async function simulateTraffic() {
    const flow = generateRandomFlow();
    const payload = { features: flow };

    try {
        const start = performance.now();
        const response = await fetch('http://127.0.0.1:8000/api/v1/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const end = performance.now();
        const latency = (end - start).toFixed(1);

        if (response.ok) {
            const data = await response.json();
            updateDashboard(data, latency);
        }
    } catch (error) {
        console.error("API Error:", error);
        document.getElementById('sys-status-text').textContent = "API Offline";
        document.getElementById('sys-status-dot').classList.remove('active');
        document.getElementById('sys-status-dot').style.backgroundColor = '#ef4444';
    }
}

function updateDashboard(data, latency) {
    // Ensure system is online
    document.getElementById('sys-status-text').textContent = "System Active";
    document.getElementById('sys-status-dot').classList.add('active');
    document.getElementById('sys-status-dot').style.backgroundColor = '';

    // Update Threat Box
    const threatLevelBadge = document.getElementById('threat-level');
    const className = data.pred_class_name;
    const isAttack = className !== 'Normal';
    
    document.getElementById('last-attack').textContent = className;
    document.getElementById('confidence').textContent = (data.pred_conf * 100).toFixed(1) + '%';
    document.getElementById('action-taken').textContent = data.action;
    document.getElementById('inference-latency').textContent = latency + ' ms';

    if (isAttack) {
        threatLevelBadge.textContent = 'Under Attack';
        threatLevelBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        threatLevelBadge.style.color = '#ef4444';
        threatLevelBadge.style.borderColor = '#ef4444';
        document.getElementById('confidence').style.color = '#ef4444';
    } else {
        threatLevelBadge.textContent = 'Secure';
        threatLevelBadge.style.background = 'rgba(16, 185, 129, 0.2)';
        threatLevelBadge.style.color = '#10b981';
        threatLevelBadge.style.borderColor = '#10b981';
        document.getElementById('confidence').style.color = '#10b981';
    }

    // Update QoS Chart
    const timeNow = new Date().toLocaleTimeString().split(' ')[0];
    timeLabels.push(timeNow);
    latencyData.push(data.qos_predictions.latency);
    throughputData.push(data.qos_predictions.throughput);

    if (timeLabels.length > maxDataPoints) {
        timeLabels.shift();
        latencyData.shift();
        throughputData.shift();
    }
    qosChart.update();

    // Update Attack Chart
    if (attackCounts[className] !== undefined) {
        attackCounts[className]++;
        attackChart.data.datasets[0].data = Object.values(attackCounts);
        attackChart.update();
    }

    // Update Energy
    const savedPct = data.energy_saved_pct.toFixed(1);
    document.getElementById('power-saved').textContent = savedPct;
    document.getElementById('current-power').textContent = data.energy_power_w.toFixed(0) + 'W';
    
    energyChart.data.datasets[0].data = [100 - savedPct, savedPct];
    energyChart.update();
}

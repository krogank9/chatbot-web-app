// Utility Functions
function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 GB';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// System Stats Management
class SystemStatsManager {
    constructor() {
        this.updateInterval = null;
    }

    updateSystemStats() {
        fetch('/system/stats')
            .then(response => response.json())
            .then(stats => {
                this.updateCPUStats(stats.cpu);
                this.updateMemoryStats(stats.memory);
                this.updateGPUStats(stats.gpu);
            });
    }

    updateCPUStats(cpu) {
        const cpuUsage = cpu.usage_percent.toFixed(1);
        document.getElementById('cpuUsage').textContent = `${cpuUsage}%`;
        document.getElementById('cpuBar').style.width = `${cpuUsage}%`;
        document.getElementById('cpuBar').className = `mini-progress-bar ${this.getLoadClass(cpuUsage)}`;
        document.getElementById('cpuFreq').textContent = `${cpu.frequency} GHz`;
        document.getElementById('cpuCores').textContent = cpu.cores;
    }

    updateMemoryStats(memory) {
        const memPercent = memory.percent;
        const memUsed = formatBytes(memory.used);
        const memTotal = formatBytes(memory.total);
        
        document.getElementById('memoryUsage').textContent = `${memPercent}%`;
        document.getElementById('memoryBar').style.width = `${memPercent}%`;
        document.getElementById('memoryBar').className = `mini-progress-bar ${this.getLoadClass(memPercent)}`;
        document.getElementById('memoryUsed').textContent = memUsed;
        document.getElementById('memoryTotal').textContent = memTotal;
    }

    updateGPUStats(gpus) {
        const gpuContainer = document.getElementById('gpuStatsContainer');
        gpuContainer.innerHTML = '';

        gpus.forEach((gpu, index) => {
            const memUsed = (gpu.memory_used).toFixed(1);
            const memTotal = (gpu.memory_total).toFixed(1);
            const memPercent = (gpu.memory_used / gpu.memory_total * 100).toFixed(1);
            const gpuLoad = gpu.gpu_load.toFixed(1);
            
            gpuContainer.innerHTML += this.createGPUStatHTML(gpu, index, memUsed, memTotal, memPercent, gpuLoad);
        });
    }

    createGPUStatHTML(gpu, index, memUsed, memTotal, memPercent, gpuLoad) {
        return `
            <div class="stat-item">
                <span class="stat-label">GPU${gpu.length > 1 ? index + 1 : ''}:</span>
                <div class="mini-progress">
                    <div class="mini-progress-bar ${this.getLoadClass(memPercent)}" 
                         style="width: ${memPercent}%"></div>
                </div>
                <span class="stat-value">${memPercent}%</span>
                <span class="stat-value">|</span>
                <span class="stat-value">${memUsed}MB</span>
                <span class="stat-value">/</span>
                <span class="stat-value">${memTotal}MB</span>
                <span class="stat-value">|</span>
                <span class="stat-value">${gpu.temperature}°C</span>
                <span class="stat-value">|</span>
                <span class="stat-value">${gpuLoad}% Load</span>
            </div>
        `;
    }

    getLoadClass(percentage) {
        return percentage > 80 ? 'bg-danger' : percentage > 60 ? 'bg-warning' : '';
    }

    startMonitoring() {
        this.updateSystemStats();
        this.updateInterval = setInterval(() => this.updateSystemStats(), 2000);
    }

    stopMonitoring() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Model Management
class ModelManager {
    constructor() {
        this.loadedModels = {};
        this.statusSource = null;
    }

    async updateLoadedModels() {
        const response = await fetch('/models/loaded');
        this.loadedModels = await response.json();
        this.updateModelsStatus();
    }

    async loadModel(modelId) {
        const button = document.getElementById(`load-${modelId}`);
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Loading...';

        try {
            const response = await fetch('/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId }),
            });
            const data = await response.json();

            if (data.success) {
                this.loadedModels = data.loaded_models;
                this.updateModelsStatus();
            } else {
                alert(`Failed to load model: ${data.error}`);
            }
        } finally {
            button.disabled = false;
            button.innerHTML = 'Load';
        }
    }

    updateModelsStatus() {
        const updateUI = (models) => {
            ['llm', 'tts', 'stt'].forEach(type => {
                const container = document.getElementById(`${type}Container`);
                if (container) container.innerHTML = '';
            });

            for (const [modelId, info] of Object.entries(models)) {
                const container = document.getElementById(`${info.type}Container`);
                if (!container) continue;

                const isLoaded = this.loadedModels[info.type] === modelId;
                container.appendChild(this.createModelCard(modelId, info, isLoaded));
            }
        };

        // Initial load
        fetch('/models/status')
            .then(response => response.json())
            .then(updateUI);

        // Setup SSE
        if (this.statusSource) {
            this.statusSource.close();
        }
        this.statusSource = new EventSource('/models/status/stream');
        this.statusSource.onmessage = (event) => {
            const models = JSON.parse(event.data);
            updateUI(models);
        };
    }

    createModelCard(modelId, info, isLoaded) {
        const div = document.createElement('div');
        div.className = 'col-12 col-md-6 col-lg-4';
        div.innerHTML = `
            <div class="card model-card h-100 ${isLoaded ? 'active' : ''}">
                <div class="card-body">
                    <h5 class="card-title">${info.name}</h5>
                    <p class="card-text">
                        ${info.description}<br>
                        Size: ${info.size}
                    </p>
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <span class="badge ${info.downloading ? 'bg-warning' : info.installed ? 'bg-success' : 'bg-secondary'}">
                                ${info.downloading ? 'Downloading...' : info.installed ? 'Installed' : 'Not Installed'}
                            </span>
                            ${isLoaded ? '<span class="badge loaded-badge ms-2">In Use</span>' : ''}
                        </div>
                        <div>
                            ${this.getActionButton(modelId, info, isLoaded)}
                        </div>
                    </div>
                </div>
            </div>
        `;
        return div;
    }

    getActionButton(modelId, info, isLoaded) {
        if (!info.installed && !info.downloading) {
            return `
                <button class="btn btn-primary btn-sm" 
                        onclick="modelManager.downloadModel('${modelId}')"
                        id="download-${modelId}">
                    Download
                </button>
            `;
        } else if (!isLoaded && info.installed) {
            return `
                <button class="btn btn-primary btn-sm"
                        onclick="modelManager.loadModel('${modelId}')"
                        id="load-${modelId}">
                    Load
                </button>
            `;
        }
        return '';
    }

    async downloadModel(modelId) {
        const button = document.getElementById(`download-${modelId}`);
        button.disabled = true;

        try {
            const response = await fetch('/models/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId }),
            });
            const data = await response.json();
            
            if (!data.success) {
                alert(`Failed to download model: ${data.error}`);
            }
        } finally {
            button.disabled = false;
        }
    }

    cleanup() {
        if (this.statusSource) {
            this.statusSource.close();
        }
    }
}

// Initialize
const systemStats = new SystemStatsManager();
const modelManager = new ModelManager();

document.addEventListener('DOMContentLoaded', () => {
    modelManager.updateLoadedModels();
    systemStats.startMonitoring();
});

window.addEventListener('beforeunload', () => {
    systemStats.stopMonitoring();
    modelManager.cleanup();
}); 
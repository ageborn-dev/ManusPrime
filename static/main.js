let currentEventSource = null;
let modelUsage = {};
let resourceStats = {};

function createTask() {
    const promptInput = document.getElementById('prompt-input');
    const prompt = promptInput.value.trim();

    if (!prompt) {
        showToast("Please enter a valid prompt", "error");
        promptInput.focus();
        return;
    }

    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }

    // Reset model usage tracking
    modelUsage = {};
    resourceStats = {};

    const container = document.getElementById('task-container');
    container.innerHTML = '<div class="loading">Initializing task...</div>';
    document.getElementById('input-container').classList.add('bottom');

    fetch('/tasks', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.detail || 'Request failed') });
        }
        return response.json();
    })
    .then(data => {
        if (!data.task_id) {
            throw new Error('Invalid task ID');
        }
        setupSSE(data.task_id);
        loadHistory();
        promptInput.value = '';
    })
    .catch(error => {
        container.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        console.error('Failed to create task:', error);
        showToast("Task creation failed", "error");
    });
}

function setupSSE(taskId) {
    let retryCount = 0;
    const maxRetries = 3;
    const retryDelay = 2000;

    function connect() {
        const eventSource = new EventSource(`/tasks/${taskId}/events`);
        currentEventSource = eventSource;

        const container = document.getElementById('task-container');

        let heartbeatTimer = setInterval(() => {
            const pingDiv = document.createElement('div');
            pingDiv.className = 'ping';
            pingDiv.innerHTML = '·';
            container.appendChild(pingDiv);
            
            // Auto-scroll
            container.scrollTo({
                top: container.scrollHeight,
                behavior: 'smooth'
            });
        }, 5000);

        const pollInterval = setInterval(() => {
            fetch(`/tasks/${taskId}`)
                .then(response => response.json())
                .then(task => {
                    updateTaskStatus(task);
                })
                .catch(error => {
                    console.error('Polling failed:', error);
                });
        }, 10000);

        let isTaskComplete = false;
        let lastResultContent = '';
        let stepContainer = null;

        function ensureStepContainer() {
            container.querySelector('.loading')?.remove();
            const welcomeMessage = document.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.style.display = 'none';
            }

            stepContainer = container.querySelector('.step-container');
            if (!stepContainer) {
                container.innerHTML = '<div class="step-container"></div>';
                stepContainer = container.querySelector('.step-container');
            }
            return stepContainer;
        }

        function addEventToContainer(eventType, content) {
            const stepContainer = ensureStepContainer();
            
            const timestamp = new Date().toLocaleTimeString();
            
            const step = document.createElement('div');
            step.className = `step-item ${eventType}`;
            
            // Check if this is a model selection event
            if (eventType === 'model' && content.includes('using')) {
                const modelMatch = content.match(/using\s+(\S+)/i);
                if (modelMatch && modelMatch[1]) {
                    const model = modelMatch[1];
                    modelUsage[model] = (modelUsage[model] || 0) + 1;
                }
            }
            
            // Check if this is a resource usage event
            if (eventType === 'resource' && content.includes('Request cost:')) {
                const costMatch = content.match(/Request cost: \$([\d.]+)/);
                const tokensMatch = content.match(/Tokens used: (\d+)/);
                
                if (costMatch && costMatch[1]) {
                    resourceStats.cost = parseFloat(costMatch[1]);
                }
                
                if (tokensMatch && tokensMatch[1]) {
                    resourceStats.tokens = parseInt(tokensMatch[1]);
                }
            }
            
            // Format the content based on type
            let formattedContent = content;
            
            step.innerHTML = `
                <div class="log-line ${eventType}">
                    <span class="log-prefix">${getEventIcon(eventType)} [${timestamp}] ${getEventLabel(eventType)}</span>
                    <pre>${formattedContent}</pre>
                </div>
            `;
            
            stepContainer.appendChild(step);
            
            // Auto-scroll
            container.scrollTo({
                top: container.scrollHeight,
                behavior: 'smooth'
            });
        }

        // Set up event listeners for all event types
        const eventTypes = [
            'status', 'think', 'tool', 'act', 'log', 'run', 
            'model', 'info', 'resource', 'budget', 'error', 'complete'
        ];
        
        eventTypes.forEach(eventType => {
            eventSource.addEventListener(eventType, (event) => {
                clearInterval(heartbeatTimer);
                try {
                    const data = JSON.parse(event.data);
                    
                    if (eventType === 'status') {
                        // Handle status updates
                        container.querySelector('.loading')?.remove();
                        container.classList.add('active');
                        
                        // Save result content
                        if (data.steps && data.steps.length > 0) {
                            // Iterate through all steps, find the last result type
                            for (let i = data.steps.length - 1; i >= 0; i--) {
                                if (data.steps[i].type === 'result') {
                                    lastResultContent = data.steps[i].result;
                                    break;
                                }
                            }
                        }
                        
                        // Parse and display each step
                        const stepContainer = ensureStepContainer();
                        stepContainer.innerHTML = data.steps.map(step => {
                            const content = step.result;
                            const timestamp = new Date().toLocaleTimeString();
                            return `
                                <div class="step-item ${step.type || 'step'}">
                                    <div class="log-line ${step.type || 'info'}">
                                        <span class="log-prefix">${getEventIcon(step.type)} [${timestamp}] ${getEventLabel(step.type)}</span>
                                        <pre>${content}</pre>
                                    </div>
                                </div>
                            `;
                        }).join('');
                    } else if (eventType === 'complete') {
                        isTaskComplete = true;
                        clearInterval(heartbeatTimer);
                        clearInterval(pollInterval);
                        
                        const stepContainer = ensureStepContainer();
                        
                        // Create a completion message
                        const complete = document.createElement('div');
                        complete.className = 'complete';
                        
                        const modelUsageHtml = Object.entries(modelUsage).map(([model, count]) => 
                            `<span class="model-badge">${model}: ${count}</span>`
                        ).join('');
                        
                        let resourceHtml = '';
                        if (resourceStats.cost || resourceStats.tokens) {
                            resourceHtml = `
                                <div class="resource-usage">
                                    <h3>Resource Usage</h3>
                                    <div class="stats-row">
                                        ${resourceStats.cost ? `<span class="cost-badge">$${resourceStats.cost.toFixed(4)}</span>` : ''}
                                        ${resourceStats.tokens ? `<span class="cost-badge">${resourceStats.tokens} tokens</span>` : ''}
                                    </div>
                                    <div class="model-usage">
                                        ${modelUsageHtml}
                                    </div>
                                </div>
                            `;
                        }
                        
                        complete.innerHTML = `
                            <div>✅ Task completed</div>
                            <pre>${lastResultContent}</pre>
                            ${resourceHtml}
                        `;
                        
                        stepContainer.appendChild(complete);
                        
                        // Auto-scroll
                        container.scrollTo({
                            top: container.scrollHeight,
                            behavior: 'smooth'
                        });
                        
                        eventSource.close();
                        currentEventSource = null;
                        showToast("Task completed successfully", "success");
                    } else if (eventType === 'error') {
                        clearInterval(heartbeatTimer);
                        clearInterval(pollInterval);
                        
                        const stepContainer = ensureStepContainer();
                        
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'error';
                        errorDiv.innerHTML = `❌ Error: ${data.message || 'Unknown error'}`;
                        
                        stepContainer.appendChild(errorDiv);
                        
                        eventSource.close();
                        currentEventSource = null;
                        showToast("Task failed", "error");
                    } else {
                        // Handle all other event types
                        addEventToContainer(eventType, data.result);
                    }
                    
                } catch (e) {
                    console.error(`Error handling ${eventType} event:`, e);
                }
            });
        });

        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });

        eventSource.onerror = (err) => {
            if (isTaskComplete) {
                return;
            }

            console.error('SSE connection error:', err);
            clearInterval(heartbeatTimer);
            clearInterval(pollInterval);
            eventSource.close();

            if (retryCount < maxRetries) {
                retryCount++;
                const warning = document.createElement('div');
                warning.className = 'warning';
                warning.innerHTML = `⚠ Connection lost, retrying in ${retryDelay/1000} seconds (${retryCount}/${maxRetries})...`;
                
                ensureStepContainer().appendChild(warning);
                
                setTimeout(connect, retryDelay);
            } else {
                const error = document.createElement('div');
                error.className = 'error';
                error.innerHTML = `⚠ Connection lost, please try refreshing the page`;
                
                ensureStepContainer().appendChild(error);
                showToast("Connection to server lost", "error");
            }
        };
    }

    connect();
}

function getEventIcon(eventType) {
    switch(eventType) {
        case 'think': return '🤔';
        case 'tool': return '🛠️';
        case 'act': return '🚀';
        case 'result': return '🏁';
        case 'error': return '❌';
        case 'complete': return '✅';
        case 'log': return '📝';
        case 'run': return '⚙️';
        case 'model': return '🧠';
        case 'resource': return '💰';
        case 'budget': return '💸';
        case 'info': return 'ℹ️';
        default: return 'ℹ️';
    }
}

function getEventLabel(eventType) {
    switch(eventType) {
        case 'think': return 'Thinking';
        case 'tool': return 'Using Tool';
        case 'act': return 'Action';
        case 'result': return 'Result';
        case 'error': return 'Error';
        case 'complete': return 'Complete';
        case 'log': return 'Log';
        case 'run': return 'Running';
        case 'model': return 'Model Selection';
        case 'resource': return 'Resource Usage';
        case 'budget': return 'Budget Alert';
        case 'info': return 'Info';
        default: return 'Info';
    }
}

function updateTaskStatus(task) {
    const statusBar = document.getElementById('status-bar');
    if (!statusBar) return;

    if (task.status === 'completed') {
        statusBar.innerHTML = `<span class="status-complete">✅ Task completed</span>`;
    } else if (task.status === 'failed') {
        statusBar.innerHTML = `<span class="status-error">❌ Task failed: ${task.error || 'Unknown error'}</span>`;
    } else {
        statusBar.innerHTML = `<span class="status-running">⚙️ Task running: ${task.status}</span>`;
    }
}

function loadHistory() {
    fetch('/tasks')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load history');
            }
            return response.json();
        })
        .then(tasks => {
            const taskList = document.getElementById('task-list');
            if (!taskList) return;

            taskList.innerHTML = '';

            if (tasks.length === 0) {
                taskList.innerHTML = '<div class="history-empty">No recent tasks</div>';
                return;
            }

            tasks.forEach(task => {
                // Extract the first few characters of the prompt
                const shortPrompt = task.prompt.length > 50 
                    ? task.prompt.substring(0, 50) + '...' 
                    : task.prompt;
                
                // Determine status for styling
                let statusClass = 'pending';
                if (task.status === 'completed') statusClass = 'completed';
                else if (task.status === 'running') statusClass = 'running';
                else if (task.status.includes('failed')) statusClass = 'failed';
                
                const taskItem = document.createElement('div');
                taskItem.className = `task-item ${statusClass}`;
                taskItem.innerHTML = `
                    <div class="task-prompt">${shortPrompt}</div>
                    <div class="task-meta">
                        <span class="task-time">${formatTime(new Date(task.created_at))}</span>
                        <span class="task-status">${getStatusIcon(task.status)}</span>
                    </div>
                `;
                taskItem.addEventListener('click', () => {
                    loadTask(task.id);
                });
                taskList.appendChild(taskItem);
            });
        })
        .catch(error => {
            console.error('Failed to load history:', error);
            showToast("Failed to load task history", "error");
        });
}

function formatTime(date) {
    // Get today's date
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    // Check if the date is today
    if (date >= today) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // If it's within the last week, show day name
    const lastWeek = new Date(today);
    lastWeek.setDate(lastWeek.getDate() - 7);
    
    if (date >= lastWeek) {
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        return days[date.getDay()];
    }
    
    // Otherwise show date
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function getStatusIcon(status) {
    if (status === 'completed') return '✅';
    if (status.includes('failed')) return '❌';
    if (status === 'running') return '⚙️';
    return '⏳';
}

function loadTask(taskId) {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }

    // Reset model usage tracking
    modelUsage = {};
    resourceStats = {};

    const container = document.getElementById('task-container');
    container.innerHTML = '<div class="loading">Loading task...</div>';
    document.getElementById('input-container').classList.add('bottom');

    fetch(`/tasks/${taskId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load task');
            }
            return response.json();
        })
        .then(task => {
            if (task.status === 'running') {
                setupSSE(taskId);
            } else {
                displayTask(task);
            }
        })
        .catch(error => {
            console.error('Failed to load task:', error);
            container.innerHTML = `<div class="error">Failed to load task: ${error.message}</div>`;
            showToast("Failed to load task", "error");
        });
}

function displayTask(task) {
    const container = document.getElementById('task-container');
    container.innerHTML = '';
    container.classList.add('active');

    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    const stepContainer = document.createElement('div');
    stepContainer.className = 'step-container';

    // Extract model usage information
    let extractedModels = new Set();
    let resourceCost = null;
    let resourceTokens = null;

    if (task.steps && task.steps.length > 0) {
        task.steps.forEach(step => {
            // Create step display
            const stepItem = document.createElement('div');
            stepItem.className = `step-item ${step.type || 'step'}`;

            const content = step.result;
            const timestamp = new Date(task.created_at).toLocaleTimeString();

            stepItem.innerHTML = `
                <div class="log-line ${step.type || 'info'}">
                    <span class="log-prefix">${getEventIcon(step.type)} [${timestamp}] ${getEventLabel(step.type)}</span>
                    <pre>${content}</pre>
                </div>
            `;

            stepContainer.appendChild(stepItem);

            // Extract model usage info
            if (step.type === 'model' && content.includes('using')) {
                const modelMatch = content.match(/using\s+(\S+)/i);
                if (modelMatch && modelMatch[1]) {
                    extractedModels.add(modelMatch[1]);
                }
            }
            
            // Extract resource usage info
            if (step.type === 'resource' || step.type === 'info') {
                const costMatch = content.match(/Request cost: \$([\d.]+)/);
                const tokensMatch = content.match(/Tokens used: (\d+)/);
                
                if (costMatch && costMatch[1]) {
                    resourceCost = parseFloat(costMatch[1]);
                }
                
                if (tokensMatch && tokensMatch[1]) {
                    resourceTokens = parseInt(tokensMatch[1]);
                }
            }
        });
    } else {
        stepContainer.innerHTML = '<div class="no-steps">No steps recorded for this task</div>';
    }

    container.appendChild(stepContainer);

    // Find the last result content
    let lastResultContent = '';
    if (task.steps && task.steps.length > 0) {
        for (let i = task.steps.length - 1; i >= 0; i--) {
            if (task.steps[i].type === 'result') {
                lastResultContent = task.steps[i].result;
                break;
            }
        }
    }

    // Add completion or error message
    if (task.status === 'completed') {
        const complete = document.createElement('div');
        complete.className = 'complete';
        
        // Create model usage HTML if models were used
        let modelUsageHtml = '';
        if (extractedModels.size > 0) {
            modelUsageHtml = Array.from(extractedModels).map(model => 
                `<span class="model-badge">${model}</span>`
            ).join('');
        }
        
        // Create resource usage HTML if cost or tokens were recorded
        let resourceHtml = '';
        if (resourceCost || resourceTokens) {
            resourceHtml = `
                <div class="resource-usage">
                    <h3>Resource Usage</h3>
                    <div class="stats-row">
                        ${resourceCost ? `<span class="cost-badge">$${resourceCost.toFixed(4)}</span>` : ''}
                        ${resourceTokens ? `<span class="cost-badge">${resourceTokens} tokens</span>` : ''}
                    </div>
                    <div class="model-usage">
                        ${modelUsageHtml}
                    </div>
                </div>
            `;
        }
        
        complete.innerHTML = `
            <div>✅ Task completed</div>
            <pre>${lastResultContent}</pre>
            ${resourceHtml}
        `;
        
        container.appendChild(complete);
    } else if (task.status.includes('failed')) {
        const error = document.createElement('div');
        error.className = 'error';
        error.innerHTML = `❌ Error: ${task.status.replace('failed: ', '')}`;
        container.appendChild(error);
    }
}

function showToast(message, type = "info") {
    // Check if toast container exists, create if not
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.position = 'fixed';
        toastContainer.style.bottom = '20px';
        toastContainer.style.right = '20px';
        toastContainer.style.zIndex = '1000';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.style.backgroundColor = type === 'error' ? '#e5383b' : 
                                 type === 'success' ? '#38b000' : '#4cc9f0';
    toast.style.color = 'white';
    toast.style.padding = '12px 20px';
    toast.style.borderRadius = '8px';
    toast.style.marginTop = '10px';
    toast.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(20px)';
    toast.style.transition = 'all 0.3s ease';
    toast.style.cursor = 'pointer';
    toast.style.fontSize = '14px';
    toast.style.fontWeight = '500';
    toast.style.display = 'flex';
    toast.style.alignItems = 'center';
    toast.style.gap = '8px';
    
    // Add icon based on type
    const icon = type === 'error' ? '❌' : 
                type === 'success' ? '✅' : 'ℹ️';
    
    toast.innerHTML = `${icon} ${message}`;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    }, 10);
    
    // Auto dismiss
    const dismissTime = 5000;
    const removeTimeout = setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 300);
    }, dismissTime);
    
    // Click to dismiss early
    toast.addEventListener('click', () => {
        clearTimeout(removeTimeout);
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            if (toastContainer.contains(toast)) {
                toastContainer.removeChild(toast);
            }
        }, 300);
    });
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();

    // Set up event listeners for prompt input
    const promptInput = document.getElementById('prompt-input');
    if (promptInput) {
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                createTask();
            }
        });
    }
});

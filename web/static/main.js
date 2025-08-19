// Web interface for ManusPrime - UPDATED with better error handling and loading states
let currentEventSource = null;
let activeTaskId = null;
let isSubmitting = false;

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    const promptInput = document.getElementById('prompt-input');
    const submitButton = document.getElementById('submit-button');
    const newTaskButton = document.getElementById('new-task-button');
    
    // Submit on enter (but allow shift+enter for newlines)
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isSubmitting) {
            e.preventDefault();
            createTask();
        }
    });
    
    // Submit on button click
    submitButton.addEventListener('click', () => {
        if (!isSubmitting) {
            createTask();
        }
    });

    // New task button handler
    newTaskButton.addEventListener('click', startNewTask);
    
    // Load task history
    loadTaskHistory();
    
    // Check system health on startup
    checkSystemHealth();
});

/**
 * Check system health and display warnings if needed
 */
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        if (!response.ok) {
            showNotification('System health check failed', 'warning');
        }
    } catch (error) {
        showNotification('Unable to connect to server', 'error');
    }
}

/**
 * Set loading state for submit button
 */
function setSubmitButtonLoading(loading) {
    const submitButton = document.getElementById('submit-button');
    const promptInput = document.getElementById('prompt-input');
    
    isSubmitting = loading;
    
    if (loading) {
        submitButton.textContent = 'Sending...';
        submitButton.disabled = true;
        submitButton.classList.add('loading');
        promptInput.disabled = true;
    } else {
        submitButton.textContent = 'Send';
        submitButton.disabled = false;
        submitButton.classList.remove('loading');
        promptInput.disabled = false;
    }
}

/**
 * Handle starting a new task
 */
async function startNewTask() {
    if (isSubmitting) {
        showNotification('Please wait for current task to complete', 'warning');
        return;
    }

    // Check for active task
    if (activeTaskId) {
        if (!confirm('Starting a new task will close the current task. Continue?')) {
            return;
        }
        
        showNotification('Cleaning up current task...', 'info');
        
        try {
            // Save current task state
            await fetch(`/api/tasks/${activeTaskId}/save`, {
                method: 'POST'
            });
            
            // Cleanup if there's an active sandbox session
            await fetch(`/api/tasks/${activeTaskId}/cleanup`, {
                method: 'POST'
            });
            
            showNotification('Task cleanup completed', 'success');
        } catch (error) {
            console.error('Error cleaning up task:', error);
            showNotification('Error cleaning up task: ' + error.message, 'error');
        }
    }
    
    // Reset UI state
    document.getElementById('task-container').innerHTML = `
        <div class="welcome">
            <h2>Welcome to ManusPrime</h2>
            <p>The intelligent multi-model AI agent that selects the optimal model for each task.</p>
            <p>Enter your task below to get started.</p>
        </div>
    `;
    document.getElementById('prompt-input').value = '';
    activeTaskId = null;
    
    // Refresh task history
    loadTaskHistory();
}

/**
 * Estimate cost for a task (rough estimation)
 */
function estimateTaskCost(prompt) {
    // Very rough estimation based on prompt length
    const tokenEstimate = Math.ceil(prompt.length / 4); // Rough tokens estimate
    const estimatedCost = (tokenEstimate / 1000) * 0.01; // Rough cost estimate
    return estimatedCost;
}

/**
 * Create a new task or continue existing task
 */
async function createTask() {
    if (isSubmitting) {
        return;
    }

    const promptInput = document.getElementById('prompt-input');
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        showNotification('Please enter a task or question', 'error');
        promptInput.focus();
        return;
    }
    
    // Show cost estimate
    const estimatedCost = estimateTaskCost(prompt);
    if (estimatedCost > 0.10) { // Show warning for expensive tasks
        if (!confirm(`This task may cost approximately $${estimatedCost.toFixed(4)}. Continue?`)) {
            return;
        }
    }
    
    // Set loading state
    setSubmitButtonLoading(true);
    
    // Close any existing event stream
    closeEventStream();
    
    // Show loading state in task container
    const taskContainer = document.getElementById('task-container');
    taskContainer.innerHTML = `
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">Creating task...</div>
            <div class="loading-subtext">Estimated cost: $${estimatedCost.toFixed(4)}</div>
        </div>
    `;
    
    // Send request to create task or continue existing
    const endpoint = activeTaskId ? `/api/tasks/${activeTaskId}/continue` : '/api/tasks';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Task created successfully
        promptInput.value = '';
        activeTaskId = data.task_id;
        
        // Display user message
        taskContainer.innerHTML = `
            <div class="message user-message">
                <div class="message-content">${sanitizeHTML(prompt)}</div>
            </div>
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Processing your request...</div>
                <div class="loading-subtext">AI is analyzing and planning execution</div>
            </div>
        `;
        
        // Connect to event stream
        connectToEventStream(data.task_id);
        
        // Refresh task history
        loadTaskHistory();
        
        showNotification('Task created successfully', 'success');
        
    } catch (error) {
        console.error('Error creating task:', error);
        showNotification(`Failed to create task: ${error.message}`, 'error');
        taskContainer.innerHTML = `
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <div class="error-title">Task Creation Failed</div>
                <div class="error-message">${sanitizeHTML(error.message)}</div>
                <button class="retry-button" onclick="createTask()">Try Again</button>
            </div>
        `;
    } finally {
        setSubmitButtonLoading(false);
    }
}

/**
 * Connect to the server-sent events stream for a task
 */
function connectToEventStream(taskId) {
    // Create event source
    const eventSource = new EventSource(`/api/tasks/${taskId}/events`);
    currentEventSource = eventSource;
    
    // Connection opened
    eventSource.onopen = () => {
        console.log('Event stream connected');
        updateLoadingText('Connected to server...');
    };
    
    // Handle status events
    eventSource.addEventListener('status', (event) => {
        const data = JSON.parse(event.data);
        updateTaskStatus(data);
    });
    
    // Handle result events
    eventSource.addEventListener('result', (event) => {
        const data = JSON.parse(event.data);
        displayResult(data);
    });
    
    // Handle completion events
    eventSource.addEventListener('complete', (event) => {
        const data = JSON.parse(event.data);
        markTaskComplete(data);
        closeEventStream();
        loadTaskHistory();
    });
    
    // Handle error events
    eventSource.addEventListener('error', (event) => {
        let errorMessage = 'An error occurred';
        try {
            const data = JSON.parse(event.data);
            errorMessage = data.message || errorMessage;
        } catch (e) {
            // Use default error message
        }
        
        showNotification(errorMessage, 'error');
        displayError(errorMessage);
        closeEventStream();
        loadTaskHistory();
    });
    
    // Handle resource usage events
    eventSource.addEventListener('resource', (event) => {
        const data = JSON.parse(event.data);
        displayResourceUsage(data);
    });
    
    // Handle connection errors
    eventSource.onerror = (event) => {
        console.error('Event source error:', event);
        if (eventSource.readyState === EventSource.CLOSED) {
            showNotification('Connection to server lost', 'error');
            updateLoadingText('Connection lost - please refresh the page');
        } else if (eventSource.readyState === EventSource.CONNECTING) {
            updateLoadingText('Reconnecting to server...');
        }
    };
    
    // Set timeout for connection
    setTimeout(() => {
        if (eventSource.readyState === EventSource.CONNECTING) {
            eventSource.close();
            showNotification('Connection timeout - please try again', 'error');
            displayError('Connection timeout');
        }
    }, 30000); // 30 second timeout
}

/**
 * Update loading text
 */
function updateLoadingText(text, subtext = '') {
    const loadingText = document.querySelector('.loading-text');
    const loadingSubtext = document.querySelector('.loading-subtext');
    
    if (loadingText) {
        loadingText.textContent = text;
    }
    if (loadingSubtext) {
        loadingSubtext.textContent = subtext;
    }
}

/**
 * Close the current event stream
 */
function closeEventStream() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
}

/**
 * Update the task status display
 */
function updateTaskStatus(data) {
    console.log('Task status:', data.status);
    
    switch (data.status) {
        case 'running':
            updateLoadingText('Processing...', 'AI models are working on your task');
            break;
        case 'completed':
            updateLoadingText('Completed', 'Task finished successfully');
            break;
        case 'failed':
            updateLoadingText('Failed', 'Task execution failed');
            break;
    }
}

/**
 * Display task result
 */
function displayResult(data) {
    const taskContainer = document.getElementById('task-container');
    
    // Remove loading indicator
    const loadingContainer = taskContainer.querySelector('.loading-container');
    if (loadingContainer) {
        loadingContainer.remove();
    }
    
    // Add assistant message
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.innerHTML = `<div class="message-content">${formatContent(data.content)}</div>`;
    
    // Add fade-in animation
    messageDiv.style.opacity = '0';
    taskContainer.appendChild(messageDiv);
    
    // Trigger fade-in
    setTimeout(() => {
        messageDiv.style.transition = 'opacity 0.3s ease-in';
        messageDiv.style.opacity = '1';
    }, 10);
    
    // Scroll to bottom
    taskContainer.scrollTop = taskContainer.scrollHeight;
}

/**
 * Display resource usage information
 */
function displayResourceUsage(data) {
    // Log resource data for debugging
    console.log('Resource usage data:', data);
    
    // Only display if we have resource data (checking for undefined/null)
    if (!data || data.tokens === undefined || data.cost === undefined) {
        console.warn('Missing resource usage data:', data);
        return;
    }
    
    const taskContainer = document.getElementById('task-container');
    
    // Create resource usage element
    const resourceDiv = document.createElement('div');
    resourceDiv.className = 'resource-usage';
    
    // Format models used
    let modelsHtml = '';
    if (data.models) {
        const modelEntries = Object.entries(data.models);
        if (modelEntries.length > 0) {
            modelsHtml = `
                <div class="resource-models">
                    <h4>Models Used:</h4>
                    <div class="model-tags">
                        ${modelEntries.map(([model, count]) => `
                            <span class="model-tag">${sanitizeHTML(model)} (${count})</span>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    }
    
    // Determine cost color based on amount
    const costClass = data.cost > 0.10 ? 'cost-high' : data.cost > 0.01 ? 'cost-medium' : 'cost-low';
    
    resourceDiv.innerHTML = `
        <h3>Resource Usage</h3>
        <div class="resource-details">
            <div class="resource-item">
                <span class="resource-label">Tokens:</span>
                <span class="resource-value">${data.tokens.total.toLocaleString()}</span>
            </div>
            <div class="resource-item">
                <span class="resource-label">Cost:</span>
                <span class="resource-value ${costClass}">$${data.cost.toFixed(4)}</span>
            </div>
        </div>
        ${modelsHtml}
    `;
    
    // Add fade-in animation
    resourceDiv.style.opacity = '0';
    taskContainer.appendChild(resourceDiv);
    
    setTimeout(() => {
        resourceDiv.style.transition = 'opacity 0.3s ease-in';
        resourceDiv.style.opacity = '1';
    }, 10);
    
    // Scroll to bottom
    taskContainer.scrollTop = taskContainer.scrollHeight;
}

/**
 * Delete a task
 */
async function deleteTask(taskId, event) {
    // Prevent the click from bubbling up to the task item
    event.stopPropagation();
    
    // Confirm deletion
    if (!confirm('Are you sure you want to delete this task?')) {
        return;
    }
    
    // Show loading state
    const taskItem = document.querySelector(`.task-item[data-task-id="${taskId}"]`);
    if (taskItem) {
        taskItem.classList.add('deleting');
        taskItem.style.opacity = '0.5';
    }
    
    try {
        const response = await fetch(`/api/tasks/${taskId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to delete task: ${response.status}`);
        }
        
        // Remove from task list with animation
        if (taskItem) {
            taskItem.style.transition = 'transform 0.3s ease-out, opacity 0.3s ease-out';
            taskItem.style.transform = 'translateX(-100%)';
            taskItem.style.opacity = '0';
            
            setTimeout(() => {
                taskItem.remove();
            }, 300);
        }
        
        // If this was the active task, clear the task container
        if (taskId === activeTaskId) {
            activeTaskId = null;
            const taskContainer = document.getElementById('task-container');
            taskContainer.innerHTML = `
                <div class="welcome">
                    <h2>Welcome to ManusPrime</h2>
                    <p>The intelligent multi-model AI agent that selects the optimal model for each task.</p>
                    <p>Enter your task below to get started.</p>
                </div>
            `;
        }
        
        showNotification('Task deleted successfully', 'success');
        
    } catch (error) {
        console.error('Error deleting task:', error);
        
        // Restore task item state
        if (taskItem) {
            taskItem.classList.remove('deleting');
            taskItem.style.opacity = '1';
        }
        
        showNotification(`Error deleting task: ${error.message}`, 'error');
    }
}

/**
 * Display an error message
 */
function displayError(message) {
    const taskContainer = document.getElementById('task-container');
    
    // Remove loading indicator
    const loadingContainer = taskContainer.querySelector('.loading-container');
    if (loadingContainer) {
        loadingContainer.remove();
    }
    
    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-container';
    errorDiv.innerHTML = `
        <div class="error-icon">⚠️</div>
        <div class="error-title">Task Failed</div>
        <div class="error-message">${sanitizeHTML(message)}</div>
        <button class="retry-button" onclick="location.reload()">Refresh Page</button>
    `;
    
    taskContainer.appendChild(errorDiv);
    
    // Scroll to bottom
    taskContainer.scrollTop = taskContainer.scrollHeight;
}

/**
 * Mark a task as complete
 */
function markTaskComplete(data) {
    console.log('Task completed:', data);
    updateLoadingText('Completed!', `Finished in ${data.execution_time?.toFixed(2)}s`);
    
    // Show completion notification
    showNotification('Task completed successfully!', 'success');
}

/**
 * Load task history with better error handling
 */
async function loadTaskHistory() {
    const taskList = document.getElementById('task-list');
    taskList.innerHTML = '<div class="loading-indicator"><span>Loading...</span></div>';
    
    try {
        const response = await fetch('/api/tasks');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const tasks = await response.json();
        
        if (tasks.length === 0) {
            taskList.innerHTML = '<div class="empty-list">No tasks yet</div>';
            return;
        }
        
        taskList.innerHTML = '';
        tasks.forEach(task => {
            const taskItem = document.createElement('div');
            taskItem.className = `task-item ${getStatusClass(task.status)}`;
            taskItem.dataset.taskId = task.id;
            
            const isActive = task.id === activeTaskId;
            if (isActive) {
                taskItem.classList.add('active');
            }
            
            taskItem.innerHTML = `
                <div class="task-item-header">
                    <span class="task-status status-${getStatusClass(task.status)}">
                        ${formatStatus(task.status)}
                    </span>
                    <span class="task-date" title="${new Date(task.created_at).toLocaleString()}">
                        ${formatDate(task.created_at)}
                    </span>
                </div>
                <div class="task-prompt">${sanitizeHTML(task.prompt)}</div>
                <div class="task-actions">
                    <button class="delete-task-btn" title="Delete task">🗑️</button>
                </div>
            `;
            
            taskItem.addEventListener('click', () => loadTask(task.id));
            
            // Add event listener to delete button
            const deleteBtn = taskItem.querySelector('.delete-task-btn');
            deleteBtn.addEventListener('click', (e) => deleteTask(task.id, e));
            
            taskList.appendChild(taskItem);
        });
        
    } catch (error) {
        console.error('Error loading task history:', error);
        taskList.innerHTML = `
            <div class="error-message">
                <div>Failed to load history</div>
                <small>${error.message}</small>
                <button onclick="loadTaskHistory()" class="retry-button-small">Retry</button>
            </div>
        `;
    }
}

/**
 * Load a specific task with better error handling
 */
async function loadTask(taskId) {
    // Close any existing event stream
    closeEventStream();
    
    // Set active task
    activeTaskId = taskId;
    
    // Update active task in task list
    const taskItems = document.querySelectorAll('.task-item');
    taskItems.forEach(item => {
        item.classList.toggle('active', item.dataset.taskId === taskId);
    });
    
    // Show loading state
    const taskContainer = document.getElementById('task-container');
    taskContainer.innerHTML = `
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">Loading task...</div>
        </div>
    `;
    
    try {
        const response = await fetch(`/api/tasks/${taskId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load task: HTTP ${response.status}`);
        }
        
        const task = await response.json();
        
        // Display task
        taskContainer.innerHTML = `
            <div class="message user-message">
                <div class="message-content">${sanitizeHTML(task.prompt)}</div>
            </div>
        `;
        
        // Display results if any
        if (task.results && task.results.length > 0) {
            task.results.forEach(result => {
                if (result.result_type === 'error') {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'message error-message';
                    errorDiv.innerHTML = `<div class="message-content">Error: ${sanitizeHTML(result.content)}</div>`;
                    taskContainer.appendChild(errorDiv);
                } else {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message assistant-message';
                    messageDiv.innerHTML = `<div class="message-content">${formatContent(result.content)}</div>`;
                    taskContainer.appendChild(messageDiv);
                }
            });
        }
        
        // Display resource usage if available
        console.log('Task resource usage:', task.resource_usage);
        if (task.resource_usage && task.resource_usage.total_tokens !== undefined && task.resource_usage.cost !== undefined) {
            const resourceDiv = document.createElement('div');
            resourceDiv.className = 'resource-usage';
            
            // Format models used
            let modelsHtml = '';
            if (task.resource_usage.models_used) {
                const modelEntries = Object.entries(task.resource_usage.models_used);
                if (modelEntries.length > 0) {
                    modelsHtml = `
                        <div class="resource-models">
                            <h4>Models Used:</h4>
                            <div class="model-tags">
                                ${modelEntries.map(([model, count]) => `
                                    <span class="model-tag">${sanitizeHTML(model)} (${count})</span>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }
            }
            
            const costClass = task.resource_usage.cost > 0.10 ? 'cost-high' : task.resource_usage.cost > 0.01 ? 'cost-medium' : 'cost-low';
            
            resourceDiv.innerHTML = `
                <h3>Resource Usage</h3>
                <div class="resource-details">
                    <div class="resource-item">
                        <span class="resource-label">Tokens:</span>
                        <span class="resource-value">${task.resource_usage.total_tokens.toLocaleString()}</span>
                    </div>
                    <div class="resource-item">
                        <span class="resource-label">Cost:</span>
                        <span class="resource-value ${costClass}">$${task.resource_usage.cost.toFixed(4)}</span>
                    </div>
                    <div class="resource-item">
                        <span class="resource-label">Execution Time:</span>
                        <span class="resource-value">${task.resource_usage.execution_time.toFixed(2)}s</span>
                    </div>
                </div>
                ${modelsHtml}
            `;
            
            taskContainer.appendChild(resourceDiv);
        }
        
        // If task is still running, connect to event stream
        if (task.status === 'running') {
            connectToEventStream(taskId);
        }
        
        // Scroll to top
        taskContainer.scrollTop = 0;
        
    } catch (error) {
        console.error('Error loading task:', error);
        showNotification(`Failed to load task: ${error.message}`, 'error');
        taskContainer.innerHTML = `
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <div class="error-title">Failed to Load Task</div>
                <div class="error-message">${sanitizeHTML(error.message)}</div>
                <button class="retry-button" onclick="loadTask('${taskId}')">Try Again</button>
            </div>
        `;
    }
}

/**
 * Format content with Markdown-like syntax
 */
function formatContent(content) {
    if (!content) {
        return '';
    }
    
    // Sanitize the content first
    let formatted = sanitizeHTML(content);
    
    // Convert code blocks
    formatted = formatted.replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, language, code) => {
        return `<pre class="code-block ${language}">${code}</pre>`;
    });
    
    // Convert inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert lists
    formatted = formatted.replace(/^\s*[-*]\s+(.*)/gm, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*<\/li>\n)+/g, '<ul>$&</ul>');
    
    // Convert headers
    formatted = formatted.replace(/^#{1,6}\s+(.*)/gm, (match, text) => {
        const level = match.split(' ')[0].length;
        return `<h${level}>${text}</h${level}>`;
    });
    
    // Convert paragraphs (simple version)
    formatted = formatted.replace(/\n\n([^<\n].*)/g, '<p>$1</p>');
    
    return formatted;
}

/**
 * Get CSS class for a task status
 */
function getStatusClass(status) {
    if (status === 'pending') {
        return 'pending';
    } else if (status === 'running') {
        return 'running';
    } else if (status === 'completed') {
        return 'completed';
    } else if (status.startsWith('failed')) {
        return 'error';
    }
    return 'pending';
}

/**
 * Format a task status for display
 */
function formatStatus(status) {
    if (status === 'pending') {
        return 'Pending';
    } else if (status === 'running') {
        return 'Running';
    } else if (status === 'completed') {
        return 'Completed';
    } else if (status.startsWith('failed')) {
        return 'Failed';
    }
    return status;
}

/**
 * Format a date for display
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    
    // If today, show time
    if (date.toDateString() === now.toDateString()) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // If this year, show month and day
    if (date.getFullYear() === now.getFullYear()) {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
    
    // Otherwise show full date
    return date.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
}

/**
 * Show a notification with better styling
 */
function showNotification(message, type = 'info') {
    // Check if notification container exists, create if not
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        document.body.appendChild(container);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Add icon based on type
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    notification.innerHTML = `
        <span class="notification-icon">${icons[type] || icons.info}</span>
        <span class="notification-message">${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    // Add notification to container
    container.appendChild(notification);
    
    // Auto-remove notification after delay
    setTimeout(() => {
        if (notification.parentElement) {
            notification.classList.add('fade-out');
            setTimeout(() => {
                if (container.contains(notification)) {
                    container.removeChild(notification);
                }
            }, 300);
        }
    }, 5000);
}

/**
 * Sanitize HTML to prevent XSS attacks
 */
function sanitizeHTML(text) {
    const element = document.createElement('div');
    element.textContent = text;
    return element.innerHTML;
}


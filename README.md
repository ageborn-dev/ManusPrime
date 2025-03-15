# 👋 ManusPrime

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ManusPrime is an advanced multi-model AI agent framework that orchestrates various AI models to solve complex tasks. Built on the foundation of [OpenManus](https://github.com/mannaandpoem/OpenManus), ManusPrime takes agent capabilities to the next level by intelligently leveraging the unique strengths of different AI models.

## 🚀 Key Features

- **Smart Model Selection**: Automatically selects the most appropriate AI model based on task type and complexity
  - Uses specialized coding models for development tasks
  - Leverages powerful models for complex reasoning
  - Falls back to efficient models for general tasks
- **Advanced Memory System**: Improved context management with vector storage 
- **Resource Optimization**: Sophisticated monitoring and control of resource usage
  - Token usage tracking and cost monitoring
  - Budget enforcement and optimization
  - Performance metrics and usage statistics
- **Resilient Error Handling**: Comprehensive error recovery and retry mechanisms
  - Automatic retries with exponential backoff
  - Graceful degradation with model fallbacks
  - Detailed error tracking and reporting
- **Multi-Format Output**: Support for generating content in various formats
  - Desktop/mobile/print-optimized versions
  - Format-specific optimizations and features
  - Consistent core content across formats

## 💡 Philosophy

ManusPrime operates on the principle that different AI models have unique strengths and capabilities. Rather than relying on a single model for all tasks, ManusPrime intelligently routes tasks to the most appropriate model:

- Use reasoning-focused models (like Claude) for planning and decision-making
- Leverage specialized coding models for software development
- Employ multimodal models for visual tasks
- Utilize fast, economical models for simple queries

This approach results in better performance, lower costs, and more robust capabilities.

## 🛠️ Architecture

ManusPrime consists of several key components:

1. **Model Router**: 
   - Dynamic model selection based on task analysis
   - Automatic fallback strategies
   - Cost-aware routing decisions

2. **Resource Monitor**:
   - Real-time token usage tracking
   - Cost monitoring and budget enforcement
   - Performance metrics collection
   - Usage statistics and reporting

3. **Memory System**: 
   - Vector-based long-term storage
   - Efficient context management
   - Cross-conversation knowledge retention

4. **Tool Ecosystem**: 
   - Rich set of built-in tools
   - Custom tool development framework
   - Tool usage optimization

5. **Flow Engine**: 
   - Multi-step task coordination
   - Parallel execution capabilities
   - Progress tracking and recovery

## 📋 Installation

```bash
# Clone the repository
git clone https://github.com/ageborn-dev/ManusPrime.git
cd ManusPrime

# Create a new virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your API keys
cp config/config.example.toml config/config.toml
# Edit config/config.toml with your API keys
```

## ⚙️ Configuration

```toml
# Global LLM configuration
[llm.default]
provider = "openai"
model = "gpt-4o"
api_key = "sk-..."
max_tokens = 4096
temperature = 0.0

# Coding-specific model
[llm.code]
provider = "deepseek"
model = "deepseek-chat"
api_key = "sk-..."

# Budget controls
[monitoring]
budget_limit = 10.0  # Optional daily budget limit
track_usage = true
```

## 🚀 Quick Start

```python
from manusprime.app.flow import FlowFactory, FlowType
from manusprime.app.agent import ManusPrimeAgent

# Create agent with budget control
agent = ManusPrimeAgent(budget_limit=10.0)

# Task-specific model selection happens automatically
result = await agent.run("Write a Python script")  # Uses deepseek-chat
result = await agent.run("Create a travel plan")   # Uses gpt-4o
result = await agent.run("Quick question")         # Uses gpt-4o-mini

# Multi-format output generation
travel_guide = await agent.run({
    "task": "Create Japan travel guide",
    "formats": ["detailed", "mobile", "print"]
})

# Access usage statistics
usage = resource_monitor.get_summary()
print(f"Total cost: ${usage['cost']:.2f}")
print(f"Models used: {usage['models']}")
```

## 🔍 Advanced Usage

### Error Handling

```python
# Configure retry behavior
agent.llm.configure_retries(
    max_attempts=6,
    min_wait=1,
    max_wait=60
)

# Handle specific error cases
try:
    result = await agent.run("Complex task")
except BudgetExceededError:
    # Switch to more cost-effective model
    result = await agent.run("Complex task", model="gpt-4o-mini")
```

### Resource Monitoring

```python
# Start monitoring session
resource_monitor.start_session(budget_limit=10.0)

# Add custom budget handler
def on_budget_exceeded(current_cost, limit):
    notify_admin(f"Budget ${current_cost:.2f} exceeded limit ${limit:.2f}")

resource_monitor.add_budget_listener(on_budget_exceeded)

# Get detailed metrics
metrics = resource_monitor.get_metrics()
print(f"Average response time: {metrics['avg_response_time']:.2f}s")
print(f"Token efficiency: {metrics['tokens_per_dollar']:.0f} tokens/$")
```

### Multi-Format Output

```python
# Generate content with format-specific optimizations
result = await agent.run({
    "task": "Create documentation",
    "formats": [{
        "type": "detailed",
        "include_examples": true
    }, {
        "type": "mobile",
        "optimize_images": true,
        "enable_dark_mode": true
    }, {
        "type": "print",
        "page_size": "A4",
        "include_toc": true
    }]
})
```

## 🌱 Project Status

ManusPrime is under active development. Current focus areas:

- Enhancing model selection algorithms
- Expanding the tool ecosystem
- Improving resource optimization
- Adding new output format capabilities

## 🤝 Contributing

We welcome any friendly suggestions and helpful contributions! Please create issues or submit pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

ManusPrime builds upon the foundation laid by [OpenManus](https://github.com/mannaandpoem/OpenManus). We're grateful to the original creators for their innovative work.

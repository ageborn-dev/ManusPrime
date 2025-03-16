# ManusPrime

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- This HTML comment trick enables image centering in markdown -->
![ManusPrime Logo](assets/manusprime-logo.svg#gh-light-mode-only)
![ManusPrime Logo](assets/manusprime-logo.svg#gh-dark-mode-only)

> A sophisticated multi-model AI agent framework with intelligent task routing, extensible plugin architecture, and advanced resource optimization.

## ✨ Highlights

- **Smart Model Selection** - Automatically routes tasks to the most suitable AI model
- **Multi-Provider Support** - Seamlessly works with OpenAI, Anthropic, Mistral, Gemini, and more
- **Modular Plugin System** - Easily extend capabilities with specialized plugins
- **Advanced Memory System** - Vector-based storage for context-aware responses
- **Resource Optimization** - Intelligent caching, batching, and fallback mechanisms
- **Input Validation** - Enhanced security and prompt optimization
- **Budget Controls** - Fine-grained cost management and tracking

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ageborn-dev/manusprime.git
cd manusprime

# Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Run the server
python server.py
```

Visit `http://localhost:8000` to access the web interface.

## 🧠 Core Architecture

ManusPrime employs a sophisticated multi-model approach that intelligently routes tasks to the optimal AI model:

┌─────────────┐     ┌───────────────┐     ┌────────────────┐     ┌─────────────┐
│ User Input  │────▶│ Task Analysis │────▶│ Model Selection│────▶│ Execution   │
└─────────────┘     └───────────────┘     └────────────────┘     └─────────────┘
       │                    ▲                     ▲                     │
       │                    │                     │                     │
       ▼                    │                     │                     ▼
┌─────────────┐     ┌───────────────┐     ┌────────────────┐     ┌─────────────┐
│ Input       │     │ Vector        │     │ Resource       │     │ Plugin      │
│ Validation  │     │ Memory        │     │ Monitor        │     │ System      │
└─────────────┘     └───────────────┘     └────────────────┘     └─────────────┘

### Model Selection Strategy

ManusPrime intelligently selects the most appropriate model based on:

- **Task Type** - Code generation, creative writing, reasoning, planning, tool use
- **Complexity** - Matches task complexity to model capabilities
- **Budget** - Considers cost implications for different models
- **Available Providers** - Uses available API keys dynamically
- **Performance History** - Learns from past successful interactions

## 📦 Supported AI Models

| Provider | Models | Specialties |
|----------|--------|------------|
| OpenAI | GPT-4o, GPT-4o-mini | General purpose, reasoning, tool use |
| Anthropic | Claude 3.7 Sonnet, Claude 3.5 Haiku | Creative tasks, planning, reasoning |
| Mistral | Mistral Large, Mistral Small, Codestral | Code generation, efficient processing |
| Deepseek | Deepseek Chat, Deepseek Reasoner | Specialized reasoning tasks |
| Gemini | Gemini 2.0 Flash, Gemini 1.5 Pro | Multimodal capabilities, planning |
| Ollama | Various local models | Privacy-sensitive operations |

## 🔌 Plugin Ecosystem

ManusPrime's plugin architecture enables powerful integrations and capabilities:

### Core Plugins

| Category | Plugin | Description |
|----------|--------|-------------|
| Provider | Multiple | Connects to various AI model providers |
| Vector Store | `vector_memory` | Long-term memory using vector embeddings |
| Utility | `input_validator` | Validates and optimizes user inputs |
| Automation | `zapier` | Connects with 5,000+ external services |

### Tool Plugins

| Category | Plugin | Description |
|----------|--------|-------------|
| Browser | `browser_user` | Automates browser interactions and screenshots |
| File System | `file_manager` | Manages file operations with security controls |
| Code Execution | `python_execute` | Safely executes Python code in sandbox |
| Search | `google_search` | Performs web searches for information retrieval |
| Web Crawler | `crawl4ai` | Extracts content from websites and web apps |

## 📊 Resource Management

ManusPrime includes sophisticated resource optimization features:

- **Token Usage Tracking** - Monitors consumption across providers
- **Budget Enforcement** - Sets daily/monthly limits and per-request caps
- **Cost Optimization** - Routes tasks to cost-effective models
- **Smart Caching** - Reduces redundant API calls
- **Batch Processing** - Efficiently handles multiple requests
- **Fallback Chains** - Gracefully handles service disruptions

## 🛠️ Advanced Features

### Vector Memory System

ManusPrime uses vector embeddings to store and retrieve relevant context:

```python
# Memory enhances future interactions
await agent.execute_task("How do I optimize Python code?")
# Later:
await agent.execute_task("What were those optimization tips again?")
# Agent remembers previous interaction and provides context-aware response
```

### Input Validation and Security

Validates inputs before processing to:

- Prevent prompt injection attacks
- Optimize token usage
- Ensure proper formatting for different models
- Improve response quality

### Fallback Chains

Provides resilience through intelligent fallback mechanisms:

- Automatically retries failed requests
- Switches to alternative models when needed
- Tracks provider performance
- Implements exponential backoff

## 💻 Usage Examples

### Command Line Interface

```bash
# One-shot execution
python main.py "Write a regex to extract emails from text"

# Interactive mode
python main.py --interactive

# Specify model
python main.py --model gpt-4o "Explain quantum computing"
```

### Python API

```python
from manusprime.core.agent import ManusPrime

async def main():
    agent = ManusPrime()
    await agent.initialize()
    
    # Execute a task
    result = await agent.execute_task(
        "Create a Python function to download images from a website",
        model="codestral-latest"  # Optional: override model selection
    )
    
    print(result["content"])
    print(f"Cost: ${result['cost']:.4f}")
    
    await agent.cleanup()
```

### Web Server API

```bash
# GET request for task status
curl http://localhost:8000/api/task/3fa85f64-5717-4562-b3fc-2c963f66afa6

# POST request to create task
curl -X POST http://localhost:8000/api/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a blog post about AI agents", "model": "claude-3.7-sonnet"}'
```

## 🔧 Configuration

ManusPrime uses TOML for configuration. Edit `config/default.toml` to customize:

```toml
# Provider configuration
[providers]
default = "openai"  # Default provider to use

# Model costs per 1K tokens
[costs]
"claude-3.7-sonnet" = 0.015
"gpt-4o" = 0.010
"mistral-large-latest" = 0.008

# Budget configuration
[budget]
limit = 5.0  # Daily budget limit

# Active plugins by category
[plugins.active]
browser = "browser_user"
file_system = "file_manager"
code_execution = "python_execute"
search = "google_search"
vector_store = "vector_memory"
web_crawler = "crawl4ai"
utility = "input_validator"
```

## 🔐 Security

ManusPrime includes multiple security features:

- **Sandboxed Code Execution**: Restricted environment for running code
- **Path Traversal Protection**: Prevents unauthorized file access
- **Input Validation**: Checks for injection attacks
- **Rate Limiting**: Prevents resource abuse
- **API Key Protection**: Secure key management via environment variables
- **Whitelist Approach**: Explicit permission for sensitive operations

## 📚 Development and Extension

### Creating Custom Plugins

```python
from plugins.base import Plugin, PluginCategory

class CustomPlugin(Plugin):
    """Custom plugin implementation."""
    
    name = "custom_plugin"
    description = "Description of custom plugin"
    version = "0.1.0"
    category = PluginCategory.UTILITY
    
    async def initialize(self) -> bool:
        # Setup code here
        return True
    
    async def execute(self, **kwargs) -> dict:
        # Implementation here
        return {"success": True, "result": "Operation completed"}
```

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=manusprime tests/
```

## 📋 Requirements

- **Python**: 3.8+
- **OS**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free disk space
- **API Keys**: For desired AI providers (OpenAI, Anthropic, etc.)

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

ManusPrime builds upon innovations from multiple open-source projects and AI research. Special thanks to the developers of the libraries and models that make this project possible.

---

| [GitHub](https://github.com/ageborn-dev/manusprime) | [Documentation](https://ageborn-dev.github.io/manusprime) | [Issues](https://github.com/ageborn-dev/manusprime/issues) |
|:---:|:---:|:---:|
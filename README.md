# 👋 ManusPrime

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ManusPrime is an advanced multi-model AI agent framework that orchestrates various AI models to solve complex tasks. Built on the foundation of [OpenManus](https://github.com/mannaandpoem/OpenManus), ManusPrime takes agent capabilities to the next level by intelligently leveraging the unique strengths of different AI models.

## 🚀 Key Features

- **Multi-Model Orchestration**: Dynamically select the most appropriate AI model for each task
- **Model-Agnostic Architecture**: Support for OpenAI, Anthropic, Google, and open-source models
- **Enhanced Memory System**: Improved context management with vector storage for long-term memory
- **Advanced Planning**: Hierarchical planning with verification and parallel execution
- **Specialized Agents**: Purpose-built agents for coding, research, planning, and more
- **Powerful Tool Ecosystem**: Rich set of tools for interacting with the digital world
- **Cost Optimization**: Use expensive models only when necessary

## 💡 Philosophy

ManusPrime operates on the principle that different AI models have unique strengths and capabilities. Rather than relying on a single model for all tasks, ManusPrime intelligently routes tasks to the most appropriate model:

- Use reasoning-focused models (like Claude) for planning and decision-making
- Leverage specialized coding models for software development
- Employ multimodal models for visual tasks
- Utilize fast, economical models for simple queries

This approach results in better performance, lower costs, and more robust capabilities.

## 🛠️ Architecture

ManusPrime consists of several key components:

1. **Model Abstraction Layer**: Provides a unified interface to different AI models
2. **Model Router**: Intelligently selects the best model for each task
3. **Agent Framework**: Specialized agents for different domains
4. **Memory System**: Manages context and knowledge across conversations
5. **Tool Ecosystem**: Provides capabilities for interacting with the world
6. **Flow Engine**: Coordinates the execution of complex, multi-step tasks

## 📋 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ManusPrime.git
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

ManusPrime requires configuration for the AI model APIs it uses. Add your API keys to the configuration file:

```toml
# Global LLM configuration (Default model)
[llm.default]
provider = "openai"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Replace with your actual API key
max_tokens = 4096
temperature = 0.0

# Anthropic Claude configuration
[llm.claude]
provider = "anthropic"
model = "claude-3-opus-20240229"
api_key = "sk-ant-..."
max_tokens = 4096
temperature = 0.0

# Add more model configurations as needed
```

## 🚀 Quick Start

```python
from manusprime.app.flow import FlowFactory, FlowType
from manusprime.app.agent import ManusPrimeAgent

# Create a ManusPrime agent with multi-model capabilities
agent = ManusPrimeAgent()

# Run the agent with a task
result = await agent.run("Build a simple weather dashboard website using React")
```

## 🔍 Example Use Cases

- **Software Development**: Build applications, debug code, create websites
- **Research & Analysis**: Gather information, analyze data, create reports
- **Content Creation**: Write articles, generate creative content, edit documents
- **Task Automation**: Interact with websites, process data, manage files

## 🌱 Project Status

ManusPrime is under active development. The current focus is on:

- Establishing the model abstraction layer
- Implementing the enhanced memory system
- Refactoring the agent infrastructure for multi-model support

## 🤝 Contributing

We welcome any friendly suggestions and helpful contributions! Please create issues or submit pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

ManusPrime builds upon the foundation laid by [OpenManus](https://github.com/mannaandpoem/OpenManus). We're grateful to the original creators for their innovative work.

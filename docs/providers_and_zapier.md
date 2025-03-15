# AI Providers and Zapier Integration

This document explains how to configure and use the enhanced AI provider system and Zapier integration.

## AI Provider System

The system now supports multiple AI providers, including cloud services and local models.

### Configuration

In your `config.toml`:

```toml
[providers]
default_provider = "openai"  # Default provider to use
enable_local_models = true   # Enable local model support

# OpenAI Configuration
[providers.openai]
type = "cloud"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."  # Set in environment: OPENAI_API_KEY
models = [
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo"
]

# Ollama Local Model Configuration
[providers.ollama]
type = "local"
base_url = "http://localhost:11434"
models = [
    "llama2",
    "codellama",
    "mistral"
]
```

### Task-Specific Models

Configure different models for specific tasks:

```toml
[task_models]
code_generation = "codellama"    # Use CodeLlama for code generation
planning = "gpt-4"              # Use GPT-4 for planning
tool_use = "gpt-4"             # Use GPT-4 for tool usage
default = "gpt-3.5-turbo"      # Default model for other tasks
```

### Using Local Models with Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull desired models:
   ```bash
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```
3. Ensure Ollama is running before starting the application

## Zapier Integration

The Zapier integration allows automation workflows between the system and other services.

### Configuration

1. Set up your Zapier configuration:
   ```toml
   [zapier]
   enabled = true
   webhook_secret = "your-secret"  # Set in environment: ZAPIER_WEBHOOK_SECRET
   api_key = "your-api-key"        # Set in environment: ZAPIER_API_KEY
   allowed_actions = [
       "task_create",
       "task_update",
       "tool_execute"
   ]
   ```

2. Create a Zapier Webhook:
   - In Zapier, create a new Zap
   - Choose "Webhook by Zapier" as the trigger
   - Copy the webhook URL
   - Configure the webhook to handle the action format:
     ```json
     {
       "action": "task_create",
       "data": {
         "title": "New Task",
         "description": "Task details",
         "priority": 1
       }
     }
     ```

### Using the Zapier Tool

The system includes a Zapier tool for AI-driven automation:

```python
# Example tool usage
result = await tool_collection.execute_tool(
    "zapier",
    {
        "webhook_url": "https://hooks.zapier.com/...",
        "action": "task_create",
        "data": {
            "title": "Generated Task",
            "description": "AI-generated task description",
            "priority": 1
        }
    }
)
```

### Available Actions

1. `task_create`: Create a new task
   ```json
   {
     "title": "string",
     "description": "string",
     "priority": "integer",
     "tags": ["string"]
   }
   ```

2. `task_update`: Update an existing task
   ```json
   {
     "task_id": "string",
     "status": "string",
     "progress": "float",
     "result": "object"
   }
   ```

3. `tool_execute`: Execute a system tool
   ```json
   {
     "tool_name": "string",
     "parameters": "object",
     "task_id": "string (optional)"
   }
   ```

### Security

The Zapier integration includes several security features:

1. Webhook signature validation
2. API key authentication
3. Allowed actions list
4. Rate limiting
5. IP whitelisting (optional)

### Environment Variables

Set these environment variables for secure configuration:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
ZAPIER_WEBHOOK_SECRET=your-webhook-secret
ZAPIER_API_KEY=your-api-key
```

## Error Handling

Both the provider system and Zapier integration include comprehensive error handling:

1. Automatic retries for transient failures
2. Rate limit handling
3. Logging of all errors
4. Fallback mechanisms for provider failures

## Monitoring

Track usage and performance:

1. Model usage statistics
2. Cost tracking
3. Success/failure rates
4. Response times
5. Token usage

## Best Practices

1. Always use environment variables for sensitive data
2. Test webhooks in Zapier's test mode first
3. Start with a limited set of allowed actions
4. Monitor usage patterns and costs
5. Regularly update model configurations based on performance
6. Use task-specific models for optimal results
7. Keep local models updated when using Ollama

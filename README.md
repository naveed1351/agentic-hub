# Agentic-hub
A unified repository for agentic AI development—includes single and multi-agent systems, Semantic Kernel, Magentic integration, and reusable components. Provides frameworks, demos, and templates to accelerate creation and deployment of modular, collaborative, and autonomous AI agents.

# Semantic Kernel

Exploring Microsoft's Semantic Kernel framework for building intelligent AI applications. This workshop provides practical experience with real-world AI application patterns using Python and Azure OpenAI.

## Repo Overview

This repo takes you from foundational concepts to advanced implementation patterns through a series of Jupyter notebooks and practical examples. You'll learn how to:

- Build AI applications using Microsoft's Semantic Kernel framework
- Create and orchestrate AI agents with different capabilities and roles
- Construct structured AI workflows using the Process Framework
- Implement enterprise-ready AI features with security and scalability in mind

## Interactive Playground Demo

Experience Semantic Kernel in action through our interactive playground! This visual demonstration allows you to directly engage with the core concepts covered in the workshop.

![Semantic Kernel Playground Demo](playground/assets/sk-playground.gif)

The playground offers a hands-on environment where you can:
- Test semantic functions in real-time
- Explore agent capabilities and interactions
- Experiment with memory and embeddings
- Try out native plugin integration
- See the Process Framework in action

No need to wait until the end of the workshop - you can start exploring the playground at any time to reinforce concepts as you learn them!

For setup instructions and details on how to run the playground, refer to the [Playground README](playground/README.md).

## Prerequisites

- Python 3.10 or higher
- Azure OpenAI API access (API key, endpoint, and deployment name)
- Basic knowledge of Python programming
- Understanding of prompt engineering concepts (helpful but not required)
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)

### Local Dependencies Setup

The project is managed by pyproject.toml and [uv package manager](https://docs.astral.sh/uv/getting-started/installation/).

For local execution init the .venv environment using [uv package manager](https://docs.astral.sh/uv/getting-started/installation/):

```shell
uv sync --prerelease=allow
. ./.venv/bin/activate
```
>OBS! At the time of writing the workshop depends on the prerelease libraries. 

## Getting Started

1. Clone this repository

1. Create a virtual environment:
   
   **Linux/macOS:**
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate
   ```
   
   **Windows:**
   ```cmd
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   venv\Scripts\activate
   ```

1. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```

1. Add your Azure OpenAI credentials to the `.env` file:
   ```
   AZURE_OPENAI_ENDPOINT=https://xxxxxx.openai.azure.com/
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=model name
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=embedding model name
   AZURE_OPENAI_API_KEY=xxxxxxxxxxx
   AZURE_OPENAI_API_VERSION=2025-03-01-preview
   ```

1. Start with the first notebook:
   - Begin with `01-intro-to-semantic-kernel/01-intro.ipynb`, which includes instructions for installing Semantic Kernel and other required packages.


## Modules

### 01. Introduction to Semantic Kernel

Learn the fundamentals of Semantic Kernel:
- Core architectural components (Kernel, AI Services, Plugins)
- Building semantic functions with prompts
- Creating native functions with Python code
- Enabling automatic function calling for AI agents

**Key Notebooks:**
- `01-intro.ipynb`: Core concepts, services, and function creation

### 02. Semantic Kernel Agents

Master the creation and orchestration of AI agents:
- Creating specialized agents with different personas
- Implementing multi-agent communication patterns
- Agent selection strategies and orchestration
- Building agent topologies for complex scenarios
- Integrating plugins with agents for enhanced capabilities

**Key Notebooks:**
- `02.1-agents.ipynb`: Creating and configuring agents
- `02.2-agents-chats.ipynb`: Inter-agent communication and complex patterns

### 03. Semantic Kernel with MCP

Learn to how to connect an SK Agent to MCP:
- Running your MCP server
- Using an Agent in Semantic Kernel to make calls to it

**Key Notebooks:**
- `03.1-sk-with-mcp.ipynb`: Semantic Kernel with MCP example

### 04. Process Framework

Learn to build structured, event-driven AI workflows:
- Understanding the Process Framework architecture
- Defining events, steps, and state management
- Building conversational AI systems with processes
- Implementing complex business logic with AI capabilities
- Creating maintainable and testable AI workflows

**Key Notebooks:**
- `04.1-intro-to-processes.ipynb`: Building stateful, event-driven AI processes

## Semantic-Kernel-agents module



## AI-Agent-with-Deep-Research-AI-Foundry module



## Learning Path

For optimal learning, follow the repository's folders in numerical order.

## Advanced Topics and Resources

For advanced patterns and enterprise deployment scenarios, explore the [Semantic Kernel Advanced Usage](https://github.com/Azure-Samples/semantic-kernel-advanced-usage) repository, which includes:

- Dapr integration for scalable, distributed systems
- Authentication and security patterns
- Natural language to SQL conversion
- Copilot Studio integration
- Microsoft Graph API integration
- Production deployment architecture

## Additional Resources

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/overview/)
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service/)
- [Microsoft Copilot Studio](https://www.microsoft.com/en-us/microsoft-copilot/microsoft-copilot-studio)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Agent O1: Intelligent Conversation Agent Framework

[![Python Version](https://img.shields.io/badge/python-3.11.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview
Agent O1 is a sophisticated conversational AI agent built on top of the Google Gemini API. It implements a structured reasoning framework that breaks down complex queries into a systematic thought process using five key components: Thought, Action, Pause, Observation, and Answer (TAPA framework).

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Available Actions](#available-actions)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [Examples](#examples)
- [Contributing](#contributing)

## Features
- 🧠 Structured reasoning framework (TAPA)
- 🔄 Intelligent conversation loop
- 🌐 Multiple information sources integration
- 🔍 Cross-validation of information
- 🌍 Translation capabilities
- 📊 Mathematical calculations
- 📰 News search functionality
- 🔒 Rate limiting and error handling
- 📝 Comprehensive logging

## Prerequisites
- Python 3.11 or higher
- Google Gemini API key
- NewsAPI key (for news search)
- Google Custom Search API key and Search Engine ID (for web search)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Feel-The-AGI/LLM_agent_o1.git
cd agent-o1
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```env
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
NEWSAPI_KEY=your_newsapi_key
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
GOOGLE_SEARCH_CX=your_search_engine_id
```

## Configuration

The agent can be configured through environment variables and the system prompt. The system prompt defines the agent's behavior and available actions.

### Environment Variables
- `GOOGLE_GEMINI_API_KEY`: Required for accessing the Gemini API
- `NEWSAPI_KEY`: Required for news search functionality
- `GOOGLE_SEARCH_API_KEY`: Required for Google search functionality
- `GOOGLE_SEARCH_CX`: Required for Google Custom Search Engine

## Usage

### Basic Usage
```python
from agent_o1 import query

# Simple query
result = query("What is quantum computing?")
print(result)

# Query with cross-validation
result = query("Who is the current president of France? Please validate this information.")
print(result)
```

### Advanced Usage
```python
from agent_o1 import Agent_o1

# Initialize agent with custom system prompt
agent = Agent_o1(custom_system_prompt)

# Process multiple messages
response1 = agent("What is the population of Paris?")
response2 = agent("How has this changed over the last decade?")
```

## Architecture

### Core Components

1. **Agent Class (`Agent_o1`)**
   - Manages conversation state
   - Processes user input
   - Coordinates with Gemini API
   - Maintains conversation history

2. **Rate Limiter (`RateLimiter`)**
   - Controls API call frequency
   - Prevents hitting rate limits
   - Configurable calls per second

3. **Action Handlers**
   - `wikipedia`: Wikipedia searches
   - `calculate`: Mathematical calculations
   - `simon_blog_search`: Blog search functionality
   - `cross_validate`: Information validation
   - `news`: News article search
   - `translate`: Text translation
   - `google_search`: Web search

### TAPA Framework Flow
```
User Query → Thought → Action → PAUSE → Observation → Answer
```

## Available Actions

| Action | Description | Example Usage |
|--------|-------------|---------------|
| `wikipedia` | Search Wikipedia articles | `wikipedia: quantum computing` |
| `calculate` | Perform mathematical calculations | `calculate: 4 * 7 / 3` |
| `simon_blog_search` | Search Simon's blog | `simon_blog_search: Django` |
| `cross_validate` | Validate information across sources | `cross_validate: France population` |
| `news` | Search news articles | `news: climate change` |
| `translate` | Translate text | `translate: Hello, world!|es` |
| `google_search` | Perform web search | `google_search: Python best practices` |

## Rate Limiting

The `RateLimiter` class implements a token bucket algorithm to control API call frequency:

```python
rate_limiter = RateLimiter(calls_per_second=0.5)
rate_limiter.wait()  # Called before API requests
```

## Error Handling

The agent implements comprehensive error handling:
- API errors
- Rate limit exceptions
- Invalid inputs
- Network timeouts
- Parse errors

All errors are logged and gracefully handled to maintain conversation flow.

## Logging

Logging is configured to track:
- API calls and responses
- Errors and exceptions
- Agent actions and observations
- Performance metrics

Logs are written to `agent.log` with timestamps and severity levels.

## Examples

### Basic Question
```python
result = query("What is quantum computing?")
```

### Mathematical Calculation
```python
result = query("Calculate the area of a circle with radius 5 units.")
```

### Translation
```python
result = query("How do you say 'Hello, how are you?' in Spanish?")
```

### Cross-Validation
```python
result = query("Who is the current president of France? Please validate this information.")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API
- NewsAPI
- Wikipedia API
- Simon Willison's Blog
- Google Custom Search API

---

For more information or support, please open an issue in the repository.

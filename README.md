# IQ-PACE LLM Agentic Framework Implementation

## Overview

This project implements an advanced cognitive agent based on the **IQ-PACE** framework, leveraging the capabilities of the **Google Gemini API**. The agent is designed for structured reasoning and information processing, providing accurate and validated responses to user queries. It features a unique architecture that integrates advanced memory management, action planning, validation, and GPU acceleration to enhance performance.

The **IQ-PACE** framework stands for:

- **I**ntake
- **Q**uery Planning
- **P**erform Action
- **A**nalyze Result
- **C**ross-Validate
- **E**valuate
- **Conclude**

The agent processes queries through these stages to ensure methodical thinking, multi-source validation, uncertainty quantification, metacognitive awareness, and bias mitigation.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [IQ-PACE Framework](#iq-pace-framework)
  - [System Prompt](#system-prompt)
  - [Classes and Modules](#classes-and-modules)
- [Example](#example)
- [Logging](#logging)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Implements the IQ-PACE framework for advanced reasoning.
- Utilizes the Google Gemini API for language generation.
- Unique architecture with modular components for extensibility.
- GPU acceleration for memory storage and retrieval using Milvus.
- Action planning using A* search algorithm.
- Validation engine with Bayesian confidence updating.
- Advanced memory management (short-term, working, long-term).
- Rate limiting and error handling for API calls.
- Supports multi-source validation and cross-referencing.
- Metacognitive awareness with continuous self-monitoring.
- Bias mitigation through source diversity and perspective balancing.

---

## Architecture

### Unique Architecture

The agent features a unique and modular architecture that integrates several advanced components to enhance its reasoning capabilities:

- **ActionManager**: Manages and executes actions with rate limiting, error handling, and GPU support.
- **MemoryBuffer**: Uses Milvus vector database with GPU acceleration for short-term memory storage and retrieval.
- **LongTermMemory**: Manages long-term memory storage with consolidation and retrieval using GPU-accelerated searches.
- **ContextManager**: Handles different types of context (short-term, working, medium-term) with customizable retention.
- **ValidationEngine**: Performs enhanced validation with Bayesian confidence updating.
- **ActionPlanner**: Implements A* search for optimal action planning.
- **IQPACEAgent**: The main agent that orchestrates all components according to the IQ-PACE framework.

### GPU Support

The agent utilizes GPU acceleration to enhance performance in the following areas:

- **Memory Storage and Retrieval**: Milvus vector database operations are accelerated using GPU for faster indexing and searching.
- **Embeddings**: SentenceTransformer models utilize GPU (if available) for efficient encoding.
- **Action Execution**: Certain actions can leverage GPU resources for faster computation.

### Framework Flow

The agent processes user queries through the following stages:

1. **Intake (I)**: Deep analysis of the query to understand context, domain, complexity, and potential biases.
2. **Query Planning (Q)**: Decomposing the query into sub-queries and planning the optimal action path using A* search.
3. **Perform Action (P)**: Executing planned actions with validation, error handling, and state maintenance.
4. **Analyze Result (A)**: Assessing data quality, detecting anomalies, and scoring confidence.
5. **Cross-Validate (C)**: Verifying information across multiple sources and updating confidence levels.
6. **Evaluate (E)**: Holistically assessing results against success criteria and determining if further action is needed.
7. **Conclude**: Synthesizing information, declaring confidence levels, and providing actionable insights.

### Core Principles

- **Deliberate Processing**: Implementing slow, methodical thinking for enhanced accuracy.
- **Multi-Source Validation**: Cross-referencing all information.
- **Uncertainty Quantification**: Explicit confidence scoring.
- **Metacognitive Awareness**: Continuous self-monitoring.
- **Bias Mitigation**: Identifying and correcting potential biases.

---

## Installation

### Prerequisites

- **Python 3.10 or higher**
- [Milvus](https://milvus.io/) vector database installed and running.
- Access to the Google Gemini API with a valid API key.
- (Optional) NewsAPI key for news search functionality.
- (Optional) Google Custom Search API key and CX ID for web search.
- **GPU with CUDA support** (optional but recommended for enhanced performance).

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/iq-pace-agent.git
   cd iq-pace-agent
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the project root directory with the following content:

   ```env
   GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key
   NEWSAPI_KEY=your_newsapi_key  # Optional
   GOOGLE_SEARCH_API_KEY=your_google_search_api_key  # Optional
   GOOGLE_SEARCH_CX=your_google_search_cx  # Optional
   ```

---

## Usage

You can test the agent by running the main script:

```bash
python iq_pace_agent.py
```

This will execute a sample query and display the processing steps and the final result.

---

## Components

### IQ-PACE Framework

The agent implements the **IQ-PACE** framework, which structures the reasoning process into seven stages, ensuring thorough analysis and reliable responses.

1. **Intake (I)**
   - Semantic parsing of the query.
   - Context identification.
   - Knowledge domain classification.
   - Complexity assessment.
   - Success criteria definition.
   - Bias identification.

2. **Query Planning (Q)**
   - Decomposition into sub-queries.
   - Optimal action path planning using A* search.
   - Resource allocation.
   - Risk assessment.

3. **Perform Action (P)**
   - Executing actions like `wikipedia`, `calculate`, `news`, `translate`, `google_search`, etc.
   - Pre-execution validation.
   - Error handling.
   - GPU-accelerated computations where applicable.

4. **Analyze Result (A)**
   - Data quality assessment.
   - Pattern recognition.
   - Confidence scoring.
   - Anomaly and contradiction detection.

5. **Cross-Validate (C)**
   - Multi-source verification.
   - Confidence updating.
   - Contradiction resolution.

6. **Evaluate (E)**
   - Holistic assessment.
   - Success criteria matching.
   - Confidence verification.

7. **Conclude**
   - Synthesized response.
   - Confidence declaration.
   - Source attribution.
   - Actionable insights.

### System Prompt

The `system_prompt` variable contains detailed instructions for the agent, outlining the framework flow, core principles, execution guidelines, and response structure. It guides the agent's behavior during processing, ensuring adherence to the IQ-PACE framework.

### Classes and Modules

#### ConfidenceLevel Enum

Defines confidence levels for responses:

```python
class ConfidenceLevel(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"
```

#### ValidationResult Dataclass

Holds validation results:

```python
@dataclass
class ValidationResult:
    confidence_score: float
    validated_claims: List[str]
    sources: List[str]
    contradictions: List[str]
    uncertainty_factors: List[str]
```

#### ActionResult Dataclass

Holds action execution results:

```python
@dataclass
class ActionResult:
    success: bool
    data: Any
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict
```

#### SearchNode Class

Used for A* search in action planning:

```python
class SearchNode:
    def __init__(self, state: str, cost: float, heuristic: float):
        ...
```

#### ActionPlanner Class

Implements A* search for action planning:

```python
class ActionPlanner:
    def __init__(self):
        ...
    def add_action(self, name: str, func: callable, cost: float):
        ...
    def plan(self, initial_state: str, goal: str) -> List[str]:
        ...
```

#### ValidationEngine Class

Performs validation with Bayesian confidence updating:

```python
class ValidationEngine:
    def __init__(self):
        ...
    def validate(self, claims: List[str], sources: List[Dict]) -> ValidationResult:
        ...
```

#### RateLimiter Class

Rate limiting utility using the token bucket algorithm:

```python
class RateLimiter:
    def __init__(self, calls_per_second: float = 0.5):
        ...
    def wait(self):
        ...
```

#### ActionManager Class

Manages and executes actions with error handling and GPU support:

```python
class ActionManager:
    def __init__(self):
        ...
    def register_action(self, name: str, func: callable):
        ...
```

**Available Actions:**

- `wikipedia`: Knowledge base search.
- `calculate`: Mathematical computation.
- `news`: Current events analysis.
- `translate`: Language translation.
- `google_search`: Web search.
- `analyze`: Pattern recognition.

#### MemoryEntry TypedDict

Defines the structure of a memory entry:

```python
class MemoryEntry(TypedDict):
    text: str
    embedding: List[float]
    type: str
    timestamp: datetime
    metadata: Dict[str, Any]
```

#### MemoryBuffer Class

Manages short-term memory using Milvus with GPU acceleration:

```python
class MemoryBuffer:
    def __init__(self, collection_name: str = "agent_memory", dim: int = 384, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        ...
    def add_entry(self, text: str, entry_type: str, metadata: Dict[str, Any] = None) -> bool:
        ...
    def search_similar(self, query: str, limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        ...
```

**GPU Support in MemoryBuffer:**

- Utilizes `torch.cuda.is_available()` to check for GPU availability.
- Embeddings generated using `SentenceTransformer` are accelerated with GPU.
- Milvus index creation and search operations are configured for GPU acceleration.

#### ContextManager Class

Manages different types of context:

```python
class ContextManager:
    def __init__(self):
        ...
    def push_context(self, context_type: str, data: Any) -> bool:
        ...
    def get_context(self, context_type: str, limit: int = None) -> List[Dict]:
        ...
    def clear_context(self, context_type: str = None):
        ...
```

#### LongTermMemory Class

Handles long-term memory storage and retrieval with GPU acceleration:

```python
class LongTermMemory:
    def __init__(self, collection_name: str = "long_term_memory", dim: int = 384, consolidation_threshold: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        ...
    def store(self, content: str, category: str, importance: float = 0.5, metadata: Dict = None) -> bool:
        ...
    def retrieve(self, query: str, category: str = None, limit: int = 5) -> List[Dict]:
        ...
```

**GPU Support in LongTermMemory:**

- Milvus operations are configured to use GPU for indexing and searching.
- Embeddings for long-term memories are generated using GPU-accelerated models.

#### IQPACEAgent Class

Main agent implementing the IQ-PACE framework:

```python
class IQPACEAgent:
    def __init__(self, system_prompt: str = ""):
        ...
    def process_query(self, query: str) -> str:
        ...
    def _intake_analysis(self, query: str) -> Dict:
        ...
    def _query_planning(self, intake_result: Dict) -> Dict:
        ...
    def _perform_action(self, action: str) -> ActionResult:
        ...
    def _analyze_results(self, results: List[ActionResult]) -> Dict:
        ...
    def _cross_validate(self, analysis: Dict) -> ValidationResult:
        ...
    def _evaluate(self, validation: ValidationResult) -> Dict:
        ...
    def _conclude(self, evaluation: Dict) -> str:
        ...
```

**Unique Features of IQPACEAgent:**

- Orchestrates all components according to the IQ-PACE framework.
- Integrates memory retrieval to provide context-aware responses.
- Uses advanced validation and error handling mechanisms.
- Supports GPU acceleration for enhanced performance.

---

## Example

To process a query using the agent:

```python
def query(question: str, max_iterations: int = 3) -> str:
    agent = IQPACEAgent(system_prompt)
    return agent.process_query(question)

if __name__ == "__main__":
    result = query("What are the latest developments in quantum computing and their potential impact on cryptography?")
    print("\nFinal result:", result)
```

---

## Logging

Logging is configured to output detailed information to `agent.log`:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='agent.log'
)
```

This includes timestamps, log levels, and messages for tracing the agent's operations.

---

## Environment Variables

Set up the following environment variables, preferably in a `.env` file:

- `GOOGLE_GEMINI_API_KEY`: Your Google Gemini API key (required).
- `NEWSAPI_KEY`: Your NewsAPI key (optional).
- `GOOGLE_SEARCH_API_KEY`: Your Google Search API key (optional).
- `GOOGLE_SEARCH_CX`: Your Google Search CX ID (optional).

The agent will raise an error if `GOOGLE_GEMINI_API_KEY` is not found.

---

## Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `google-generativeai`
- `pymilvus`
- `sentence-transformers`
- `torch` (with CUDA support for GPU acceleration)
- `googletrans`
- `httpx`
- `requests`
- `python-dotenv`

**Note:** Ensure that your system has the necessary drivers and CUDA toolkit installed for GPU support with PyTorch.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

# Additional Details

## Action Execution

### Available Actions

- **wikipedia**: `[query]` - Searches Wikipedia for the given query.
- **calculate**: `[expression]` - Safely evaluates a mathematical expression.
- **news**: `[query]` - Fetches recent news articles related to the query.
- **translate**: `[text|target_language]` - Translates the text into the target language.
- **google_search**: `[query]` - Performs a Google search for the query.
- **analyze**: `[data]` - Performs pattern recognition on the data.

### Action Protocol

- **Pre-execution Validation**: Checks if the action can be executed.
- **Resource Availability Check**: Ensures necessary resources are available.
- **Rate Limit Compliance**: Observes API rate limits.
- **Error Handling Preparation**: Sets up try-except blocks.
- **Result Buffering**: Stores results for analysis.
- **State Maintenance**: Updates internal state as needed.

## Validation Engine

- Implements Bayesian updating for confidence scores.
- Considers source reliability and contradiction penalties.
- Updates confidence levels based on new evidence.

## Memory Management

### MemoryBuffer

- Uses Milvus for short-term memory storage with GPU acceleration.
- Embeds text using `SentenceTransformer` with GPU support.
- Retrieves similar memories for context-aware processing.

### LongTermMemory

- Stores consolidated memories for long-term retention.
- Uses clustering algorithms to group similar memories.
- Consolidates memories when a threshold is reached.
- Employs GPU acceleration for embedding and searching.

## Context Management

- **Short-Term Memory**: Stores recent queries and active contexts.
- **Working Memory**: Manages current tasks, goals, and interim results.
- **Medium-Term Memory**: Holds conversation summaries and thought patterns.

## Error Handling and Rate Limiting

- **RateLimiter**: Ensures API calls comply with rate limits using the token bucket algorithm.
- **Error Handling**: Catches exceptions, logs errors, and provides fallback mechanisms.

## Execution Guidelines

- Maintain explicit reasoning chains throughout processing.
- Quantify uncertainty at each stage of the framework.
- Document all assumptions made during processing.
- Identify and mitigate potential biases.
- Consider alternative hypotheses and viewpoints.
- Implement progressive validation strategies.
- Practice metacognitive monitoring for self-awareness.

## Response Structure

- **Validated Claims**: Present confirmed information.
- **Confidence Scores**: Declare the confidence level explicitly.
- **Source Attribution**: Cite sources used for information.
- **Uncertainty Factors**: Acknowledge any uncertainties.
- **Limitations**: Disclose limitations of the response.
- **Alternative Perspectives**: Consider different viewpoints.
- **Future Considerations**: Suggest areas for further exploration.
- **Actionable Insights**: Provide practical recommendations.

---

By following the IQ-PACE framework and utilizing the components described, the agent aims to provide high-quality, validated, and insightful responses to user queries. The unique architecture and GPU support enhance performance and enable advanced reasoning capabilities.
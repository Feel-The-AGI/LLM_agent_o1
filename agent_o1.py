"""
This is Agent O1, a simple agent that uses the Google Gemini API to generate responses.
She thinks with second thought process.
This is a custom framework.
Thought, Action, Pause, Observation, and Answer.
We will implement a loop as well
"""


import os
import re
import httpx
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Any, Dict
import ast  #  safer eval
import time  #  rate limiting
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import json
from datetime import datetime
import requests
from googletrans import Translator
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='agent.log'
)

load_dotenv()

if not os.getenv("GOOGLE_GEMINI_API_KEY"):
    raise ValueError("GOOGLE_GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

class RateLimiter:
    """
    A rate limiting utility to control the frequency of API calls.

    This class implements a simple rate limiting mechanism to prevent
    exceeding API rate limits by enforcing a minimum time interval between calls.

    Attributes:
        calls_per_second (float): Maximum number of calls allowed per second
        last_call (float): Timestamp of the last API call

    Example:
        >>> limiter = RateLimiter(calls_per_second=0.5)
        >>> limiter.wait()  # Will pause if needed to maintain rate limit
    """
    def __init__(self, calls_per_second=0.5):
        """
        Initialize the rate limiter.

        Args:
            calls_per_second (float): Maximum number of calls allowed per second
        """
        self.calls_per_second = calls_per_second
        self.last_call = 0

    def wait(self):
        """
        Wait if necessary to maintain the rate limit.

        This method calculates the time since the last call and sleeps if
        needed to ensure the rate limit is not exceeded.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_call
        if time_since_last < (1 / self.calls_per_second):
            time.sleep((1 / self.calls_per_second) - time_since_last)
        self.last_call = time.time()

rate_limiter = RateLimiter()

# Safer calculation function
def calculate(expression: str) -> float:
    """
    Safely evaluate mathematical expressions.
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Verify that the expression only contains safe operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
                raise ValueError("Invalid mathematical expression")
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)):
                    raise ValueError("Invalid operator in expression")
        
        # Evaluate the expression
        result = eval(compile(tree, '<string>', 'eval'))
        return float(result)
    except Exception as e:
        logging.error(f"Calculation error: {str(e)} for expression: {expression}")
        raise ValueError(f"Invalid calculation: {str(e)}")

# Enhanced Wikipedia search
def wikipedia(query: str) -> str:
    """
    Search Wikipedia with error handling and rate limiting.
    """
    try:
        rate_limiter.wait()
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("query", {}).get("search"):
            logging.warning(f"No Wikipedia results found for: {query}")
            return "No results found on Wikipedia for this query."
        
        snippet = data["query"]["search"][0]["snippet"]
        clean_snippet = re.sub(r'<[^>]+>', '', snippet)
        logging.info(f"Successfully retrieved Wikipedia info for: {query}")
        return clean_snippet
    except httpx.TimeoutException:
        logging.error(f"Wikipedia API timeout for query: {query}")
        return "Wikipedia search timed out. Please try again."
    except Exception as e:
        logging.error(f"Wikipedia search error: {str(e)} for query: {query}")
        return f"Error searching Wikipedia: {str(e)}"

# Enhanced blog search
def simon_blog_search(query: str) -> str:
    """
    Search Simon's blog with error handling and rate limiting.
    """
    try:
        rate_limiter.wait()
        response = httpx.get(
            "https://datasette.simonwillison.net/simonwillisonblog.json",
            params={
                "sql": """
                select
                    blog_entry.title || ': ' || substr(html_strip_tags(blog_entry.body), 0, 1000) as text,
                    blog_entry.created
                from
                    blog_entry join blog_entry_fts on blog_entry.rowid = blog_entry_fts.rowid
                where
                    blog_entry_fts match escape_fts(:q)
                order by
                    blog_entry_fts.rank
                limit 5
                """,
                "q": query
            },
            timeout=10.0
        )
        response.raise_for_status()
        results = response.json()
        
        if not results or len(results) == 0:
            logging.warning(f"No blog results found for: {query}")
            return "No matching blog entries found."
            
        return results[0]["text"]
    except httpx.HTTPError as e:  # Changed from TimeoutException
        logging.error(f"Blog search HTTP error: {str(e)} for query: {query}")
        return "Blog search failed. Please try again."
    except Exception as e:
        logging.error(f"Blog search error: {str(e)} for query: {query}")
        return f"Error searching blog: {str(e)}"

# Enhanced Agent class
class Agent_o1:
    """
    An AI agent that uses the Google Gemini API to generate responses.

    This agent implements a structured reasoning framework with five components:
    Thought, Action, Pause, Observation, and Answer. It maintains a conversation
    history and can execute various actions like web searches and calculations.

    Attributes:
        system (str): The system prompt that defines the agent's behavior
        messages (list): History of conversation messages
        total_tokens (int): Counter for total tokens used

    Example:
        >>> agent = Agent_o1(system_prompt)
        >>> response = agent("What is quantum computing?")
    """

    def __init__(self, system: str = ""):
        """
        Initialize the agent with an optional system prompt.

        Args:
            system (str): The system prompt that defines the agent's behavior
        """
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
        self.total_tokens = 0
        logging.info("Agent initialized with system prompt")
    
    def __call__(self, message: str) -> str:
        """
        Process a user message and generate a response.

        Args:
            message (str): The user's input message

        Returns:
            str: The agent's response

        Raises:
            Exception: If there's an error processing the message
        """
        try:
            self.messages.append({"role": "user", "content": message})
            result = self.execute()
            self.messages.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            logging.error(f"Agent call error: {str(e)}")
            return f"An error occurred: {str(e)}"
    
    def execute(self) -> str:
        """
        Execute the agent's reasoning process using the Gemini API.

        Returns:
            str: The generated response from the model

        Raises:
            Exception: If there's an error during execution
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            prompt_parts = []
            for message in self.messages:
                role_prefix = f"{message['role'].title()}: " if message['role'] != 'system' else ""
                prompt_parts.append(f"{role_prefix}{message['content']}")
            
            full_prompt = "\n".join(prompt_parts)
            response = model.generate_content(full_prompt)
            
            if not response.text:
                logging.warning("Empty response from model")
                return "I apologize, but I couldn't generate a proper response. Let me try a different approach."
            
            return response.text
        except Exception as e:
            logging.error(f"Execute error: {str(e)}")
            return f"An error occurred: {str(e)}. Let me try a different approach."
    
    def process_validation_results(self, validation_results: str) -> str:
        """Process and interpret cross-validation results."""
        try:
            results = json.loads(validation_results)
            
            if not results.get("validated_claims"):
                return "I couldn't validate this information with sufficient confidence."
            
            confidence_level = results["confidence_score"]
            confidence_text = (
                "high" if confidence_level > 0.7
                else "moderate" if confidence_level > 0.4
                else "low"
            )
            
            response = [
                f"Based on {results['number_of_sources']} sources, "
                f"with {confidence_text} confidence ({confidence_level:.2f}), "
                "I can confirm the following:"
            ]
            
            for claim in results["validated_claims"]:
                response.append(f"- {claim}")
            
            response.append("\nSources consulted:")
            response.extend([f"- {source}" for source in results["sources"]])
            
            return "\n".join(response)
        except Exception as e:
            logging.error(f"Error processing validation results: {str(e)}")
            return str(validation_results)  # Return raw results if processing fails

# Change 'prompt' to 'system_prompt' and enhance it
system_prompt = """
You are an advanced AI agent operating within a structured reasoning framework that consists of five key components:
Thought → Action → PAUSE → Observation → Answer

1. THOUGHT PROCESS:
- Begin each step with "Thought: " followed by your reasoning
- Break down complex problems into smaller steps
- Consider what information you need and which action would be most appropriate
- Explain your thinking process clearly and logically

2. ACTION EXECUTION:
- After thinking, choose ONE of these available actions:
  * news: [query]
    - Search for recent news articles
    - Example: news: climate change
    - Returns relevant news articles with summaries

  * translate: [text|target_language]
    - Translate text to target language
    - Example: translate: Hello, world!|es
    - Returns original and translated text

  * cross_validate: [query]
    - Validates information across multiple sources
    - Example: cross_validate: England borders
    - Returns validated claims with confidence scores
    - Use for fact-checking important information

  * calculate: [expression]
    - For mathematical calculations
    - Example: calculate: 4 * 7 / 3
    - Use floating point numbers when needed
    - Only basic mathematical operations allowed

  * wikipedia: [query]
    - To search Wikipedia for factual information
    - Example: wikipedia: England
    - Use specific search terms for better results

  * simon_blog_search: [query]
    - To search Simon Willison's blog
    - Example: simon_blog_search: Django
    - Useful for technical information and programming topics

  * google_search: [query]
    - Perform a Google search for any topic
    - Example: google_search: best practices for Python
    - Returns titles, URLs, and snippets from top results
    - Use for current information and general web search

- Format your action exactly as: "Action: [action_name]: [input]"
- Always follow an action with "PAUSE"

3. PAUSE:
- After every action, write "PAUSE" on a new line
- This indicates you're waiting for the observation
- Never continue without receiving an observation

4. OBSERVATION:
- You will receive an observation after each PAUSE
- This is the result of your action
- Use this information to inform your next thought
- If the observation isn't helpful, try a different action

5. ANSWER:
- Only provide a final answer when you have enough information
- Start your final response with "Answer: "
- Make your answer clear, concise, and directly address the original question
- Include relevant details from your observations
- If you need more information, continue the thought process instead

IMPORTANT RULES:
- Always follow the Thought → Action → PAUSE → Observation sequence
- Only one action can be performed at a time
- Don't make assumptions - use actions to gather information
- If an action fails, try a different approach
- Stay focused on the original question
- Be thorough but efficient

Example session:
Question: What is the population of Paris?

Thought: I should look up Paris on Wikipedia to find population information.
Action: wikipedia: Paris population
PAUSE

Observation: Paris has an estimated population of 2,102,650 for the city proper as of 2020.

Answer: The population of Paris proper is 2,102,650 (as of 2020).

---
Begin your response to each question with "Thought: " and follow the framework precisely.
""".strip()

# Now we define the query function
action_re = re.compile('^Action: (\w+): (.*)')

"""
ADDING THE ACTIONS TO THE AGENT
"""

# Move these functions before the known_actions dictionary

def cross_validate(query: str) -> str:
    """
    Cross-validate information from multiple sources.
    Returns validated information with confidence score.
    """
    try:
        # Gather information from multiple sources
        wiki_results = wikipedia_multiple(query)
        blog_result = simon_blog_search(query)
        
        if blog_result and blog_result != "No matching blog entries found.":
            blog_results = [{"title": "Blog Entry", "content": blog_result, "source": "blog"}]
        else:
            blog_results = []
        
        all_sources = wiki_results + blog_results
        
        if not all_sources:
            return "Could not gather enough information for validation."
        
        # Extract potential claims from sources
        claims = []
        for source in all_sources:
            # Split content into sentences (rough approximation)
            sentences = source["content"].split(". ")
            claims.extend(sentences)
        
        # Validate claims
        validated_claims, confidence = validate_information(claims, all_sources)
        
        if not validated_claims:
            return "Could not validate any claims with sufficient confidence."
        
        # Format response
        response = {
            "validated_claims": validated_claims,
            "confidence_score": confidence,
            "number_of_sources": len(all_sources),
            "sources": [f"{s['source']}: {s['title']}" for s in all_sources]
        }
        
        return json.dumps(response, indent=2)
    except Exception as e:
        logging.error(f"Cross-validation error: {str(e)}")
        return f"Error during cross-validation: {str(e)}"

def news_search(query: str, max_results: int = 3) -> str:
    """
    Search for news articles using NewsAPI.
    """
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return "News API key not configured."
        
        rate_limiter.wait()
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "apiKey": api_key,
                "pageSize": max_results,
                "language": "en",
                "sortBy": "relevancy"
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("articles"):
            return f"No news articles found for: {query}"
        
        articles = []
        for article in data["articles"]:
            articles.append(
                f"Title: {article['title']}\n"
                f"Source: {article['source']['name']}\n"
                f"Date: {article['publishedAt'][:10]}\n"
                f"Summary: {article['description']}\n"
            )
        
        return "\n---\n".join(articles)
    except Exception as e:
        logging.error(f"News API error: {str(e)}")
        return f"Error fetching news: {str(e)}"

def translate_text(text: str, target_lang: str = "en") -> str:
    """
    Translate text using Google Translate.
    Format: 'text|target_language'
    Example: 'Hello, world!|es' to translate to Spanish
    """
    try:
        translator = Translator()
        rate_limiter.wait()
        
        # Parse input
        parts = text.split('|')
        if len(parts) != 2:
            return "Invalid format. Use: 'text|target_language'"
        
        text_to_translate, target = parts
        
        # Detect source language
        detection = translator.detect(text_to_translate)
        translation = translator.translate(text_to_translate, dest=target)
        
        return (f"Translation from {detection.lang} to {target}:\n"
                f"Original: {text_to_translate}\n"
                f"Translated: {translation.text}")
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return f"Error during translation: {str(e)}"

def google_search(query: str, num_results: int = 3) -> str:
    """
    Perform a Google search using Google Custom Search API.
    """
    try:
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        cx = os.getenv("GOOGLE_SEARCH_CX")
        
        if not api_key or not cx:
            return "Google Search API credentials not configured."
        
        if cx == "YOUR_SEARCH_ENGINE_ID":
            return "Please configure a valid Search Engine ID in .env file"
        
        rate_limiter.wait()
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                'key': api_key,
                'cx': cx,
                'q': query,
                'num': min(num_results, 10),
            },
            timeout=10.0
        )
        
        if response.status_code == 429:
            return "Rate limit exceeded for Google Search API"
        elif response.status_code != 200:
            logging.error(f"Google Search API error: {response.status_code} - {response.text}")
            return f"Error: Could not complete Google search. Status code: {response.status_code}"
            
        data = response.json()
        
        if 'items' not in data:
            return f"No results found for: {query}"
        
        results = []
        for item in data['items'][:num_results]:
            results.append(
                f"Title: {item['title']}\n"
                f"URL: {item['link']}\n"
                f"Snippet: {item['snippet']}\n"
            )
        
        return "\n---\n".join(results)
    except Exception as e:
        logging.error(f"Google Search API error: {str(e)}")
        return f"Error performing Google search: {str(e)}"

# Now define known_actions after all functions are defined
known_actions = {
    "wikipedia": wikipedia,
    "calculate": calculate,
    "simon_blog_search": simon_blog_search,
    "cross_validate": cross_validate,
    "news": news_search,
    "translate": translate_text,
    "google_search": google_search
}

# Add these new validation functions
def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate the similarity ratio between two text strings.

    Uses SequenceMatcher to compute a similarity score between 0 and 1,
    where 1 indicates identical texts and 0 indicates completely different texts.

    Args:
        text1 (str): First text string to compare
        text2 (str): Second text string to compare

    Returns:
        float: Similarity score between 0 and 1

    Example:
        >>> score = similarity_score("hello world", "hello earth")
        >>> print(score)  # Returns a value between 0 and 1
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def validate_information(claims: List[str], sources: List[Dict]) -> Tuple[List[str], float]:
    """
    Cross-validate information from multiple sources and return validated claims
    with confidence scores.
    """
    validated_claims = []
    confidence_scores = []
    
    for claim in claims:
        claim_scores = []
        supporting_evidence = 0
        
        for source in sources:
            score = similarity_score(claim, source['content'])
            claim_scores.append(score)
            if score > 0.3:  # Threshold for supporting evidence
                supporting_evidence += 1
        
        avg_confidence = sum(claim_scores) / len(claim_scores)
        source_agreement = supporting_evidence / len(sources)
        final_confidence = (avg_confidence + source_agreement) / 2
        
        if final_confidence > 0.2:  # Minimum confidence threshold
            validated_claims.append(claim)
            confidence_scores.append(final_confidence)
    
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    return validated_claims, overall_confidence

# Add new validation-specific actions
def wikipedia_multiple(query: str, limit: int = 3) -> List[Dict]:
    """Get multiple Wikipedia results for validation."""
    try:
        rate_limiter.wait()
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": limit
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data["query"]["search"][:limit]:
            clean_snippet = re.sub(r'<[^>]+>', '', item["snippet"])
            results.append({
                "title": item["title"],
                "content": clean_snippet,
                "source": "wikipedia"
            })
        return results
    except Exception as e:
        logging.error(f"Multiple Wikipedia search error: {str(e)}")
        return []

# Update system_prompt to include the new google_search action
# Add this after the other actions in the system prompt
system_prompt = system_prompt.replace(
    "  * simon_blog_search: [query]",
    """  * simon_blog_search: [query]
    - To search Simon Willison's blog
    - Example: simon_blog_search: Django
    - Useful for technical information and programming topics

  * google_search: [query]
    - Perform a Google search for any topic
    - Example: google_search: best practices for Python
    - Returns titles, URLs, and snippets from top results
    - Use for current information and general web search"""
)

# Integrate actions with Agent
def query(question: str, max_returns: int = 10) -> str:
    """
    Process a question through the agent's thought-action loop.

    This function implements the main interaction loop where the agent thinks,
    takes actions, and processes observations until reaching an answer.

    Args:
        question (str): The user's question to be answered
        max_returns (int): Maximum number of iterations in the thought-action loop

    Returns:
        str: The final answer or response

    Example:
        >>> result = query("What is the capital of France?")
        >>> print(result)
    """
    i = 0
    bot = Agent_o1(system_prompt)
    next_prompt = question
    while i < max_returns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}:? {action_input}")
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("observation:", observation)
            next_prompt = f"Observation: {observation}"
        else:
            return result

# Run query
print(query("What is quantum computing?"))

# Run query
print(query("Calculate the area of a circle with radius 5 units."))
print(query("How do you say 'Hello, how are you?' in Spanish?"))
print(query("What are the latest developments in artificial intelligence?"))
print(query("What are the best practices for Python programming?"))
print(query("Who is the current president of France? Please validate this information."))
print(query("What has Simon written about SQLite?"))
print(query("Compare the population of New York and Tokyo, and convert the difference to scientific notation."))

# Replace the multiple test queries with just one to start
if __name__ == "__main__":
    try:
        print("Testing the agent with a simple query...")
        result = query("What is quantum computing?")
        print("\nFinal result:", result)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        logging.error(f"Execution error: {str(e)}")

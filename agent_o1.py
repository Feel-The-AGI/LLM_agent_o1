"""
IQ-PACE LLM Agentic Framework Implementation
Advanced Agentic Reasoning System using Google Gemini API

Framework Flow:
User Query -> Intake -> Query Planning -> Perform Action -> Analyze Result -> Cross-Validate -> Evaluate -> Conclude
"""

import os
import re
import httpx
import logging
import json
import time
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from difflib import SequenceMatcher
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import ast
from googletrans import Translator


system_prompt = """
You are an advanced cognitive agent implementing the IQ-PACE framework - a state-of-the-art approach to agentic reasoning and information processing.

CORE PRINCIPLES:
1. Deliberate Processing: Implement slow, methodical thinking for enhanced accuracy
2. Multi-Source Validation: Cross-reference all information
3. Uncertainty Quantification: Explicit confidence scoring
4. Metacognitive Awareness: Continuous self-monitoring
5. Bias Mitigation: Active identification and correction of potential biases


FRAMEWORK FLOW:
User Query -> Intake -> Query Planning -> Perform Action -> Analyze Result -> Cross-Validate -> Evaluate -> Conclude

1. INTAKE (I):
Begin with "Intake Analysis:"
- Deep semantic parsing of query
- Context identification and boundary setting
- Knowledge domain classification
- Complexity assessment (scale 0-1)
- Success criteria definition
- Required confidence threshold determination
- Potential bias identification
- Resource requirement estimation

2. QUERY PLANNING (Q):
Begin with "Query Strategy:"
- Decomposition into atomic sub-queries
- A* search for optimal action path
- Priority queue implementation
- Fallback strategy definition
- Resource allocation planning
- Validation requirement specification
- Cost-benefit analysis
- Risk assessment

3. PERFORM ACTION (P):
Format: "Action Execution: [action_name]: [input]"
Available Actions:
* wikipedia: [query] - Knowledge base search
* calculate: [expression] - Mathematical computation
* news: [query] - Current events analysis
* translate: [text|target_language] - Language translation
* cross_validate: [query] - Multi-source verification
* google_search: [query] - Web search
* analyze: [data] - Pattern recognition

Action Protocol:
- Pre-execution validation
- Resource availability check
- Rate limit compliance
- Error handling preparation
- Result buffering
- State maintenance

4. ANALYZE RESULT (A):
Begin with "Result Analysis:"
- Data quality assessment
- Pattern recognition
- Anomaly detection
- Confidence scoring
- Information gap identification
- Contradiction detection
- Relevance evaluation
- Bias detection

5. CROSS-VALIDATE (C):
Begin with "Validation Protocol:"
- Multi-source triangulation
- Bayesian confidence updating
- Contradiction resolution
- Source reliability weighting
- Temporal relevance assessment
- Semantic consistency check
- Expert knowledge integration
- Uncertainty quantification

6. EVALUATE (E):
Begin with "Evaluation Summary:"
- Holistic assessment
- Success criteria matching
- Confidence threshold verification
- Information completeness check
- Uncertainty analysis
- Bias impact assessment
- Additional iteration need evaluation
- Risk assessment

7. CONCLUDE:
Begin with "Final Response:"
- Synthesized information presentation
- Confidence level declaration
- Source attribution
- Uncertainty acknowledgment
- Limitation disclosure
- Future investigation suggestions
- Alternative viewpoint consideration
- Actionable insights

QUALITY CONTROL PROTOCOLS:

1. Confidence Scoring System:
- Minimum threshold: 0.2
- Supporting evidence threshold: 0.3
- Source diversity requirement: ≥2
- Bayesian confidence updating
- Contradiction penalty
- Time decay factor
- Expert source bonus
- Verification multiplier

2. Error Prevention:
- Input validation
- Assumption testing
- Logic verification
- Source credibility check
- Temporal relevance check
- Contradiction detection
- Bias identification
- Hallucination prevention

3. Bias Mitigation:
- Source diversity requirement
- Perspective balancing
- Cultural sensitivity check
- Temporal context consideration
- Expert knowledge integration
- Stakeholder impact analysis
- Alternative viewpoint inclusion
- Assumption challenging

4. Response Requirements:
- Explicit confidence scoring
- Source attribution
- Uncertainty acknowledgment
- Limitation disclosure
- Bias consideration
- Alternative viewpoint inclusion
- Future investigation suggestions
- Actionable insights

EXECUTION GUIDELINES:

1. Always maintain explicit reasoning chains
2. Quantify uncertainty at each stage
3. Document all assumptions
4. Identify potential biases
5. Consider alternative hypotheses
6. Maintain source diversity
7. Implement progressive validation
8. Practice metacognitive monitoring

RESPONSE STRUCTURE:
1. Validated Claims
2. Confidence Scores
3. Source Attribution
4. Uncertainty Factors
5. Limitations
6. Alternative Perspectives
7. Future Considerations
8. Actionable Insights


Example Flow:

User: "What are the implications of quantum computing for cybersecurity?"

Intake Analysis:
- Domain: Computer Science, Cryptography
- Complexity: 0.8 (High)
- Required Confidence: 0.7
- Critical Factors: Technical accuracy, current developments
- Potential Biases: Technological hype

Query Strategy:
1. Quantum computing fundamentals
2. Current cryptography landscape
3. Quantum impact assessment
4. Timeline projections
5. Expert perspectives
6. Industry preparations

[Continue with framework stages...]

Remember:
- Maintain methodical processing
- Prioritize accuracy over speed
- Validate all claims
- Acknowledge uncertainties
- Consider alternative viewpoints
- Provide actionable insights
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='agent.log'
)

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_GEMINI_API_KEY"):
    raise ValueError("GOOGLE_GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

class ConfidenceLevel(Enum):
    """Enumeration of confidence levels for responses"""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"

@dataclass
class ValidationResult:
    """Data class for validation results"""
    confidence_score: float
    validated_claims: List[str]
    sources: List[str]
    contradictions: List[str]
    uncertainty_factors: List[str]

@dataclass
class ActionResult:
    """Data class for action execution results"""
    success: bool
    data: Any
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict

class SearchNode:
    """A* search node for action planning"""
    def __init__(self, state: str, cost: float, heuristic: float):
        self.state = state
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic
        self.parent = None
        self.action = None

    def __lt__(self, other):
        return self.total_cost < other.total_cost

class ActionPlanner:
    """A* search implementation for action planning"""
    def __init__(self):
        self.actions = {}
        self.heuristics = {}

    def add_action(self, name: str, func: callable, cost: float):
        self.actions[name] = (func, cost)

    def set_heuristic(self, state: str, goal: str, value: float):
        self.heuristics[(state, goal)] = value

    def plan(self, initial_state: str, goal: str) -> List[str]:
        frontier = []
        heapq.heappush(frontier, SearchNode(initial_state, 0, self.heuristics.get((initial_state, goal), 0)))
        explored = set()
        
        while frontier:
            current = heapq.heappop(frontier)
            if current.state == goal:
                return self._reconstruct_path(current)
            
            explored.add(current.state)
            for action, (_, cost) in self.actions.items():
                next_state = self._apply_action(current.state, action)
                if next_state not in explored:
                    node = SearchNode(
                        next_state,
                        current.cost + cost,
                        self.heuristics.get((next_state, goal), 0)
                    )
                    node.parent = current
                    node.action = action
                    heapq.heappush(frontier, node)
        
        return []

    def _apply_action(self, state: str, action: str) -> str:
        # Simplified state transition
        return f"{state}_{action}"

    def _reconstruct_path(self, node: SearchNode) -> List[str]:
        path = []
        while node.parent:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))

class ValidationEngine:
    """Enhanced validation system with Bayesian updating"""
    def __init__(self):
        self.prior_confidence = 0.5
        self.min_threshold = 0.2
        self.support_threshold = 0.3

    def validate(self, claims: List[str], sources: List[Dict]) -> ValidationResult:
        validated_claims = []
        contradictions = []
        uncertainty_factors = []
        
        for claim in claims:
            confidence = self._calculate_confidence(claim, sources)
            if confidence >= self.min_threshold:
                validated_claims.append(claim)
            else:
                uncertainty_factors.append(f"Low confidence ({confidence:.2f}) for claim: {claim}")

        return ValidationResult(
            confidence_score=self._aggregate_confidence(validated_claims, sources),
            validated_claims=validated_claims,
            sources=[s["source"] for s in sources],
            contradictions=contradictions,
            uncertainty_factors=uncertainty_factors
        )

    def _calculate_confidence(self, claim: str, sources: List[Dict]) -> float:
        scores = []
        for source in sources:
            similarity = SequenceMatcher(None, claim.lower(), source["content"].lower()).ratio()
            scores.append(similarity)
        
        if not scores:
            return 0.0
        
        # Bayesian updating
        confidence = self.prior_confidence
        for score in scores:
            confidence = self._update_confidence(confidence, score)
        
        return confidence

    def _update_confidence(self, prior: float, evidence: float) -> float:
        # Simplified Bayesian update
        likelihood = evidence
        normalization = (likelihood * prior + (1 - likelihood) * (1 - prior))
        if normalization == 0:
            return 0.0
        return (likelihood * prior) / normalization

    def _aggregate_confidence(self, claims: List[str], sources: List[Dict]) -> float:
        if not claims:
            return 0.0
        return np.mean([self._calculate_confidence(claim, sources) for claim in claims])

class RateLimiter:
    """Rate limiting utility with token bucket algorithm"""
    def __init__(self, calls_per_second: float = 0.5):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        self.min_interval = 1.0 / calls_per_second

    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_call = time.time()

class ActionManager:
    """Manages and executes actions with rate limiting and error handling"""
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.actions = {}
        self._register_default_actions()

    def _register_default_actions(self):
        """Register default available actions"""
        self.register_action("wikipedia", self._wikipedia_search)
        self.register_action("calculate", self._calculate)
        self.register_action("news", self._news_search)
        self.register_action("translate", self._translate)
        self.register_action("google_search", self._google_search)

    def register_action(self, name: str, func: callable):
        """Register a new action"""
        self.actions[name] = func

    @lru_cache(maxsize=100)
    def _wikipedia_search(self, query: str) -> ActionResult:
        """Enhanced Wikipedia search with caching"""
        try:
            self.rate_limiter.wait()
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
                return ActionResult(
                    success=False,
                    data="No results found",
                    confidence=0.0,
                    source="wikipedia",
                    timestamp=datetime.now(),
                    metadata={}
                )
            
            result = data["query"]["search"][0]
            return ActionResult(
                success=True,
                data=re.sub(r'<[^>]+>', '', result["snippet"]),
                confidence=0.8,
                source="wikipedia",
                timestamp=datetime.now(),
                metadata={"title": result["title"]}
            )
        except Exception as e:
            logging.error(f"Wikipedia search error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="wikipedia",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _calculate(self, expression: str) -> ActionResult:
        """Safe mathematical calculations"""
        try:
            tree = ast.parse(expression, mode='eval')
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num)):
                    raise ValueError("Invalid mathematical expression")
            
            result = eval(compile(tree, '<string>', 'eval'))
            return ActionResult(
                success=True,
                data=float(result),
                confidence=1.0,
                source="calculator",
                timestamp=datetime.now(),
                metadata={"expression": expression}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="calculator",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _news_search(self, query: str) -> ActionResult:
        """News search with API key handling"""
        try:
            api_key = os.getenv("NEWSAPI_KEY")
            if not api_key:
                raise ValueError("NewsAPI key not configured")
            
            self.rate_limiter.wait()
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": api_key,
                    "pageSize": 3,
                    "language": "en"
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("articles"):
                return ActionResult(
                    success=False,
                    data="No news articles found",
                    confidence=0.0,
                    source="newsapi",
                    timestamp=datetime.now(),
                    metadata={}
                )
            
            articles = [
                {
                    "title": article["title"],
                    "description": article["description"],
                    "source": article["source"]["name"],
                    "date": article["publishedAt"]
                }
                for article in data["articles"][:3]
            ]
            
            return ActionResult(
                success=True,
                data=articles,
                confidence=0.7,
                source="newsapi",
                timestamp=datetime.now(),
                metadata={"query": query}
            )
        except Exception as e:
            logging.error(f"News search error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="newsapi",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _translate(self, text_and_target: str) -> ActionResult:
        """Translation service"""
        try:
            text, target_lang = text_and_target.split('|')
            translator = Translator()
            translation = translator.translate(text, dest=target_lang)
            
            return ActionResult(
                success=True,
                data={
                    "original": text,
                    "translated": translation.text,
                    "source_lang": translation.src,
                    "target_lang": target_lang
                },
                confidence=0.9,
                source="translator",
                timestamp=datetime.now(),
                metadata={"detected_language": translation.src}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="translator",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _google_search(self, query: str) -> ActionResult:
        """Google Custom Search implementation"""
        try:
            api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
            cx = os.getenv("GOOGLE_SEARCH_CX")
            
            if not api_key or not cx:
                raise ValueError("Google Search API credentials not configured")
            
            self.rate_limiter.wait()
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    'key': api_key,
                    'cx': cx,
                    'q': query,
                    'num': 3
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if 'items' not in data:
                return ActionResult(
                    success=False,
                    data="No results found",
                    confidence=0.0,
                    source="google",
                    timestamp=datetime.now(),
                    metadata={}
                )
            
            results = [
                {
                    "title": item["title"],
                    "link": item["link"],
                    "snippet": item["snippet"]
                }
                for item in data["items"][:3]
            ]
            
            return ActionResult(
                success=True,
                data=results,
                confidence=0.8,
                source="google",
                timestamp=datetime.now(),
                metadata={"query": query}
            )
        except Exception as e:
            logging.error(f"Google search error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="google",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

class IQPACEAgent:
    """
    Advanced cognitive agent implementing the IQ-PACE framework for structured reasoning.
    
    Framework Flow:
    User Query -> Intake -> Query Planning -> Perform Action -> Analyze Result -> Cross-Validate -> Evaluate -> Conclude
    """
    
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.action_manager = ActionManager()
        self.validation_engine = ValidationEngine()
        self.action_planner = ActionPlanner()
        self.conversation_history = []
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.current_context = {}
        
    def process_query(self, query: str) -> str:
        """Main processing loop implementing IQ-PACE framework with verbose output"""
        try:
            print("\n=== IQ-PACE Framework Execution ===\n")
            
            # 1. Intake
            print("🔍 INTAKE PHASE")
            print("----------------")
            print(f"Analyzing query: '{query}'")
            intake_result = self._intake_analysis(query)
            if not intake_result['success']:
                print("❌ Intake failed:", intake_result['error'])
                return f"Error during intake: {intake_result['error']}"
            print("✓ Intake Analysis Results:")
            for key, value in intake_result['analysis'].items():
                print(f"  • {key}: {value}")
            
            # 2. Query Planning
            print("\n📋 QUERY PLANNING PHASE")
            print("----------------------")
            plan = self._query_planning(intake_result)
            if not plan['success']:
                print("❌ Planning failed:", plan['error'])
                return f"Error during planning: {plan['error']}"
            print("✓ Action Plan:")
            for i, action in enumerate(plan['actions'], 1):
                print(f"  {i}. {action}")
            
            # 3. Action Execution
            print("\n⚡ ACTION EXECUTION PHASE")
            print("-----------------------")
            action_results = []
            for action in plan['actions']:
                print(f"\nExecuting: {action}")
                result = self._perform_action(action)
                action_results.append(result)
                print(f"Status: {'✓ Success' if result.success else '❌ Failed'}")
                print(f"Confidence: {result.confidence:.2f}")
                if not result.success and not plan.get('continue_on_failure', True):
                    return f"Critical action failure: {result.data}"
            
            # 4. Result Analysis
            print("\n🔎 RESULT ANALYSIS PHASE")
            print("----------------------")
            analysis = self._analyze_results(action_results)
            print(f"Aggregate Confidence: {analysis.get('aggregate_confidence', 0):.2f}")
            if analysis.get('information_gaps'):
                print("Information Gaps Detected:")
                for gap in analysis['information_gaps']:
                    print(f"  • {gap['error']}")
            
            # 5. Cross-Validation
            print("\n✔️ CROSS-VALIDATION PHASE")
            print("------------------------")
            validation = self._cross_validate(analysis)
            print(f"Validation Score: {validation.confidence_score:.2f}")
            print("Validated Claims:")
            for claim in validation.validated_claims:
                print(f"  • {claim}")
            
            # 6. Evaluation
            print("\n⚖️ EVALUATION PHASE")
            print("------------------")
            evaluation = self._evaluate(validation)
            print(f"Confidence Sufficient: {'✓' if evaluation['confidence_sufficient'] else '❌'}")
            if evaluation['recommended_actions']:
                print("Recommended Additional Actions:")
                for action in evaluation['recommended_actions']:
                    print(f"  • {action}")
            
            # 7. Conclusion
            print("\n📝 CONCLUSION PHASE")
            print("-----------------")
            conclusion = self._conclude(evaluation)
            print("\nFinal Response:")
            print("---------------")
            print(conclusion)
            
            # Update conversation history
            self._update_history(query, conclusion)
            
            return conclusion
            
        except Exception as e:
            logging.error(f"Process query error: {str(e)}")
            print(f"\n❌ ERROR: {str(e)}")
            return f"An error occurred during query processing: {str(e)}"
        finally:
            print("\n=== End of Processing ===\n")
            logging.info(f"Query processing completed: {query[:100]}...")

    def _intake_analysis(self, query: str) -> Dict:
        """
        Perform deep analysis of the input query to understand requirements and context.
        """
        try:
            prompt = f"""
            Intake Analysis Required:
            Query: {query}
            
            Please analyze:
            1. Core information needs
            2. Required confidence level
            3. Domain classification
            4. Complexity assessment
            5. Critical constraints
            
            Format: JSON with keys: information_needs, confidence_required, domain, complexity, constraints
            """
            
            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            self.current_context.update({
                'query': query,
                'intake_analysis': analysis
            })
            
            return {
                'success': True,
                'analysis': analysis
            }
        except Exception as e:
            logging.error(f"Intake analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _query_planning(self, intake_result: Dict) -> Dict:
        """
        Strategic planning of actions using A* search for optimal action sequence.
        """
        try:
            analysis = intake_result['analysis']
            
            # Define goal state based on requirements
            goal_state = {
                'confidence_level': analysis['confidence_required'],
                'information_completeness': 1.0,
                'validation_level': 'high' if analysis['confidence_required'] > 0.7 else 'moderate'
            }
            
            # Generate action sequence using A* search
            initial_state = "start"
            action_sequence = self.action_planner.plan(initial_state, json.dumps(goal_state))
            
            # Convert action sequence to structured plan
            plan = {
                'success': True,
                'actions': action_sequence,
                'continue_on_failure': analysis['complexity'] < 0.7,
                'validation_requirements': {
                    'min_sources': max(2, int(analysis['confidence_required'] * 5)),
                    'confidence_threshold': analysis['confidence_required']
                }
            }
            
            self.current_context['plan'] = plan
            return plan
            
        except Exception as e:
            logging.error(f"Query planning error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _perform_action(self, action: str) -> ActionResult:
        """Execute planned action with error handling and rate limiting"""
        try:
            action_name, action_input = action.split(':', 1)
            action_name = action_name.strip()
            action_input = action_input.strip()
            
            if action_name not in self.action_manager.actions:
                raise ValueError(f"Unknown action: {action_name}")
                
            result = self.action_manager.actions[action_name](action_input)
            
            # Log action execution
            logging.info(f"Action executed: {action_name} with input: {action_input}")
            logging.info(f"Action result: {result}")
            
            return result
            
        except Exception as e:
            logging.error(f"Action execution error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="error",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _analyze_results(self, results: List[ActionResult]) -> Dict:
        """Deep analysis of action results"""
        try:
            analysis = {
                'success': True,
                'results': [],
                'aggregate_confidence': 0.0,
                'information_gaps': [],
                'contradictions': [],
                'sources': set()
            }
            
            for result in results:
                if result.success:
                    analysis['results'].append({
                        'data': result.data,
                        'confidence': result.confidence,
                        'source': result.source,
                        'timestamp': result.timestamp.isoformat(),
                        'metadata': result.metadata
                    })
                    analysis['sources'].add(result.source)
                    
                    # Update aggregate confidence using Bayesian updating
                    if analysis['aggregate_confidence'] == 0.0:
                        analysis['aggregate_confidence'] = result.confidence
                    else:
                        analysis['aggregate_confidence'] = self._update_confidence(
                            analysis['aggregate_confidence'],
                            result.confidence
                        )
                else:
                    analysis['information_gaps'].append({
                        'source': result.source,
                        'error': result.data
                    })
            
            analysis['sources'] = list(analysis['sources'])
            return analysis
            
        except Exception as e:
            logging.error(f"Result analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _cross_validate(self, analysis: Dict) -> ValidationResult:
        """Cross-validate results using the validation engine"""
        try:
            # Extract claims from results
            claims = []
            sources = []
            
            for result in analysis['results']:
                if isinstance(result['data'], str):
                    claims.append(result['data'])
                elif isinstance(result['data'], dict):
                    claims.extend(str(v) for v in result['data'].values())
                elif isinstance(result['data'], list):
                    claims.extend(str(item) for item in result['data'])
                
                sources.append({
                    'content': str(result['data']),
                    'source': result['source']
                })
            
            return self.validation_engine.validate(claims, sources)
            
        except Exception as e:
            logging.error(f"Cross-validation error: {str(e)}")
            return ValidationResult(
                confidence_score=0.0,
                validated_claims=[],
                sources=[],
                contradictions=[str(e)],
                uncertainty_factors=["Validation failed"]
            )

    def _evaluate(self, validation: ValidationResult) -> Dict:
        """Evaluate validation results and determine next steps"""
        try:
            evaluation = {
                'success': True,
                'confidence_sufficient': False,
                'validated_claims': validation.validated_claims,
                'confidence_score': validation.confidence_score,
                'requires_additional_validation': False,
                'recommended_actions': []
            }
            
            # Check if confidence meets requirements
            required_confidence = self.current_context['intake_analysis']['confidence_required']
            evaluation['confidence_sufficient'] = validation.confidence_score >= required_confidence
            
            # Determine if additional validation is needed
            if not evaluation['confidence_sufficient']:
                evaluation['requires_additional_validation'] = True
                evaluation['recommended_actions'] = self._generate_additional_actions(validation)
            
            return evaluation
            
        except Exception as e:
            logging.error(f"Evaluation error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _conclude(self, evaluation: Dict) -> str:
        """Generate final response based on evaluation"""
        try:
            if not evaluation['success']:
                return f"Error during evaluation: {evaluation['error']}"
            
            response_parts = []
            
            # Add validated claims
            if evaluation['validated_claims']:
                response_parts.append("Validated Information:")
                for claim in evaluation['validated_claims']:
                    response_parts.append(f"- {claim}")
            
            # Add confidence information
            response_parts.append(f"\nConfidence Level: {evaluation['confidence_score']:.2f}")
            
            # Add caveats if confidence is not sufficient
            if not evaluation['confidence_sufficient']:
                response_parts.append("\nNote: Some information could not be fully validated.")
                if evaluation['recommended_actions']:
                    response_parts.append("Recommended additional verification needed.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logging.error(f"Conclusion error: {str(e)}")
            return f"Error generating conclusion: {str(e)}"

    def _generate_additional_actions(self, validation: ValidationResult) -> List[str]:
        """Generate recommended additional actions for validation"""
        actions = []
        
        if len(validation.sources) < 3:
            actions.append("google_search: " + self.current_context['query'])
        
        if validation.confidence_score < 0.5:
            actions.append("cross_validate: " + self.current_context['query'])
        
        return actions

    def _update_history(self, query: str, response: str):
        """Update conversation history"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'context': self.current_context.copy()
        })

    def _update_confidence(self, prior: float, evidence: float) -> float:
        """Bayesian confidence updating"""
        likelihood = evidence
        normalization = (likelihood * prior + (1 - likelihood) * (1 - prior))
        if normalization == 0:
            return 0.0
        return (likelihood * prior) / normalization

# Main execution function
def query(question: str, max_iterations: int = 3) -> str:
    """
    Process a question through the IQ-PACE framework.
    
    Args:
        question: The user's question
        max_iterations: Maximum number of framework iterations
        
    Returns:
        str: The final response
    """
    agent = IQPACEAgent(system_prompt)
    return agent.process_query(question)

if __name__ == "__main__":
    try:
        print("Testing IQ-PACE agent with a sample query...")
        result = query("What are the latest developments in quantum computing and their potential impact on cryptography?")
        print("\nFinal result:", result)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        logging.error(f"Execution error: {str(e)}")

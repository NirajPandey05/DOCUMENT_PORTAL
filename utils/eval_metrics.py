from typing import List, Dict, Any, Optional
import time
from functools import wraps
import os
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, HallucinationMetric, ConversationCompletenessMetric
from deepeval.models.llms.gemini_model import GeminiModel
try:
    FactualConsistencyMetric = AnswerRelevancyMetric
    ResponseCompleteness = ConversationCompletenessMetric
    AnswerRelevanceMetric = AnswerRelevancyMetric
    ContextualRelevanceMetric = ContextualRelevancyMetric
    ContextualPrecisionMetric = ContextualPrecisionMetric
except ImportError:
    FactualConsistencyMetric = None
    ResponseCompleteness = None
    AnswerRelevanceMetric = None
    ContextualRelevanceMetric = None
    ContextualPrecisionMetric = None
from deepeval.test_case import LLMTestCase

class ResponseEvaluator:
    def __init__(self, thresholds: Dict[str, float] = None):
        """Initialize evaluation metrics with customizable thresholds and Google Gemini model."""
        default_threshold = 0.7
        self.thresholds = thresholds or {
            "factual_consistency": default_threshold,
            "answer_relevance": default_threshold,
            "context_relevance": default_threshold,
            "context_precision": default_threshold,
            "response_completeness": default_threshold
        }

        # Load Google API key from environment
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

        # Initialize Gemini model for DeepEval
        gemini_model = GeminiModel(api_key=google_api_key)

        # Initialize metrics with Gemini model
        self.metrics = {
            "factual_consistency": FactualConsistencyMetric(threshold=self.thresholds["factual_consistency"], model=gemini_model),
            "response_completeness": ResponseCompleteness(threshold=self.thresholds["response_completeness"], model=gemini_model)
        }
        if AnswerRelevanceMetric:
            self.metrics["answer_relevance"] = AnswerRelevanceMetric(threshold=self.thresholds["answer_relevance"], model=gemini_model)
        if ContextualRelevanceMetric:
            self.metrics["context_relevance"] = ContextualRelevanceMetric(threshold=self.thresholds["context_relevance"], model=gemini_model)
        if ContextualPrecisionMetric:
            self.metrics["context_precision"] = ContextualPrecisionMetric(threshold=self.thresholds["context_precision"], model=gemini_model)

        # Store evaluation history
        self.evaluation_history = []
        
    def evaluate_response(self, 
                         question: str,
                         response: str,
                         context: Optional[str] = None,
                         reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate an LLM response using multiple metrics, only if required fields are present.
        Skips metrics that require missing context or expected_output.
        """
        start_time = time.time()

        # Prepare test case
        test_case = LLMTestCase(
            input=question,
            actual_output=response,
            context=[context] if context is not None else None,
            expected_output=reference
        )

        # Run evaluations, skipping metrics if required fields are missing
        results = {}
        for metric_name, metric in self.metrics.items():
            # Skip context-based metrics if context is missing
            if metric_name in ["context_relevance", "context_precision"] and not context:
                results[metric_name] = {"skipped": "No context provided; metric not applicable."}
                continue
            # Skip metrics that require expected_output if not provided
            if metric_name == "context_precision" and not reference:
                results[metric_name] = {"skipped": "No expected_output provided; metric not applicable."}
                continue
            # Skip response_completeness if not a conversational test case (no chat history, etc.)
            if metric_name == "response_completeness":
                # Only run if question/response are conversational (simple check: question looks like a conversation)
                if not (question and ("?" in question or len(question.split()) > 2)):
                    results[metric_name] = {"skipped": "Not a conversational test case; metric not applicable."}
                    continue
            try:
                metric.measure(test_case)
                result = {
                    "score": getattr(metric, "score", None),
                    "reason": getattr(metric, "reason", None)
                }
                if hasattr(metric, "passed"):
                    result["passed"] = metric.passed
                results[metric_name] = result
            except Exception as e:
                results[metric_name] = {
                    "error": str(e)
                }

        # Add performance metrics
        results["performance"] = {
            "response_time": time.time() - start_time
        }

        # Store in history
        evaluation_record = {
            "timestamp": time.time(),
            "question": question,
            "response": response,
            "context": context,
            "reference": reference,
            "metrics": results
        }
        self.evaluation_history.append(evaluation_record)

        return results
    
    def get_evaluation_summary(self, n_recent: int = None) -> Dict[str, Any]:
        """Get summary statistics of evaluations."""
        history = self.evaluation_history[-n_recent:] if n_recent else self.evaluation_history
        
        if not history:
            return {"error": "No evaluation history available"}
        
        summary = {
            "total_evaluations": len(history),
            "average_scores": {},
            "pass_rates": {},
            "average_response_time": 0.0
        }
        
        # Calculate averages for each metric
        for metric_name in self.metrics.keys():
            scores = []
            passes = []
            
            for record in history:
                metric_result = record["metrics"].get(metric_name, {})
                if "score" in metric_result:
                    scores.append(metric_result["score"])
                if "passed" in metric_result:
                    passes.append(metric_result["passed"])
            
            if scores:
                summary["average_scores"][metric_name] = sum(scores) / len(scores)
            if passes:
                summary["pass_rates"][metric_name] = sum(passes) / len(passes)
        
        # Calculate average response time
        response_times = [
            record["metrics"]["performance"]["response_time"]
            for record in history
        ]
        summary["average_response_time"] = sum(response_times) / len(response_times)
        
        return summary

def evaluation_decorator(evaluator: ResponseEvaluator):
    """Decorator to automatically evaluate LLM responses."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract question from args or kwargs
            question = kwargs.get('query') or kwargs.get('question')
            if not question and args:
                question = args[0]
            
            # Extract context if available
            context = kwargs.get('context')
            
            # Get the response from the wrapped function
            start_time = time.time()
            response = func(*args, **kwargs)
            
            # Evaluate the response
            evaluation = evaluator.evaluate_response(
                question=question,
                response=response,
                context=context
            )
            
            # Add performance metrics
            evaluation["performance"]["total_time"] = time.time() - start_time
            
            return {
                "response": response,
                "evaluation": evaluation
            }
        
        return wrapper
    return decorator

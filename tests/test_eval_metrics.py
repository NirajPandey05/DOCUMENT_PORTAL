import pytest
import os
from utils.eval_metrics import ResponseEvaluator
from src.document_chat.retrieval import ConversationalRAG

def test_response_evaluator_initialization():
    """Test that ResponseEvaluator initializes correctly."""
    evaluator = ResponseEvaluator()
    assert evaluator is not None
    assert len(evaluator.metrics) == 5  # All metrics initialized
    assert evaluator.evaluation_history == []

def test_response_evaluation():
    """Test basic response evaluation."""
    evaluator = ResponseEvaluator()
    
    # Test case
    question = "What is the capital of France?"
    response = "The capital of France is Paris."
    context = "France is a country in Europe. Its capital city is Paris."
    
    results = evaluator.evaluate_response(
        question=question,
        response=response,
        context=context
    )
    
    # Check results structure
    assert "factual_consistency" in results
    assert "answer_relevance" in results
    assert "context_relevance" in results
    assert "context_precision" in results
    assert "response_completeness" in results
    assert "performance" in results
    
    # Check history
    assert len(evaluator.evaluation_history) == 1
    assert evaluator.evaluation_history[0]["question"] == question
    assert evaluator.evaluation_history[0]["response"] == response

def test_evaluation_summary():
    """Test evaluation summary statistics."""
    evaluator = ResponseEvaluator()
    
    # Add multiple evaluations
    test_cases = [
        {
            "question": "What is Python?",
            "response": "Python is a programming language.",
            "context": "Python is a high-level programming language."
        },
        {
            "question": "What is machine learning?",
            "response": "Machine learning is a subset of AI.",
            "context": "Machine learning is a field of artificial intelligence."
        }
    ]
    
    for case in test_cases:
        evaluator.evaluate_response(**case)
    
    summary = evaluator.get_evaluation_summary()
    
    assert summary["total_evaluations"] == 2
    assert "average_scores" in summary
    assert "pass_rates" in summary
    assert "average_response_time" in summary

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping OpenAIEmbeddings test."
)
async def test_rag_with_evaluation():
    """Test RAG system with integrated evaluation."""
    rag = ConversationalRAG(session_id="test_session")

    # Set up test index
    test_docs = [
        "Python is a popular programming language.",
        "Machine learning is a subset of artificial intelligence."
    ]

    # Create temporary FAISS index
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings

    vectorstore = FAISS.from_texts(
        test_docs,
        OpenAIEmbeddings()
    )
    
    rag.retriever = vectorstore.as_retriever()
    rag._build_lcel_chain()
    
    # Test RAG with evaluation
    result = rag.invoke("What is Python?")
    
    assert "response" in result
    assert "evaluation" in result
    assert "context" in result
    
    # Check evaluation metrics
    eval_results = result["evaluation"]
    assert "factual_consistency" in eval_results
    assert "context_relevance" in eval_results
    assert "performance" in eval_results

def test_custom_thresholds():
    """Test ResponseEvaluator with custom thresholds."""
    custom_thresholds = {
        "factual_consistency": 0.8,
        "answer_relevance": 0.85,
        "context_relevance": 0.75,
        "context_precision": 0.8,
        "response_completeness": 0.9
    }
    
    evaluator = ResponseEvaluator(thresholds=custom_thresholds)
    
    # Verify thresholds were set
    for metric_name, threshold in custom_thresholds.items():
        assert evaluator.thresholds[metric_name] == threshold
        assert evaluator.metrics[metric_name].threshold == threshold

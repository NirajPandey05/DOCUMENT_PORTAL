import pytest
import pandas as pd
import os
import json
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from utils.eval_metrics import ResponseEvaluator
from utils.document_ops import load_documents
from utils.model_loader import ModelLoader
from src.document_chat.retrieval import ConversationalRAG
from src.document_ingestion.data_ingestion import FaissManager
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM

class TestDocumentPortalComprehensive:
    @pytest.fixture(scope="class")
    def setup_test_environment(self):
        """Set up test environment with necessary components"""
        # Create test documents directory
        os.makedirs("test_data", exist_ok=True)
        
        # Create test files
        test_files = {
            "test.txt": "This is a test document about AI and machine learning.",
            "test.pdf": b"%PDF-1.4\n...test PDF content...",
            "test.csv": "id,name,value\n1,test,100",
            "test.xlsx": b"PK\x03\x04\x14\x00\x00\x00\x08\x00...test xlsx content..."
        }
        
        for filename, content in test_files.items():
            with open(f"test_data/{filename}", 'wb' if isinstance(content, bytes) else 'w') as f:
                f.write(content)
        
        return "test_data"

    @pytest.fixture
    def model_loader(self):
        return ModelLoader()


    @pytest.fixture
    def document_path(self, setup_test_environment):
        return setup_test_environment

    @pytest.fixture
    def faiss_manager(self):
        return FaissManager(index_path="test_data/faiss_index")

    def test_1_document_loading_multiple_formats(self, document_path):
        """Test 1: Verify support for multiple document formats"""
        supported_files = [
            f"{document_path}/test.txt",
            f"{document_path}/test.pdf",
            f"{document_path}/test.csv",
            f"{document_path}/test.xlsx"
        ]
        docs = load_documents([os.path.abspath(f) for f in supported_files])
        assert docs is not None
        assert len(docs) > 0
        for doc in docs:
            assert hasattr(doc, 'page_content')

    def test_2_vector_store_operations(self, faiss_manager, document_path):
        """Test 2: Verify vector store operations"""
        test_doc_path = os.path.abspath(f"{document_path}/test.txt")
        test_docs = load_documents([test_doc_path])
        faiss_manager.add_documents(test_docs)
        results = faiss_manager.similarity_search("AI and machine learning", k=1)
        assert len(results) == 1
        assert "AI" in results[0].page_content
        faiss_manager.save_index()
        assert os.path.exists("test_data/faiss_index")

    def test_3_rag_chain_functionality(self, model_loader):
        """Test 3: Verify RAG chain functionality"""
        rag = ConversationalRAG(session_id="test_session")
        
        # Load test index
        rag.load_retriever_from_faiss(
            index_path="test_data/faiss_index",
            k=1
        )
        
        # Test question answering
        result = rag.invoke("What is this document about?")
        assert "response" in result
        assert "evaluation" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_4_response_evaluation_metrics(self):
        """Test 4: Verify response evaluation metrics"""
        evaluator = ResponseEvaluator()
        
        test_cases = [
            {
                "question": "What is AI?",
                "response": "AI is artificial intelligence, a field of computer science.",
                "context": "AI (artificial intelligence) is a branch of computer science focused on creating intelligent machines.",
                "reference": "Artificial Intelligence is a field of computer science."
            }
        ]
        
        for case in test_cases:
            results = evaluator.evaluate_response(**case)
            
            # Check all required metrics are present
            assert "factual_consistency" in results
            assert "answer_relevance" in results
            assert "context_relevance" in results
            assert "response_completeness" in results
            assert "performance" in results
            
            # Check metric scores are within valid range
            for metric in results.values():
                if isinstance(metric, dict) and "score" in metric:
                    assert 0 <= metric["score"] <= 1

    def test_5_document_analysis(self, document_path):
        """Test 5: Verify document analysis capabilities"""
        analyzer = DocumentAnalyzer()
        test_doc_path = os.path.abspath(f"{document_path}/test.txt")
        test_docs = load_documents([test_doc_path])
        if not test_docs:
            pytest.skip("No test documents loaded for analysis.")
        analysis = analyzer.analyze_document(test_docs[0])
        assert "word_count" in analysis
        assert "key_phrases" in analysis
        assert "summary" in analysis
        assert analysis["word_count"] > 0

    def test_6_document_comparison(self, document_path):
        """Test 6: Verify document comparison functionality"""
        comparator = DocumentComparatorLLM()
        doc1 = "This is the first test document about AI."
        doc2 = "This is another document discussing artificial intelligence."
        with open(f"{document_path}/compare1.txt", 'w') as f:
            f.write(doc1)
        with open(f"{document_path}/compare2.txt", 'w') as f:
            f.write(doc2)
        doc1_path = os.path.abspath(f"{document_path}/compare1.txt")
        doc2_path = os.path.abspath(f"{document_path}/compare2.txt")
        docs = load_documents([doc1_path, doc2_path])
        if not docs:
            pytest.skip("No documents loaded for comparison.")
        combined_docs = "\n\n".join([d.page_content for d in docs])
        comparison_result = comparator.compare_documents(combined_docs)
        assert hasattr(comparison_result, 'to_dict') or isinstance(comparison_result, dict)

    def test_7_model_loading_and_config(self, model_loader):
        """Test 7: Verify model loading and configuration"""
        # Test LLM loading
        llm = model_loader.load_llm()
        assert llm is not None
        
        # Test embedding model loading
        embeddings = model_loader.load_embeddings()
        assert embeddings is not None
        
        # Test model configuration
        config = model_loader.get_model_config()
        assert "model_name" in config
        assert "temperature" in config

    def test_8_error_handling(self, document_processor):
        """Test 8: Verify error handling"""
        # Test non-existent file
        with pytest.raises(Exception):
            document_processor.process_document("non_existent_file.txt")
        
        # Test unsupported file format
        with pytest.raises(Exception):
            document_processor.process_document("test.unknown")
        
        # Test invalid vector store operations
        invalid_faiss = FaissManager(index_path="invalid_path")
        with pytest.raises(Exception):
            invalid_faiss.load_index()

    def test_9_concurrent_operations(self, faiss_manager, document_processor):
        """Test 9: Verify concurrent operations handling"""
        import concurrent.futures
        
        def concurrent_search(query: str) -> List[str]:
            return faiss_manager.similarity_search(query, k=1)
        
        # Create multiple search queries
        queries = ["AI", "machine learning", "data science", "neural networks"]
        
        # Execute searches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {executor.submit(concurrent_search, query): query 
                             for query in queries}
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    assert len(results) > 0
                except Exception as e:
                    assert False, f"Concurrent operation failed for query {query}: {str(e)}"

    def test_10_performance_benchmarks(self, document_processor, faiss_manager):
        """Test 10: Verify performance benchmarks"""
        import time
        
        # Test document processing speed
        start_time = time.time()
        doc = document_processor.process_document("test_data/test.txt")
        processing_time = time.time() - start_time
        assert processing_time < 5.0  # Should process within 5 seconds
        
        # Test vector search speed
        start_time = time.time()
        results = faiss_manager.similarity_search("quick test query", k=5)
        search_time = time.time() - start_time
        assert search_time < 1.0  # Should search within 1 second
        
        # Test batch operations speed
        docs = [doc] * 10  # Create batch of 10 documents
        start_time = time.time()
        faiss_manager.add_documents(docs)
        batch_time = time.time() - start_time
        assert batch_time < 10.0  # Should handle batch within 10 seconds

    @classmethod
    def teardown_class(cls):
        """Clean up test data after all tests complete"""
        import shutil
        shutil.rmtree("test_data", ignore_errors=True)

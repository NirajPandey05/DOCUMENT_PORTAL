import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from utils.eval_metrics import ResponseEvaluator, evaluation_decorator
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None, enable_cache: bool = True):
        try:
            self.session_id = session_id

            # Load LLM and prompts once
            self.model_loader = ModelLoader(enable_cache=enable_cache)
            self.llm = self.model_loader.load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Initialize evaluator
            self.evaluator = ResponseEvaluator()

            # Cache for retrieved contexts
            self._context_cache = {}

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", 
                    session_id=self.session_id,
                    cache_enabled=enable_cache,
                    cache_info=self.model_loader.cache_info)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok if you trust the index
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """
        Invoke the LCEL pipeline with evaluation metrics.
        Returns both the answer and evaluation results.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}

            # Get context from retriever
            question = self.contextualize_prompt.invoke({
                "input": user_input,
                "chat_history": chat_history
            }).to_string()
            retrieved_docs = self.retriever.invoke(question)
            context = self._format_docs(retrieved_docs)

            # Generate answer
            answer = self.chain.invoke(payload)

            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return {"response": "no answer generated.", "evaluation": None, "context": context}

            # Evaluate the response
            evaluation_results = self.evaluator.evaluate_response(
                question=user_input,
                response=answer,
                context="\n\n".join([c["summary"] for c in context])
            )

            log.info(
                "Chain invoked successfully with evaluation",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
                evaluation_summary=evaluation_results
            )

            return {
                "response": answer,
                "evaluation": evaluation_results,
                "context": context
            }
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> list:
        # Return a list of dicts with summary, original, and metadata for each doc
        formatted = []
        for d in docs:
            summary = getattr(d, "page_content", str(d))
            meta = getattr(d, "metadata", {})
            original = meta.get("original", summary)
            formatted.append({
                "summary": summary,
                "original": original,
                "metadata": meta
            })
        return formatted
    
    def _get_cached_context(self, question: str) -> Optional[str]:
        """Get cached context for a question if available."""
        return self._context_cache.get(question)
    
    def _cache_context(self, question: str, context):
        """Cache context for a question (list of dicts)."""
        self._context_cache[question] = context
        # Implement basic cache size management
        if len(self._context_cache) > 1000:  # Limit cache size
            # Remove oldest entries
            oldest_keys = sorted(self._context_cache.keys())[:100]
            for key in oldest_keys:
                del self._context_cache[key]

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question with caching
            def cached_retriever(question: str):
                cached = self._get_cached_context(question)
                if cached is not None:
                    log.info("Context cache hit", question=question[:100])
                    return cached
                docs = self.retriever.get_relevant_documents(question)
                context = self._format_docs(docs)
                self._cache_context(question, context)
                return context

            retrieve_docs = question_rewriter | cached_retriever

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)

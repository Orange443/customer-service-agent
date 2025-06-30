from langchain.chains import RetrievalQA
from ..database.vector_store import VectorStoreManager
from ..llm.providers import LLMProvider
from .prompts import PromptTemplates
from ..utils.config import config
from typing import Dict, List, Any

class RAGAgent:
    """Main RAG agent for customer support."""
    
    def __init__(self):
        self.config = config
        self.config.validate()
        
        if self.config.pgvector_connection_string is None:
            raise ValueError("pgvector_connection_string must be provided and cannot be None.")
        self.vector_store_manager = VectorStoreManager(
            connection_string=self.config.pgvector_connection_string,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model
        )
        
        self.llm = self._setup_llm()
        self.qa_chain = self._setup_retrieval_chain()
    
    def _setup_llm(self):
        """Initialize the language model."""
        provider = self.config.get_preferred_llm()
        
        if provider == "groq":
            if self.config.groq_api_key is None:
                raise ValueError("Groq API key must be provided and cannot be None.")
            return LLMProvider.create_groq_llm(
                api_key=self.config.groq_api_key,
                temperature=self.config.temperature
            )
        elif provider == "google":
            if self.config.google_api_key is None:
                raise ValueError("Google API key must be provided and cannot be None.")
            return LLMProvider.create_google_llm(
                api_key=self.config.google_api_key,
                temperature=self.config.temperature
            )
    
    def _setup_retrieval_chain(self):
        """Create the RAG chain."""
        prompt = PromptTemplates.get_customer_support_prompt()
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.get_retriever(
                search_kwargs={"k": self.config.search_k}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def _is_relevant_context(self, question: str, docs: List[Any]) -> bool:
        """Check if retrieved documents are relevant to the question."""
        question_lower = question.lower()
        
        support_keywords = [
            'password', 'reset', 'login', 'account', 'locked', 'access', 
            'billing', 'subscription', 'cancel', 'refund', 'support',
            'error', 'bug', 'issue', 'problem', 'help', 'dashboard',
            'profile', 'settings', 'notification', 'email', 'payment',
            'upgrade', 'downgrade', 'delete', 'update', 'install',
            'configuration', 'setup', 'connection', 'sync', 'backup'
        ]
        
        has_support_keywords = any(keyword in question_lower for keyword in support_keywords)
        
        if docs:
            doc_content = " ".join([doc.page_content.lower() for doc in docs])
            content_relevance = any(keyword in doc_content for keyword in support_keywords)
            question_words = question_lower.split()
            word_overlap = any(word in doc_content for word in question_words if len(word) > 3)
            
            return has_support_keywords and (content_relevance or word_overlap)
        
        return has_support_keywords
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer from the RAG system."""
        try:
            # Get relevant documents first
            docs = self.vector_store_manager.similarity_search(question, k=self.config.search_k)
            
            # Check if documents are relevant
            if not docs or not self._is_relevant_context(question, docs):
                return {
                    'answer': "I don't have information about this in our support database. I have forwarded your query to our team and will provide a response soon. Thank you for your patience!",
                    'sources': [],
                    'is_fallback': True
                }
            
            # Proceed with RAG if context is relevant
            result = self.qa_chain.invoke({"query": question})
            
            return {
                'answer': result['result'],
                'sources': result.get('source_documents', []),
                'is_fallback': False
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error processing your question: {str(e)}. Please try again or contact our support team directly.",
                'sources': [],
                'is_fallback': True,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'llm_provider': self.config.get_preferred_llm(),
            'embedding_model': self.config.embedding_model,
            'collection_name': self.config.collection_name,
            'vector_store_stats': self.vector_store_manager.get_stats()
        }

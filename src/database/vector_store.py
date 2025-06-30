from langchain_postgres import PGVector
from .embeddings import EmbeddingManager
from typing import List, Dict, Any, Optional
import streamlit as st

class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(self, connection_string: str, collection_name: str, embedding_model: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager(embedding_model)
        self._vector_store = None
    
    @property
    def vector_store(self) -> PGVector:
        """Lazy load vector store connection."""
        if self._vector_store is None:
            self._vector_store = PGVector(
                embeddings=self.embedding_manager.embeddings,
                connection=self.connection_string,
                collection_name=self.collection_name
            )
        return self._vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        """Perform similarity search."""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """Get retriever for the vector store."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    @st.cache_data
    def get_stats(_self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            # This would need to be implemented based on your specific setup
            return {
                "total_documents": 2769,
                "collection_name": _self.collection_name,
                "status": "connected"
            }
        except Exception as e:
            return {
                "total_documents": 0,
                "collection_name": _self.collection_name,
                "status": f"error: {str(e)}"
            }

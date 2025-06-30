import os
from dotenv import load_dotenv
from typing import Optional

class Config:
    """Configuration management for the RAG application."""
    
    def __init__(self):
        load_dotenv()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        # API Keys
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Database
        self.pgvector_connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")
        
        # Model Settings
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.collection_name = os.getenv("COLLECTION_NAME", "support_tickets")
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        
        # Search Settings
        self.search_k = int(os.getenv("SEARCH_K", "5"))
        
        # UI Settings
        self.app_title = os.getenv("APP_TITLE", "🤖 Customer Support RAG Assistant")
        self.page_icon = os.getenv("PAGE_ICON", "🤖")
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.pgvector_connection_string:
            raise ValueError("PGVECTOR_CONNECTION_STRING is required")
        
        if not any([self.google_api_key, self.groq_api_key]):
            raise ValueError("At least one API key (GOOGLE_API_KEY or GROQ_API_KEY) is required")
        
        return True
    
    def get_preferred_llm(self) -> str:
        """Get the preferred LLM provider."""
        if self.groq_api_key:
            return "groq"
        elif self.google_api_key:
            return "google"
        else:
            raise ValueError("No valid API key found")

# Global config instance
config = Config()

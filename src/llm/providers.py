from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Union
from pydantic import SecretStr

class LLMProvider:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_groq_llm(api_key: str, model_name: str = "meta-llama/llama-4-maverick-17b-128e-instruct", temperature: float = 0.3) -> ChatGroq:
        """Create Groq LLM instance."""
        return ChatGroq(
            api_key=SecretStr(api_key),
            model=model_name,
            temperature=0.3
        )
    
    @staticmethod
    def create_google_llm(api_key: str, model: str = "gemini-2.5-flash-lite-preview-06-17", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
        """Create Google LLM instance."""
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature
        )
    
    @staticmethod
    def create_llm(provider: str, api_key: str, **kwargs) -> Union[ChatGroq, ChatGoogleGenerativeAI]:
        """Create LLM based on provider."""
        if provider == "groq":
            return LLMProvider.create_groq_llm(api_key, **kwargs)
        elif provider == "google":
            return LLMProvider.create_google_llm(api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

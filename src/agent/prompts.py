from langchain.prompts import PromptTemplate

class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    CUSTOMER_SUPPORT_TEMPLATE = """
You are a helpful customer support assistant. Use ONLY the following context from resolved support tickets to answer the user's question.

Context from resolved tickets:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
- Use ONLY information from the provided context above
- End each factual statement with [Source: Ticket #X] where X corresponds to the document number in the context
- If the context doesn't contain relevant information about the question, respond exactly: "I don't have information about this in our support database. I have forwarded your query to our team and will provide a response soon. Thank you for your patience!"
- Be professional, empathetic, and helpful
- Provide step-by-step solutions when available in the context
- Never generate information not present in the context

Answer with proper citations:
"""
    
    @classmethod
    def get_customer_support_prompt(cls) -> PromptTemplate:
        """Get customer support prompt template."""
        return PromptTemplate(
            template=cls.CUSTOMER_SUPPORT_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    @classmethod
    def get_custom_prompt(cls, template: str, input_variables: list) -> PromptTemplate:
        """Create custom prompt template."""
        return PromptTemplate(
            template=template,
            input_variables=input_variables
        )

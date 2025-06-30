import streamlit as st
from typing import Dict, List, Any

class UIComponents:
    """Reusable UI components for the Streamlit app."""
    
    @staticmethod
    def render_chat_message(role: str, content: str, is_fallback: bool = False):
        """Render a chat message."""
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
        else:
            message_class = "fallback-message" if is_fallback else "assistant-message"
            st.markdown(f'<div class="chat-message {message_class}"><strong>Assistant:</strong> {content}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_sources(sources: List[Any], max_sources: int = 3):
        """Render source documents."""
        if sources:
            with st.expander(f"📚 View Sources ({len(sources)} tickets)", expanded=False):
                for i, source in enumerate(sources[:max_sources], 1):
                    preview = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                    st.markdown(f'<div class="source-box"><strong>Ticket #{i}:</strong><br>{preview}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_stats(stats: Dict[str, Any]):
        """Render statistics in sidebar."""
        st.sidebar.markdown("### 📊 System Stats")
        
        vector_stats = stats.get('vector_store_stats', {})
        st.sidebar.metric("Support Tickets", vector_stats.get('total_documents', 'N/A'))
        st.sidebar.metric("Collection", stats.get('collection_name', 'N/A'))
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 AI Model")
        st.sidebar.info(f"Provider: {stats.get('llm_provider', 'Unknown').title()}")
        st.sidebar.info(f"Embeddings: {stats.get('embedding_model', 'Unknown').split('/')[-1]}")
    

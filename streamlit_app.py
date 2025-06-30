import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.rag_agent import RAGAgent
from src.ui.components import UIComponents
from src.ui.styles import CUSTOM_CSS
from src.utils.config import config

# Page configuration
st.set_page_config(
    page_title=config.app_title,
    page_icon=config.page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None

def initialize_agent():
    """Initialize the RAG agent."""
    if st.session_state.agent is None:
        with st.sidebar:
            with st.spinner("🔄 Connecting to knowledge base..."):
                try:
                    st.session_state.agent = RAGAgent()
                    st.success("✅ Connected successfully!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    st.stop()

def main():
    """Main Streamlit application."""
    # Header
    st.markdown(f'<h1 class="main-header">{config.app_title}</h1>', unsafe_allow_html=True)
    
    # Initialize agent
    initialize_agent()
    
    # Sidebar
    stats = st.session_state.agent.get_stats()
    UIComponents.render_sidebar_stats(stats)
    
    st.sidebar.markdown("---")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Main chat interface
    st.markdown("### 💬 Ask your customer support question:")
    
    # Display chat messages
    for message in st.session_state.messages:
        UIComponents.render_chat_message(
            role=message["role"],
            content=message["content"],
            is_fallback=message.get("is_fallback", False)
        )
        
        # Display sources if available
        if message.get("sources"):
            UIComponents.render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        UIComponents.render_chat_message("user", prompt)
        
        # Get assistant response
        with st.spinner("🔍 Searching knowledge base..."):
            response = st.session_state.agent.ask(prompt)
            
            # Add assistant message
            assistant_message = {
                "role": "assistant",
                "content": response['answer'],
                "sources": response['sources'],
                "is_fallback": response['is_fallback']
            }
            st.session_state.messages.append(assistant_message)
            
            # Display response
            UIComponents.render_chat_message(
                role="assistant",
                content=response['answer'],
                is_fallback=response['is_fallback']
            )
            
            # Display sources
            if response['sources']:
                UIComponents.render_sources(response['sources'])

if __name__ == "__main__":
    main()

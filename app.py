"""
Streamlit web interface for Turkish RAG Chatbot.
Provides a clean UI for interacting with the RAG pipeline.
"""

import streamlit as st
import os
from typing import Dict, List, Any
import time
from dotenv import load_dotenv

# Import our custom modules
from data_loader import SuperLigDataLoader, create_document_store
from rag_pipeline import RAGPipelineManager

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Türkçe RAG Chatbot",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #333333;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #333333;
    }
    .context-doc {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #ff9800;
        color: #333333;
    }
    .score-badge {
        background-color: #4caf50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """
    Initialize the RAG system with caching.
    
    Returns:
        RAGPipelineManager instance
    """
    with st.spinner("RAG sistemi yükleniyor..."):
        try:
            # Load data
            loader = SuperLigDataLoader()
            documents = loader.load_and_prepare_data()
            
            # Create document store
            doc_store = create_document_store(documents)
            
            # Initialize RAG pipeline
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("GOOGLE_API_KEY environment variable is not set!")
                return None
            
            rag_manager = RAGPipelineManager(doc_store, api_key)
            return rag_manager
            
        except Exception as e:
            st.error(f"RAG sistemi başlatılamadı: {str(e)}")
            return None


def display_chat_message(message: str, is_user: bool = True):
    """
    Display a chat message with appropriate styling.
    
    Args:
        message: Message text
        is_user: Whether this is a user message
    """
    css_class = "user-message" if is_user else "bot-message"
    st.markdown(f'<div class="chat-message {css_class}">{message}</div>', 
                unsafe_allow_html=True)


def display_context_documents(documents: List[Dict[str, Any]]):
    """
    Display retrieved context documents.
    
    Args:
        documents: List of document dictionaries
    """
    if not documents:
        return
    
    st.subheader("📚 Bulunan Kaynak Belgeler")
    
    for i, doc in enumerate(documents, 1):
        with st.expander(f"Belge {i}: {doc.get('title', 'Başlıksız')} (Skor: {doc.get('score', 0):.3f})"):
            st.markdown(f"**İçerik:** {doc.get('content', '')}")
            if doc.get('url'):
                st.markdown(f"**Kaynak:** {doc['url']}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Türkçe RAG Chatbot - Süper Lig Asistanı</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Bu chatbot, Türkçe Wikipedia verilerini kullanarak Süper Lig hakkında sorularınızı yanıtlar.
    Haystack RAG pipeline ve Google Gemini 2.0 Flash modeli kullanılmaktadır.
    """)
    
    # Load API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Show API key status
        if api_key:
            st.success("✅ Google API Key yüklendi")
        else:
            st.error("❌ GOOGLE_API_KEY bulunamadı")
            st.markdown("""
            **API Key ayarlamak için:**
            1. `.env` dosyası oluşturun
            2. `GOOGLE_API_KEY=your_key_here` ekleyin
            3. Uygulamayı yeniden başlatın
            """)
        
        # Retrieval settings
        st.subheader("🔍 Arama Ayarları")
        top_k = st.slider(
            "Maksimum belge sayısı",
            min_value=1,
            max_value=10,
            value=8,
            help="Aranacak maksimum belge sayısı"
        )
        
        # Show context option
        show_context = st.checkbox(
            "Kaynak belgeleri göster",
            value=True,
            help="Bulunan kaynak belgeleri göster"
        )
        
        # Clear chat button
        if st.button("🗑️ Sohbeti Temizle"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize RAG system
    if not api_key:
        st.error("❌ Google API anahtarı bulunamadı!")
        st.markdown("""
        **API Key ayarlamak için:**
        
        1. **Google AI Studio'dan API Key alın**: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
        
        2. **Proje klasöründe `.env` dosyası oluşturun**:
           ```bash
           echo GOOGLE_API_KEY=your_actual_api_key_here > .env
           ```
        
        3. **Uygulamayı yeniden başlatın**:
           ```bash
           streamlit run app.py
           ```
        """)
        
        st.stop()
    
    # Initialize RAG system
    if 'rag_manager' not in st.session_state:
        with st.spinner("RAG sistemi başlatılıyor..."):
            try:
                # Load data
                loader = SuperLigDataLoader()
                documents = loader.load_and_prepare_data()
                
                # Create document store
                doc_store = create_document_store(documents)
                
                # Initialize RAG pipeline
                st.session_state.rag_manager = RAGPipelineManager(doc_store, api_key)
                st.success("✅ RAG sistemi başarıyla yüklendi!")
                
            except Exception as e:
                st.error(f"❌ RAG sistemi başlatılamadı: {str(e)}")
                st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message['content'], message['role'] == 'user')
        
        # Show context if available
        if message['role'] == 'assistant' and 'context' in message and show_context:
            display_context_documents(message['context'])
    
    # Chat input
    st.subheader("💬 Soru Sorun")
    
    # Example questions
    st.markdown("**Örnek sorular:**")
    example_questions = [
        "Galatasaray hakkında bilgi ver",
        "Süper Lig'de hangi takımlar var?",
        "Fenerbahçe'nin renkleri nelerdir?",
        "Trabzonspor ne zaman kuruldu?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"example_{i}"):
                st.session_state.user_input = question
    
    # Text input
    user_input = st.text_input(
        "Sorunuzu yazın:",
        value=st.session_state.get('user_input', ''),
        placeholder="Örn: Galatasaray hakkında bilgi ver...",
        key="question_input"
    )
    
    # Clear the user input from session state after using it
    if 'user_input' in st.session_state:
        del st.session_state.user_input
    
    # Get Answer button
    if st.button("🔍 Yanıt Al", type="primary") or user_input:
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input
            })
            
            # Display user message
            display_chat_message(user_input, is_user=True)
            
            # Get answer from RAG system
            with st.spinner("🤔 Düşünüyorum..."):
                try:
                    result = st.session_state.rag_manager.ask_question(
                        user_input, 
                        show_context=show_context
                    )
                    
                    # Add assistant response to chat
                    response_data = {
                        'role': 'assistant',
                        'content': result['answer']
                    }
                    
                    if show_context and 'documents' in result:
                        response_data['context'] = result['documents']
                    
                    st.session_state.messages.append(response_data)
                    
                    # Display assistant response
                    display_chat_message(result['answer'], is_user=False)
                    
                    # Display context documents
                    if show_context and 'documents' in result:
                        display_context_documents(result['documents'])
                    
                except Exception as e:
                    error_msg = f"Üzgünüm, bir hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': error_msg
                    })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚽ Türkçe RAG Chatbot | Haystack + Google Gemini 2.0 Flash | Süper Lig Asistanı</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
LawLM - Streamlit Frontend (No Authentication - Testing)
Connects to FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="LawLM - Indian Law QA",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""

def main_app():
    """Main application"""
    
    # Header
    st.title("⚖️ LawLM - Indian Law Question Answering")
    st.markdown("*Powered by RAG + Llama 3.1 (Open-Source)*")
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Ask Question", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Ask a Legal Question")
        
        # Settings in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            law_type = st.selectbox(
                "Filter by Law Type",
                ["All", "IPC", "CrPC", "Constitution"]
            )
        with col2:
            top_k = st.slider("Sources to retrieve", 1, 10, 5)
        with col3:
            use_llm = st.checkbox("Use LLM Generation", value=True, 
                                 help="Use Llama 3.1 to generate answers")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            examples = [
                "What is the punishment for murder under IPC?",
                "What does Section 302 IPC deal with?",
                "What are the types of punishments in IPC?",
                "What is meant by 'commutation of sentence'?",
                "What are the fundamental rights in the Constitution?"
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex[:20]}"):
                    st.session_state.query = ex
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.query,
            height=100,
            placeholder="e.g., What is the punishment for theft under IPC?"
        )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Searching legal database..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/query",
                            json={
                                "query": query,
                                "top_k": top_k,
                                "law_type": None if law_type == "All" else law_type,
                                "use_llm": use_llm
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display answer
                            st.success(f"✓ Answer generated in {result['response_time_ms']}ms")
                            
                            st.subheader("📋 Answer")
                            st.info(result['answer'])
                            
                            st.caption(f"Model used: {result['model_used']}")
                            
                            # Display sources
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(
                                    f"Source {i}: {source['law_type']}" + 
                                    (f" - Section {source['section_number']}" if source['section_number'] else "") +
                                    f" (Relevance: {source['similarity']:.1%})"
                                ):
                                    st.markdown(f"**Q:** {source['question']}")
                                    st.markdown(f"**A:** {source['answer']}")
                        
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to the backend server. Please ensure the FastAPI server is running on http://localhost:8000")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("📊 System Analytics")
        
        try:
            response = requests.get(f"{API_URL}/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Queries", stats['total_queries'])
                with col2:
                    st.metric("Avg Response Time", f"{stats['avg_response_time_ms']:.0f}ms")
                with col3:
                    st.metric("Model", stats['model_info']['generation_model'].split(':')[0])
                
                # Model info
                st.subheader("🤖 Model Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Embedding Model**\n{stats['model_info']['embedding_model']}")
                with col2:
                    st.info(f"**Generation Model**\n{stats['model_info']['generation_model']}")
                with col3:
                    st.info(f"**Embedding Dimension**\n{stats['model_info']['embedding_dimension']}")
            else:
                st.error("Unable to load statistics")
        
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    with tab3:
        st.header("About LawLM")
        
        st.markdown("""
        ### 🎯 What is LawLM?
        
        LawLM is an AI-powered legal question-answering system specialized in Indian law, 
        using **100% open-source technology**.
        
        ### 🏗️ Architecture
        
        - **Backend**: FastAPI
        - **Database**: PostgreSQL + pgvector
        - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
        - **LLM**: Llama 3.1 8B via Ollama (runs locally!)
        - **Frontend**: Streamlit
        
        ### 📚 Data Sources
        
        - **IPC** (Indian Penal Code)
        - **CrPC** (Criminal Procedure Code)
        - **Constitution** of India
        
        ### ✨ Key Features
        
        ✅ Open-source LLM (no API costs!)  
        ✅ Runs completely locally  
        ✅ Vector similarity search  
        ✅ RESTful API  
        ✅ Real-time analytics  
        
        ### ⚠️ Disclaimer
        
        This system is for educational purposes only. Not a substitute for professional 
        legal advice. Always consult a qualified lawyer.
        
        ### 🔧 Technology Stack
        
        **Backend:**
        - FastAPI, PostgreSQL, pgvector, Ollama
        
        **ML/AI:**
        - Sentence Transformers, Llama 3.1, RAG Architecture
        
        **Frontend:**
        - Streamlit, Plotly
        
        ---
        
        **Built with ❤️ using 100% open-source tools**
        """)
        
        # Health check
        st.subheader("🏥 System Health")
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                health = response.json()
                col1, col2, col3 = st.columns(3)
                with col1:
                    status = "🟢" if health['database'] == 'healthy' else "🔴"
                    st.metric("Database", f"{status} {health['database']}")
                with col2:
                    status = "🟢" if health['ollama'] == 'healthy' else "🔴"
                    st.metric("Ollama", f"{status} {health['ollama']}")
                with col3:
                    st.metric("Model", health['model'])
        except Exception as e:
            st.error(f"Could not check health: {str(e)}")

if __name__ == "__main__":
    main_app()

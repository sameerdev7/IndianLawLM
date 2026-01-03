"""
LawLM - Streamlit Frontend with Authentication
Connects to FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="LawLM - Indian Law QA",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None

# Helper functions
def make_authenticated_request(endpoint, method="GET", data=None):
    """Make authenticated API request"""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 401:
            st.session_state.token = None
            st.session_state.user_email = None
            st.error("Session expired. Please login again.")
            st.rerun()
        
        return response
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Make sure FastAPI is running on port 8000.")
        return None

def login_page():
    """Login/Register page"""
    st.title("⚖️ LawLM - Indian Law Question Answering")
    st.markdown("*Powered by RAG + Open-Source LLM (Llama 3.1)*")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if email and password:
                response = requests.post(
                    f"{API_BASE_URL}/auth/login",
                    json={"email": email, "password": password}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user_email = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Login failed"))
            else:
                st.warning("Please enter email and password")
    
    with tab2:
        st.subheader("Create New Account")
        
        full_name = st.text_input("Full Name", key="reg_name")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")
        password2 = st.text_input("Confirm Password", type="password", key="reg_password2")
        
        if st.button("Register", type="primary", use_container_width=True):
            if not all([full_name, email, password, password2]):
                st.warning("Please fill all fields")
            elif password != password2:
                st.error("Passwords don't match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                response = requests.post(
                    f"{API_BASE_URL}/auth/register",
                    json={
                        "email": email,
                        "password": password,
                        "full_name": full_name
                    }
                )
                
                if response.status_code == 201:
                    data = response.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user_email = email
                    st.success("Registration successful!")
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Registration failed"))

def main_app():
    """Main application after login"""
    
    # Header with logout
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("⚖️ LawLM - Indian Law QA")
    with col2:
        st.write(f"👤 {st.session_state.user_email}")
    with col3:
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.user_email = None
            st.rerun()
    
    st.markdown("*Powered by RAG + Llama 3.1 (Open-Source)*")
    
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
                    st.session_state['query'] = ex
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.get('query', ''),
            height=100,
            placeholder="e.g., What is the punishment for theft under IPC?"
        )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Searching legal database..."):
                    response = make_authenticated_request(
                        "/query",
                        method="POST",
                        data={
                            "query": query,
                            "top_k": top_k,
                            "law_type": None if law_type == "All" else law_type,
                            "use_llm": use_llm
                        }
                    )
                
                if response and response.status_code == 200:
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
                    
                elif response:
                    st.error(response.json().get("detail", "Error processing query"))
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("📊 System Analytics")
        
        # Get stats
        response = make_authenticated_request("/stats")
        
        if response and response.status_code == 200:
            stats = response.json()
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", stats['total_queries'])
            with col2:
                st.metric("Active Users", stats['total_users'])
            with col3:
                st.metric("Avg Response Time", f"{stats['avg_response_time_ms']:.0f}ms")
            with col4:
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
    
    with tab3:
        st.header("About LawLM")
        
        st.markdown("""
        ### 🎯 What is LawLM?
        
        LawLM is an AI-powered legal question-answering system specialized in Indian law, 
        using **100% open-source technology**.
        
        ### 🏗️ Architecture
        
        - **Backend**: FastAPI with JWT authentication
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
        ✅ JWT authentication  
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
        response = make_authenticated_request("/health")
        if response and response.status_code == 200:
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

# Main app logic
def main():
    if st.session_state.token is None:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()

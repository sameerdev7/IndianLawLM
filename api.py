"""
LawLM - FastAPI Backend (No Authentication - Testing)
RESTful API with Ollama (Llama 3.1) integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from sentence_transformers import SentenceTransformer
import ollama
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
OLLAMA_MODEL = "llama3.2:3b"  # You can change to mistral, phi3, etc.

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'jameerbaba7'),
    'database': os.getenv('DB_NAME', 'lawlm')
}

# Initialize FastAPI
app = FastAPI(
    title="LawLM API",
    description="Indian Law Question Answering System with RAG + Open-Source LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Embedding model loaded")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    law_type: Optional[str] = None
    use_llm: Optional[bool] = True

class Source(BaseModel):
    law_id: int
    law_type: str
    question: str
    answer: str
    section_number: Optional[str]
    similarity: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    response_time_ms: int
    model_used: str

class Stats(BaseModel):
    total_queries: int
    avg_response_time_ms: float
    model_info: dict

# Database functions
def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

# RAG functions
def retrieve_documents(conn, query: str, top_k: int = 5, law_type: str = None):
    """Retrieve relevant documents using vector similarity"""
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    cursor = conn.cursor()
    
    sql = """
        SELECT 
            l.id, l.law_type, l.question, l.answer, l.section_number,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM embeddings e
        JOIN laws l ON e.law_id = l.id
    """
    
    params = [query_embedding]
    
    if law_type:
        sql += " WHERE l.law_type = %s"
        params.append(law_type)
    
    sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
    params.extend([query_embedding, top_k])
    
    cursor.execute(sql, params)
    results = cursor.fetchall()
    cursor.close()
    
    return [
        {
            'law_id': r[0],
            'law_type': r[1],
            'question': r[2],
            'answer': r[3],
            'section_number': r[4],
            'similarity': float(r[5])
        }
        for r in results
    ]

def generate_answer_with_ollama(query: str, sources: List[dict]) -> str:
    """Generate answer using Ollama (Llama 3.1 or other open-source model)"""
    
    # Build context from sources
    context_parts = []
    for i, source in enumerate(sources, 1):
        context_parts.append(
            f"[Source {i}] {source['law_type']}"
            + (f" - Section {source['section_number']}" if source['section_number'] else "")
            + f"\nQuestion: {source['question']}\nAnswer: {source['answer']}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a legal assistant specializing in Indian law. Answer the user's question based ONLY on the provided legal sources below. Be precise, cite specific sections, and use clear language.

Legal Sources:
{context}

User Question: {query}

Instructions:
1. Provide a clear, accurate answer based strictly on the sources above
2. Cite specific sections and laws (e.g., "According to IPC Section 302...")
3. If sources don't fully answer the question, state what information is available
4. Keep your answer concise and focused
5. Use simple language while maintaining legal accuracy

Answer:"""
    
    try:
        # Call Ollama
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                'temperature': 0.3,
                'top_p': 0.9,
                'num_predict': 500
            }
        )
        
        return response['response'].strip()
    
    except Exception as e:
        # Fallback to best matching source if Ollama fails
        print(f"Ollama error: {e}")
        if sources:
            return f"Based on {sources[0]['law_type']}: {sources[0]['answer']}"
        return "Unable to generate answer. Please try again."

# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "LawLM API",
        "version": "1.0.0",
        "model": OLLAMA_MODEL
    }

@app.post("/query", response_model=QueryResponse)
async def query_law(request: QueryRequest):
    """Query the legal database with RAG"""
    start_time = time.time()
    
    conn = get_db_connection()
    
    try:
        # Retrieve relevant documents
        sources = retrieve_documents(conn, request.query, request.top_k, request.law_type)
        
        if not sources:
            raise HTTPException(
                status_code=404,
                detail="No relevant legal information found"
            )
        
        # Generate answer
        if request.use_llm:
            answer = generate_answer_with_ollama(request.query, sources)
            model_used = OLLAMA_MODEL
        else:
            answer = sources[0]['answer']
            model_used = "direct_retrieval"
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log query
        cursor = conn.cursor()
        law_ids = [s['law_id'] for s in sources]
        cursor.execute(
            """INSERT INTO queries 
               (user_query, retrieved_law_ids, response_text, response_time_ms)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (request.query, law_ids, answer, response_time_ms)
        )
        conn.commit()
        cursor.close()
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=[Source(**s) for s in sources],
            response_time_ms=response_time_ms,
            model_used=model_used
        )
    
    finally:
        conn.close()

@app.get("/stats", response_model=Stats)
async def get_stats():
    """Get system statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM queries")
        total_queries = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(response_time_ms) FROM queries")
        avg_response_time = cursor.fetchone()[0] or 0
        
        return Stats(
            total_queries=total_queries,
            avg_response_time_ms=float(avg_response_time),
            model_info={
                "embedding_model": "all-MiniLM-L6-v2",
                "generation_model": OLLAMA_MODEL,
                "embedding_dimension": 384
            }
        )
    finally:
        cursor.close()
        conn.close()

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Check Ollama
    try:
        ollama.list()
        ollama_status = "healthy"
    except:
        ollama_status = "unhealthy"
    
    return {
        "status": "online",
        "database": db_status,
        "ollama": ollama_status,
        "model": OLLAMA_MODEL
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print(f"✓ LawLM API started with {OLLAMA_MODEL}")
    print("✓ Docs available at: http://localhost:8000/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

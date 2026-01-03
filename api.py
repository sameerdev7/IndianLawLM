"""
LawLM - FastAPI Backend with Authentication & Open-Source LLM
RESTful API with JWT authentication and Ollama (Llama 3.1) integration
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import psycopg2
from sentence_transformers import SentenceTransformer
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import ollama
import os
import time

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
OLLAMA_MODEL = "llama3.1:3b"  # You can change to mistral, phi3, etc.

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
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
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Embedding model loaded")

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

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

class FeedbackRequest(BaseModel):
    query_id: int
    rating: int  # 1-5
    comment: Optional[str] = None

class Stats(BaseModel):
    total_queries: int
    total_users: int
    avg_response_time_ms: float
    model_info: dict

# Database functions
def get_db():
    """Get database connection"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

def init_auth_tables():
    """Initialize authentication tables"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            full_name VARCHAR(255),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        ALTER TABLE queries ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id)
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✓ Auth tables initialized")

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user email"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

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
                'temperature': 0.3,  # Lower temperature for factual answers
                'top_p': 0.9,
                'num_predict': 500  # Max tokens
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

@app.post("/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, conn = Depends(get_db)):
    """Register a new user"""
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
    if cursor.fetchone():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    hashed_password = get_password_hash(user.password)
    cursor.execute(
        "INSERT INTO users (email, hashed_password, full_name) VALUES (%s, %s, %s) RETURNING id",
        (user.email, hashed_password, user.full_name)
    )
    conn.commit()
    cursor.close()
    
    # Create token
    access_token = create_access_token(data={"sub": user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin, conn = Depends(get_db)):
    """Login and get access token"""
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT hashed_password, is_active FROM users WHERE email = %s",
        (user.email,)
    )
    result = cursor.fetchone()
    cursor.close()
    
    if not result or not verify_password(user.password, result[0]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not result[1]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    access_token = create_access_token(data={"sub": user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/query", response_model=QueryResponse)
async def query_law(
    request: QueryRequest,
    current_user: str = Depends(get_current_user),
    conn = Depends(get_db)
):
    """Query the legal database with RAG"""
    start_time = time.time()
    
    # Retrieve relevant documents
    sources = retrieve_documents(conn, request.query, request.top_k, request.law_type)
    
    if not sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
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
    cursor.execute("SELECT id FROM users WHERE email = %s", (current_user,))
    user_id = cursor.fetchone()[0]
    
    law_ids = [s['law_id'] for s in sources]
    cursor.execute(
        """INSERT INTO queries 
           (user_query, retrieved_law_ids, response_text, response_time_ms, user_id)
           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
        (request.query, law_ids, answer, response_time_ms, user_id)
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

@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: str = Depends(get_current_user),
    conn = Depends(get_db)
):
    """Submit feedback for a query"""
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE queries SET feedback_score = %s WHERE id = %s",
        (feedback.rating, feedback.query_id)
    )
    conn.commit()
    cursor.close()
    
    return {"message": "Feedback submitted successfully"}

@app.get("/stats", response_model=Stats)
async def get_stats(
    current_user: str = Depends(get_current_user),
    conn = Depends(get_db)
):
    """Get system statistics"""
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM queries")
    total_queries = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = true")
    total_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(response_time_ms) FROM queries")
    avg_response_time = cursor.fetchone()[0] or 0
    
    cursor.close()
    
    return Stats(
        total_queries=total_queries,
        total_users=total_users,
        avg_response_time_ms=float(avg_response_time),
        model_info={
            "embedding_model": "all-MiniLM-L6-v2",
            "generation_model": OLLAMA_MODEL,
            "embedding_dimension": 384
        }
    )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check database
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    init_auth_tables()
    print(f"✓ LawLM API started with {OLLAMA_MODEL}")
    print("✓ Docs available at: http://localhost:8000/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
LawLM - Indian Law Question Answering System
Database Schema Setup with PostgreSQL + pgvector
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'jameerbaba7',  # Change this
    'database': 'lawlm'
}

def create_database():
    """Create the LawLM database if it doesn't exist"""
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname='lawlm'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute('CREATE DATABASE lawlm')
        print("✓ Database 'lawlm' created")
    else:
        print("✓ Database 'lawlm' already exists")
    
    cursor.close()
    conn.close()

def setup_schema():
    """Create tables and enable pgvector extension"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("✓ pgvector extension enabled")
    
    # Create laws table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laws (
            id SERIAL PRIMARY KEY,
            law_type VARCHAR(50) NOT NULL,  -- IPC, CrPC, Constitution
            section_number VARCHAR(100),
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'laws' created")
    
    # Create embeddings table with vector column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            law_id INTEGER REFERENCES laws(id) ON DELETE CASCADE,
            embedding vector(384),  -- dimension for all-MiniLM-L6-v2
            chunk_text TEXT NOT NULL,
            chunk_type VARCHAR(50),  -- 'question', 'answer', 'combined'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'embeddings' created")
    
    # Create index for vector similarity search
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
        ON embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
    print("✓ Vector index created")
    
    # Create queries table for logging
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            user_query TEXT NOT NULL,
            retrieved_law_ids INTEGER[],
            response_text TEXT,
            feedback_score INTEGER,  -- 1-5 rating
            response_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'queries' created")
    
    # Create users/sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            query_count INTEGER DEFAULT 0,
            first_query_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_query_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'sessions' created")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("\n✅ Database schema setup complete!")

if __name__ == "__main__":
    print("Setting up LawLM Database...\n")
    create_database()
    setup_schema()
    print("\nNext steps:")
    print("1. Install pgvector: sudo apt-get install postgresql-14-pgvector")
    print("2. Run data loading script to import JSON files")

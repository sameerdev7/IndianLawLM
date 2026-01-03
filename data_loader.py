"""
LawLM - Data Loader
Load JSON files into PostgreSQL and create embeddings
"""

import json
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from tqdm import tqdm
from psycopg2.extras import execute_values
import os 
from dotenv import load_dotenv 

load_dotenv()


def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "lawlm_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "your_password"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'jameerbaba7',
    'database': 'lawlm'
}

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Model loaded\n")

def load_json_file(filepath):
    """Load a JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def insert_law_data(cursor, law_type, data):
    """Insert law data into database and return law_id"""
    cursor.execute("""
        INSERT INTO laws (law_type, question, answer, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (
        law_type,
        data['question'],
        data['answer'],
        json.dumps({'source_file': law_type})
    ))
    return cursor.fetchone()[0]

def create_embeddings(cursor, law_id, question, answer):
    """Create and store embeddings for question, answer, and combined text"""
    
    # Embedding strategies
    embeddings_data = [
        ('question', question),
        ('answer', answer),
        ('combined', f"{question} {answer}")
    ]
    
    for chunk_type, text in embeddings_data:
        embedding = model.encode(text, normalize_embeddings=True)
        embedding_list = embedding.tolist()
        
        cursor.execute("""
            INSERT INTO embeddings (law_id, embedding, chunk_text, chunk_type)
            VALUES (%s, %s, %s, %s)
        """, (law_id, embedding_list, text, chunk_type))

def process_legal_data(json_files):
    """Process all JSON files and load into database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    total_loaded = 0
    
    for filepath, law_type in json_files:
        print(f"\nProcessing {law_type}...")
        
        if not Path(filepath).exists():
            print(f"⚠ File not found: {filepath}")
            continue
        
        data = load_json_file(filepath)
        
        for item in tqdm(data, desc=f"Loading {law_type}"):
            # Insert law data
            law_id = insert_law_data(cursor, law_type, item)
            
            # Create embeddings
            create_embeddings(cursor, law_id, item['question'], item['answer'])
            
            total_loaded += 1
        
        conn.commit()
        print(f"✓ Loaded {len(data)} entries from {law_type}")
    
    cursor.close()
    conn.close()
    
    return total_loaded

def verify_data():
    """Verify the loaded data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check total entries
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total = cursor.fetchone()[0]
        print(f"\n✓ Total entries in database: {total}")
        
        # Check embedding dimensions using pgvector's vector_dims function
        cursor.execute("SELECT vector_dims(embedding) FROM embeddings LIMIT 1")
        dims = cursor.fetchone()
        if dims:
            print(f"✓ Embedding dimensions: {dims[0]}")
        
        # Sample some entries
        cursor.execute("""
            SELECT source, question, answer 
            FROM embeddings 
            LIMIT 3
        """)
        
        print("\n📋 Sample entries:")
        print("=" * 80)
        for source, question, answer in cursor.fetchall():
            print(f"\nSource: {source}")
            print(f"Q: {question[:100]}...")
            print(f"A: {answer[:100]}...")
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
    finally:
        cursor.close()
        conn.close()
if __name__ == "__main__":
    # Define your JSON files and their types
    json_files = [
        ('ipc_qa.json', 'IPC'),
        ('crpc_qa.json', 'CrPC'),
        ('constitution_qa.json', 'Constitution')
    ]
    
    print("Starting LawLM Data Loading...\n")
    print("This will:")
    print("1. Load JSON files")
    print("2. Create embeddings for each Q&A pair")
    print("3. Store in PostgreSQL with pgvector\n")
    
    total = process_legal_data(json_files)
    
    print(f"\n✅ Successfully loaded {total} legal entries!")
    
    verify_data()
    
    print("✅ Data loading complete! Ready for RAG system.")

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

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'your_password',
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
    """Verify data loaded correctly"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Count laws by type
    cursor.execute("""
        SELECT law_type, COUNT(*) 
        FROM laws 
        GROUP BY law_type
    """)
    
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    for law_type, count in cursor.fetchall():
        print(f"{law_type:20s}: {count:5d} entries")
    
    # Count embeddings
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    embedding_count = cursor.fetchone()[0]
    print(f"{'Total Embeddings':20s}: {embedding_count:5d}")
    
    # Sample embedding dimension
    cursor.execute("SELECT array_length(embedding, 1) FROM embeddings LIMIT 1")
    dim = cursor.fetchone()[0]
    print(f"{'Embedding Dimension':20s}: {dim:5d}")
    
    print("="*50 + "\n")
    
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

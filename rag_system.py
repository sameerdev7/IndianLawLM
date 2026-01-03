"""
LawLM - RAG (Retrieval-Augmented Generation) System
Core retrieval and generation logic
"""

import psycopg2
from sentence_transformers import SentenceTransformer
import anthropic
from typing import List, Dict, Tuple
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'your_password',
    'database': 'lawlm'
}

class LawLMRAG:
    def __init__(self, anthropic_api_key: str = None):
        """Initialize RAG system"""
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.conn = psycopg2.connect(**DB_CONFIG)
        
        # Initialize Claude API (optional - can use other LLMs)
        self.use_llm = anthropic_api_key is not None
        if self.use_llm:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        print("✓ LawLM RAG System initialized")
    
    def retrieve(self, query: str, top_k: int = 5, law_type: str = None) -> List[Dict]:
        """
        Retrieve most relevant law entries using vector similarity
        
        Args:
            query: User question
            top_k: Number of results to retrieve
            law_type: Filter by law type (IPC, CrPC, Constitution) or None for all
        
        Returns:
            List of relevant law entries with similarity scores
        """
        # Create query embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        cursor = self.conn.cursor()
        
        # Build SQL query with optional filtering
        sql = """
            SELECT 
                l.id,
                l.law_type,
                l.question,
                l.answer,
                l.section_number,
                e.chunk_type,
                1 - (e.embedding <=> %s::vector) AS similarity
            FROM embeddings e
            JOIN laws l ON e.law_id = l.id
        """
        
        params = [query_embedding]
        
        if law_type:
            sql += " WHERE l.law_type = %s"
            params.append(law_type)
        
        sql += """
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """
        params.extend([query_embedding, top_k])
        
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'law_id': row[0],
                'law_type': row[1],
                'question': row[2],
                'answer': row[3],
                'section_number': row[4],
                'chunk_type': row[5],
                'similarity': float(row[6])
            })
        
        cursor.close()
        return results
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate response using Claude with retrieved context
        
        Args:
            query: User question
            retrieved_docs: Retrieved relevant documents
        
        Returns:
            Generated answer with citations
        """
        if not self.use_llm:
            # Fallback: return most relevant answer
            if retrieved_docs:
                return retrieved_docs[0]['answer']
            return "No relevant information found."
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {i}] {doc['law_type']}"
                + (f" - Section {doc['section_number']}" if doc['section_number'] else "")
                + f"\nQ: {doc['question']}\nA: {doc['answer']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a legal assistant specializing in Indian law. Answer the following question based ONLY on the provided legal sources. Be precise and cite specific sections when applicable.

Legal Sources:
{context}

User Question: {query}

Instructions:
1. Provide a clear, accurate answer based on the sources
2. Cite specific sections and laws (e.g., "According to IPC Section 300...")
3. If the sources don't fully answer the question, state what information is available
4. Use simple language while maintaining legal accuracy

Answer:"""
        
        # Call Claude API
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def answer_query(self, query: str, top_k: int = 5, law_type: str = None) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate + log
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            law_type: Filter by law type
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k, law_type)
        
        # Generate response
        response = self.generate_response(query, retrieved_docs)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log query
        self.log_query(query, retrieved_docs, response, response_time_ms)
        
        return {
            'query': query,
            'answer': response,
            'sources': retrieved_docs,
            'response_time_ms': response_time_ms,
            'num_sources': len(retrieved_docs)
        }
    
    def log_query(self, query: str, retrieved_docs: List[Dict], 
                  response: str, response_time_ms: int):
        """Log query to database for analytics"""
        cursor = self.conn.cursor()
        
        law_ids = [doc['law_id'] for doc in retrieved_docs]
        
        cursor.execute("""
            INSERT INTO queries 
            (user_query, retrieved_law_ids, response_text, response_time_ms)
            VALUES (%s, %s, %s, %s)
        """, (query, law_ids, response, response_time_ms))
        
        self.conn.commit()
        cursor.close()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    # For Claude: rag = LawLMRAG(anthropic_api_key="your-api-key")
    rag = LawLMRAG()  # Without LLM, returns best match
    
    # Test queries
    test_queries = [
        "What is the punishment for murder under IPC?",
        "What does Section 302 IPC deal with?",
        "What are the rights of an accused person?"
    ]
    
    print("\n" + "="*70)
    print("TESTING LawLM RAG SYSTEM")
    print("="*70 + "\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 70)
        
        result = rag.answer_query(query, top_k=3)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"\nSources Found: {result['num_sources']}")
        print(f"Response Time: {result['response_time_ms']}ms")
        
        print("\nTop Sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source['law_type']} - Similarity: {source['similarity']:.3f}")
        
        print("\n" + "="*70 + "\n")
    
    rag.close()
    print("✅ Testing complete!")

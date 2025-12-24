from sentence_transformers import SentenceTransformer
from app.core.database import get_db_connection
from pgvector.psycopg2 import register_vector  # <--- C'est la ligne manquante !
import numpy as np

class RAGEngine:
    def __init__(self):
        # Modèle multilingue (Français/Arabe/Anglais) léger et performant
        print("Chargement du modèle d'embedding...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def embed_text(self, text: str):
        return self.model.encode(text)

    def search(self, query: str, limit: int = 3):
        """Recherche les documents les plus proches sémantiquement"""
        query_vector = self.embed_text(query)
        
        conn = get_db_connection()
        register_vector(conn)
        cur = conn.cursor()
        
        # Recherche par distance Cosine (<=>)
        # On retourne le contenu et la distance (plus c'est petit, plus c'est proche)
        cur.execute("""
            SELECT content, embedding <=> %s::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT %s;
        """, (query_vector, limit))
        
        results = cur.fetchall()
        cur.close()
        conn.close()

        print(f"\n🔍 DEBUG RAG pour : '{query}'")
        formatted_results = []
        for r in results:
            score = 1 - r[1]
            print(f"   👉 Trouvé : '{r[0][:50]}...' | Distance: {r[1]:.4f} | Score: {score:.4f}")
            formatted_results.append({"content": r[0], "score": score})
        
        # On formate la réponse
        return formatted_results

# Singleton
rag_engine = RAGEngine()

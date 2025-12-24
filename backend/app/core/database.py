import psycopg2
from pgvector.psycopg2 import register_vector
import json
from datetime import datetime
import os

def get_db_connection():
    conn = psycopg2.connect(
        host="postgres",  # Nom du service dans docker-compose
        database=os.getenv("POSTGRES_DB", "agent_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "adminpassword")
    )
    return conn

def init_db():
    """Initialise l'extension vectorielle et la table documents"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Activer l'extension pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Créer la table documents
    # Le modèle 'paraphrase-multilingual-MiniLM-L12-v2' sort des vecteurs de dimension 384
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding vector(384)
        );
    """)

    # 3. NOUVEAU : Table conversations
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_message TEXT,
            bot_response TEXT,
            language VARCHAR(10),
            sentiment_score INT,
            intent VARCHAR(50)
        );
    """)
    
    # 3. Créer un index pour la recherche rapide (IVFFlat)
    # Note: On le fait généralement après avoir inséré des données, mais on prépare le terrain
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Base de données initialisée avec pgvector.")

# NOUVEAU : Fonction pour logger une interaction
def log_conversation(user_msg, bot_resp, lang, sentiment, intent):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversations (user_message, bot_response, language, sentiment_score, intent)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_msg, bot_resp, lang, sentiment, intent)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Erreur logging DB: {e}")

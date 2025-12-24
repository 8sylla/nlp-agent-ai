from app.core.database import init_db, get_db_connection
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# Données factices (FAQ)
FAQ_DATA = [
    "Les retours sont gratuits sous 30 jours pour tous les produits non ouverts.",
    "Le remboursement se fait sous 5 jours ouvrés sur le moyen de paiement d'origine.",
    "Nous livrons dans tout le Sénégal, la Côte d'Ivoire et le Maroc via DHL.",
    "Pour suivre votre commande, utilisez le numéro commençant par CMD sur notre page de suivi.",
    "Le service client est ouvert 24h/24 et 7j/7 pour les urgences.",
    "Les frais de douane sont à la charge du client pour les livraisons hors zone CEDEAO."
]

def ingest():
    print("Initialisation de la DB...")
    init_db()
    
    print("Chargement du modèle...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    print(f"Insertion de {len(FAQ_DATA)} documents...")
    for text in FAQ_DATA:
        # Création du vecteur
        embedding = model.encode(text).tolist()
        
        # Insertion SQL
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, embedding)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Ingestion terminée avec succès !")

if __name__ == "__main__":
    ingest()

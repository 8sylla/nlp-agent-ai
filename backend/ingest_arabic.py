from app.core.database import get_db_connection
from sentence_transformers import SentenceTransformer

# FAQ en Arabe (Dialecte mixte/Standard simple)
ARABIC_FAQ = [
    "يمكنك إرجاع المنتج مجانًا خلال 30 يومًا.",  # Retours gratuits 30 jours
    "يتم استرداد المبلغ خلال 5 أيام عمل.",        # Remboursement 5 jours
    "نحن نقوم بالتوصيل إلى المغرب والسنغال وساحل العاج عبر DHL.", # Livraison Maroc/Sénégal...
    "لتتبع طلبك، استخدم الرقم الذي يبدأ بـ CMD.", # Suivi commande
    "خدمة العملاء متاحة 24/7 للطوارئ."            # Service client 24/7
]

def ingest_arabic():
    print("Chargement du modèle multilingue...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    print(f"Insertion de {len(ARABIC_FAQ)} documents arabes...")
    for text in ARABIC_FAQ:
        embedding = model.encode(text).tolist()
        
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, embedding)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Ingestion Arabe terminée !")

if __name__ == "__main__":
    ingest_arabic()

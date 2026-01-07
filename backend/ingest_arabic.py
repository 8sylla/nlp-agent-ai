from app.core.database import get_db_connection
from sentence_transformers import SentenceTransformer

# FAQ en Arabe (Dialecte mixte/Standard simple)
ARABIC_FAQ = [
    # --- SIYASSAT AL-ISTIRJAA (Politique de Retour) ---
    "يمكنك إرجاع المنتج مجانًا خلال 30 يومًا إذا لم يتم فتحه.",  # Retour gratuit 30 jours non ouvert
    "في حالة وصول المنتج مكسورًا أو تالفًا، لديك 60 يومًا للإرجاع.", # 60 jours si cassé
    "يتم استرداد المبلغ خلال 5 أيام عمل عبر نفس وسيلة الدفع.", # Remboursement 5 jours même méthode
    "لا يمكن إرجاع منتجات العناية الشخصية (مثل سماعات الأذن) إذا تم فتحها لأسباب صحية.", # Hygiène (écouteurs)
    "مصاريف الإرجاع مجانية إذا كان الخطأ من جانبنا.", # Frais retour gratuits si erreur vendeur

    # --- AL-TAWSSIL (Livraison) ---
    "نقوم بالتوصيل إلى جميع مدن المغرب (الدار البيضاء، الرباط، مراكش) خلال 24 ساعة.", # Livraison Maroc 24h
    "الشحن إلى السنغال وكوت ديفوار يتم عبر DHL Express ويستغرق 3 أيام.", # Sénégal/CI DHL 3 jours
    "التوصيل مجاني للطلبات التي تزيد عن 1000 درهم أو 100 يورو.", # Livraison gratuite > 100€
    "يمكنك تتبع طلبيتك باستخدام الرقم الذي يبدأ بـ CMD على صفحة التتبع.", # Suivi CMD
    "إذا لم تكن في المنزل، سيحاول المندوب الاتصال بك لتحديد موعد آخر.", # Livreur appelle si absent

    # --- AL-DAF3 (Paiement) ---
    "نقبل الدفع عند الاستلام (Cash on Delivery) في الدار البيضاء ودكار فقط.", # Cash on Delivery (COD)
    "يمكنك الدفع عبر Orange Money أو Wave في دول غرب إفريقيا.", # Orange Money / Wave
    "الرسوم الجمركية مشمولة في السعر النهائي لدول المغرب العربي.", # Douanes incluses Maghreb
    "لا نقبل الشيكات كوسيلة للدفع.", # Pas de chèques

    # --- AL-DAMAN (Garantie) ---
    "جميع الهواتف الذكية تأتي مع ضمان لمدة سنتين من الشركة المصنعة.", # Garantie 2 ans smartphones
    "ضمان الآيفون 15 برو ماكس يغطي عيوب التصنيع وليس الكسر العرضي.", # Garantie iPhone couvre pas casse
    "للاستفادة من الضمان، يرجى الاحتفاظ بالفاتورة الأصلية.", # Garder facture pour garantie

    # --- KHIDMAT AL-OUMALAA (Service Client) ---
    "خدمة العملاء متاحة 24/7 للطوارئ عبر الدردشة المباشرة.", # Service client 24/7 chat
    "يمكنك التواصل معنا عبر واتساب على الرقم الموجود أسفل الموقع.", # Contact WhatsApp
    "مقرنا الرئيسي لعمليات إفريقيا يقع في الدار البيضاء.", # Siège Casablanca
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

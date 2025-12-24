import logging
from langdetect import detect
from app.core.nlu_engine import nlu_engine
from app.core.rag_engine import rag_engine
from app.core.mock_services import order_service
from app.core.sentiment_engine import sentiment_engine
from app.core.database import log_conversation

# Configuration du logger
logger = logging.getLogger("orchestrator")

async def process_user_message(text: str):
    """
    Cerveau central de l'agent :
    1. Détecte la langue
    2. Analyse le sentiment (Colère ?)
    3. Comprend l'intention (NLU) ou cherche dans la base (RAG)
    4. Construit la réponse
    5. Loggue la conversation en base de données
    """
    
    # --- 1. Détection de Langue ---
    try:
        lang = detect(text)
    except Exception:
        lang = "fr" # Fallback par défaut
    
    print(f"🌍 [Orchestrator] Langue détectée: {lang}")

    # --- 2. Analyse de Sentiment ---
    sentiment = sentiment_engine.analyze(text)
    sentiment_score = sentiment['stars']
    is_negative = sentiment['is_negative']
    
    print(f"❤️ [Orchestrator] Sentiment: {sentiment_score}/5 (Négatif: {is_negative})")

    # Préfixe d'empathie si le client est en colère
    prefix = ""
    if is_negative:
        if lang == 'ar':
            prefix = "نعتذر عن الإزعاج. " # "Nous nous excusons pour le désagrément"
        else:
            prefix = "Je détecte une insatisfaction et je suis navré pour ce désagrément. "

    # --- 3. Compréhension (NLU vs RAG) ---
    
    intent = "UNKNOWN"
    score = 0.0
    entities = []
    
    # Stratégie :
    # - Si FR : On utilise le modèle Spacy NLU (car entraîné en FR)
    # - Si AR : On saute le NLU (sauf si Regex) et on privilégie le RAG sémantique
    
    if lang == 'fr':
        nlu_result = nlu_engine.analyze(text)
        intent = nlu_result["intent"]
        score = nlu_result["confidence"]
        entities = nlu_result["entities"]
        print(f"🤖 [Orchestrator] NLU Intent: {intent} ({score:.2f})")
    
    # --- 4. Logique de Réponse (Décision) ---
    
    final_response = ""

    # SCÉNARIO A : Suivi de Commande (Prioritaire)
    # On vérifie si l'intent est TRACK_ORDER OU si on trouve une entité ORDER_ID via Regex (pour l'Arabe aussi)
    # Note: nlu_engine.analyze fait déjà une regex ORDER_ID qui marche peu importe la langue si le format est CMD-XXX
    
    # On force la recherche d'entité ORDER_ID même si NLU a échoué (via regex simple du nlu_engine)
    if not entities and "CMD-" in text.upper():
         # Petit hack pour récupérer l'entité si le modèle NLU l'a ratée mais que la regex l'a vue
         temp_analysis = nlu_engine.analyze(text)
         entities = temp_analysis["entities"]

    order_id_entity = next((e["text"] for e in entities if e["label"] == "ORDER_ID"), None)

    if (intent == "TRACK_ORDER" and score > 0.6) or order_id_entity:
        if order_id_entity:
            # On a l'ID, on appelle le Mock Service
            status = order_service.get_order_status(order_id_entity)
            if status:
                final_response = prefix + status
            else:
                # ID trouvé mais inconnu dans le Mock
                msg = "لم يتم العثور على هذا الطلب." if lang == 'ar' else "Je ne trouve pas de commande avec ce numéro."
                final_response = prefix + msg
        else:
            # On a l'intention mais pas l'ID
            msg = "للتتبع، يرجى تقديم رقم الطلب (مثال: CMD-123)." if lang == 'ar' else "Pour suivre votre colis, j'ai besoin de votre numéro de commande (ex: CMD-123)."
            final_response = prefix + msg

    # SCÉNARIO B : Recherche dans la Base de Connaissances (RAG)
    # Si ce n'est pas une commande, ou si l'intent est RETURN/TECH, ou si c'est de l'Arabe
    if not final_response:
        print("🔍 [Orchestrator] Appel RAG...")
        rag_results = rag_engine.search(text)
        
        # On abaisse le seuil de pertinence car le modèle multilingue peut être subtil
        if rag_results and rag_results[0]['score'] > 0.25:
            best_answer = rag_results[0]['content']
            final_response = prefix + best_answer
        else:
            print("❌ [Orchestrator] RAG score trop faible.")

    # SCÉNARIO C : Fallback (Échec)
    if not final_response:
        if lang == 'ar':
            final_response = prefix + "لست متأكداً من فهمي. هل يمكنك إعادة الصياغة؟ يمكنني تتبع الطلبات أو الإجابة على الأسئلة."
        else:
            final_response = prefix + "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ? (Je peux suivre vos commandes ou répondre à vos questions)."

    # --- 5. Logging (Sauvegarde) ---
    try:
        # On loggue l'échange pour le Dashboard Admin
        log_conversation(
            user_msg=text,
            bot_resp=final_response,
            lang=lang,
            sentiment=sentiment_score,
            intent=intent
        )
    except Exception as e:
        print(f"⚠️ Erreur Logging: {e}")

    return final_response
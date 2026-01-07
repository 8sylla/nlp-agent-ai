import logging
import re # NOUVEAU : Nécessaire pour trouver plusieurs patterns
from langdetect import detect

# --- Import des Moteurs IA ---
from app.core.nlu_engine import nlu_engine          
from app.core.rag_engine import rag_engine          
from app.core.graph_engine import graph_engine      
from app.core.sentiment_engine import sentiment_engine 
from app.core.contextualizer import contextualizer  

# --- Import des Services & Données ---
from app.core.mock_services import order_service    
from app.core.database import log_conversation      
from app.core.memory import ConversationMemory      

logger = logging.getLogger("orchestrator")

async def process_user_message(text: str, session_id: str = "default"):
    """
    Cerveau central de l'agent (Architecture Hybride avec Mémoire).
    Gère désormais les MULTIPLES numéros de commande.
    """
    
    print(f"\n📨 [Orchestrator] Reçu : '{text}' (Session: {session_id})")

    # =========================================================================
    # ÉTAPE 1 : GESTION DE LA MÉMOIRE & CONTEXTE
    # =========================================================================
    memory = ConversationMemory(session_id)
    history = memory.get_history() 
    standalone_text = text

    if history:
        # On ne reformule PAS si le texte contient explicitement "CMD-", 
        # car c'est une requête précise qui n'a pas besoin de contexte.
        if "CMD-" not in text.upper():
            print("🤔 [Contexte] Reformulation en cours...")
            try:
                standalone_text = contextualizer.rewrite(history, text)
                print(f"📝 [Contexte] Question Reformulée : '{standalone_text}'")
            except Exception as e:
                print(f"⚠️ Erreur reformulation: {e}")

    # =========================================================================
    # ÉTAPE 2 : ANALYSE LINGUISTIQUE
    # =========================================================================
    try:
        lang = detect(text)
    except:
        lang = "fr"
    print(f"🌍 [Langue] Détectée: {lang}")

    sentiment = sentiment_engine.analyze(text)
    is_negative = sentiment['stars'] <= 1
    
    prefix = ""
    if is_negative:
        if lang == 'ar':
            prefix = "نعتذر عن الإزعاج. "
        else:
            prefix = "Je suis navré pour ce désagrément. "

    intent = "UNKNOWN"
    score = 0.0
    
    if lang == 'fr':
        nlu_result = nlu_engine.analyze(text)
        intent = nlu_result["intent"]
        score = nlu_result["confidence"]
        # Note: On n'utilise plus entities ici pour les commandes, on va utiliser une regex plus robuste

    # =========================================================================
    # ÉTAPE 3 : STRATÉGIE DE RÉSOLUTION (CASCADE)
    # =========================================================================
    
    final_response = ""
    resolved_intent = intent

    # -------------------------------------------------------------------------
    # STRATÉGIE A : TRANSACTIONNEL (MULTI-COMMANDES)
    # -------------------------------------------------------------------------
    
    # 1. Extraction robuste de TOUS les numéros de commande (ex: CMD-123 et CMD-456)
    # On utilise une Regex qui cherche le pattern CMD- suivi de chiffres/lettres
    found_order_ids = re.findall(r'CMD-[\w\d]+', text.upper())
    
    # Dédoublonnage (au cas où l'utilisateur répète le même ID)
    found_order_ids = list(set(found_order_ids))

    if found_order_ids:
        print(f"📦 [Transactionnel] IDs détectés : {found_order_ids}")
        responses_list = []
        
        # On boucle sur chaque commande trouvée
        for order_id in found_order_ids:
            status = order_service.get_order_status(order_id)
            if status:
                responses_list.append(status)
            else:
                msg = f"Commande {order_id} : Introuvable." if lang == 'fr' else f"الطلب {order_id} غير موجود."
                responses_list.append(msg)
        
        # On joint toutes les réponses
        final_response = prefix + "\n\n".join(responses_list)
        resolved_intent = "TRACK_ORDER_SUCCESS"

    elif intent == "TRACK_ORDER" and score > 0.6:
        # L'intention est là ("Où est mon colis ?") mais AUCUN ID n'a été trouvé
        msg = "Pour suivre vos commandes, j'ai besoin des numéros (ex: CMD-123)." if lang == 'fr' else "يرجى تزويدي برقم الطلب للتتبع (مثال: CMD-123)."
        final_response = prefix + msg
        resolved_intent = "TRACK_ORDER_ASK_ID"

    # -------------------------------------------------------------------------
    # STRATÉGIE B : GRAPHRAG (Si ce n'est pas une commande)
    # -------------------------------------------------------------------------
    if not final_response:
        print("🔍 [GraphRAG] Interrogation Neo4j...")
        try:
            graph_answer = graph_engine.query(standalone_text)
            
            if graph_answer:
                cleaned = graph_answer.replace("Answer:", "").strip()
                # Filtre anti-hallucination
                invalid_triggers = ["je ne sais pas", "i don't know", "no information", "don't have information"]
                is_valid = len(cleaned) > 5 and not any(trig in cleaned.lower() for trig in invalid_triggers)
                
                if is_valid:
                    print("✅ [GraphRAG] Réponse trouvée !")
                    final_response = prefix + cleaned
                    resolved_intent = "GRAPH_QUERY"
        except Exception as e:
            print(f"⚠️ Erreur GraphRAG: {e}")

    # -------------------------------------------------------------------------
    # STRATÉGIE C : VECTORRAG (Fallback)
    # -------------------------------------------------------------------------
    if not final_response:
        print("⚠️ [VectorRAG] Fallback sur Postgres...")
        rag_results = rag_engine.search(standalone_text)
        
        if rag_results and rag_results[0]['score'] > 0.25:
            print(f"✅ [VectorRAG] Trouvé (Score: {rag_results[0]['score']:.2f})")
            final_response = prefix + rag_results[0]['content']
            resolved_intent = "VECTOR_QUERY"
        else:
            print("❌ [VectorRAG] Score trop faible.")

    # -------------------------------------------------------------------------
    # STRATÉGIE D : ÉCHEC TOTAL
    # -------------------------------------------------------------------------
    if not final_response:
        if lang == 'ar':
            final_response = prefix + "لست متأكداً من فهمي. هل يمكنك إعادة الصياغة؟"
        else:
            final_response = prefix + "Je ne trouve pas l'information précise. Pouvez-vous reformuler ?"
        resolved_intent = "FALLBACK"

    # =========================================================================
    # ÉTAPE 4 : MÉMOIRE & LOGGING
    # =========================================================================
    
    try:
        memory.add_message("user", text)
        memory.add_message("ai", final_response)
    except Exception as e:
        print(f"⚠️ Erreur Redis: {e}")

    msg_id = None
    try:
        msg_id = log_conversation(
            user_msg=text,
            bot_resp=final_response,
            lang=lang,
            sentiment=sentiment['stars'],
            intent=resolved_intent
        )
    except Exception as e:
        print(f"⚠️ Erreur Logging DB: {e}")

    return {
        "text": final_response,
        "id": msg_id,
        "intent": resolved_intent
    }
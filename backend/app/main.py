import uuid
import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Imports internes
from app.schemas.api_models import NLURequest, NLUResponse, RAGRequest, RAGResponse
from app.core.nlu_engine import nlu_engine
from app.core.rag_engine import rag_engine
from app.core.orchestrator import process_user_message
from app.core.database import get_db_connection

app = FastAPI(
    title="Agent Support IA - API",
    version="2.0",
    description="API de support client multilingue (GraphRAG + VectorRAG)"
)

# Configuration CORS pour le Frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mettre http://localhost:3000 en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HEALTHCHECK ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "support-agent-ai"}

# --- DEBUG NLU (Optionnel) ---
@app.post("/v1/analyze", response_model=NLUResponse)
async def analyze_text(request: NLURequest):
    try:
        result = nlu_engine.analyze(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- DEBUG RAG (Optionnel) ---
@app.post("/v1/ask", response_model=RAGResponse)
async def ask_knowledge_base(request: RAGRequest):
    try:
        results = rag_engine.search(request.query)
        if not results or results[0]['score'] < 0.1:
            return {
                "answer": "Je suis désolé, je n'ai pas trouvé d'information précise.",
                "sources": []
            }
        best_match = results[0]['content']
        return {
            "answer": f"D'après mes informations : {best_match}",
            "sources": [r['content'] for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- DASHBOARD ADMIN (LOGS) ---
@app.get("/v1/admin/logs")
async def get_logs():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # MODIF: Ajout de la colonne 'feedback' dans la requête
    cur.execute("""
        SELECT id, timestamp, user_message, bot_response, language, sentiment_score, intent, feedback 
        FROM conversations 
        ORDER BY id DESC LIMIT 50
    """)
    rows = cur.fetchall()
    
    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "timestamp": r[1],
            "user_message": r[2],
            "bot_response": r[3],
            "language": r[4],
            "sentiment_score": r[5],
            "intent": r[6],
            "feedback": r[7] # Nouveau champ (peut être None)
        })
    
    cur.close()
    conn.close()
    return logs

# --- FEEDBACK UTILISATEUR ---
class FeedbackRequest(BaseModel):
    feedback: int # 1 (Like) ou 0 (Dislike)

@app.post("/v1/feedback/{message_id}")
async def vote_message(message_id: int, request: FeedbackRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET feedback = %s WHERE id = %s", (request.feedback, message_id))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

# --- WEBSOCKET CHAT (CŒUR DU SYSTÈME) ---
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Génère un ID unique pour cette session de chat (Mémoire Redis)
    session_id = str(uuid.uuid4())
    print(f"🔗 Nouvelle session: {session_id}")
    
    try:
        while True:
            # 1. Réception
            data = await websocket.receive_text()
            
            # 2. Traitement (Orchestrator renvoie maintenant un DICT)
            response_data = await process_user_message(data, session_id)
            
            # 3. Envoi JSON (Important: json.dumps)
            # Le frontend attend { "text": "...", "id": 123 }
            await websocket.send_text(json.dumps(response_data))
            
    except WebSocketDisconnect:
        print(f"Déconnexion session: {session_id}")
    except Exception as e:
        print(f"Erreur Critique WS: {e}")
from fastapi import FastAPI, HTTPException
from app.schemas.api_models import NLURequest, NLUResponse
from app.core.nlu_engine import nlu_engine
from app.core.rag_engine import rag_engine
from app.schemas.api_models import RAGRequest, RAGResponse
from fastapi import WebSocket, WebSocketDisconnect
from app.core.orchestrator import process_user_message
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import get_db_connection

app = FastAPI(
    title="Agent Support IA - API",
    version="1.0",
    description="API de support client multilingue (Sprint 1 MVP)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "support-agent-ai"}

@app.post("/v1/analyze", response_model=NLUResponse)
async def analyze_text(request: NLURequest):
    """
    Analyse le texte utilisateur pour extraire l'intention et les entités.
    """
    try:
        result = nlu_engine.analyze(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/v1/ask", response_model=RAGResponse)
async def ask_knowledge_base(request: RAGRequest):
    """
    Recherche la réponse la plus pertinente dans la base de connaissances.
    """
    try:
        # 1. Recherche des documents pertinents
        results = rag_engine.search(request.query)
        
        if not results or results[0]['score'] < 0.1:
            return {
                "answer": "Je suis désolé, je n'ai pas trouvé d'information précise à ce sujet dans ma base de connaissances.",
                "sources": []
            }
        
        # 2. Construction de la réponse (Simulation de la Génération pour le Sprint 2)
        # Dans le Sprint 4, on remplacera ça par un appel LLM pour reformuler.
        best_match = results[0]['content']
        
        return {
            "answer": f"D'après mes informations : {best_match}",
            "sources": [r['content'] for r in results]
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Recevoir le message utilisateur
            data = await websocket.receive_text()
            
            # 2. Traiter le message via l'orchestrateur
            response = await process_user_message(data)
            
            # 3. Renvoyer la réponse
            await websocket.send_text(response)
            
    except WebSocketDisconnect:
        print("Client déconnecté")

@app.get("/v1/admin/logs")
async def get_logs():
    conn = get_db_connection()
    cur = conn.cursor()
    # Récupérer les 50 dernières conversations
    cur.execute("SELECT id, timestamp, user_message, bot_response, language, sentiment_score, intent FROM conversations ORDER BY id DESC LIMIT 50")
    rows = cur.fetchall()
    
    # Convertir en JSON
    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "timestamp": r[1], # Datetime
            "user_message": r[2],
            "bot_response": r[3],
            "language": r[4],
            "sentiment_score": r[5],
            "intent": r[6]
        })
    
    cur.close()
    conn.close()
    return logs
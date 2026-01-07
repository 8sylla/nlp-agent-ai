import redis
import json
import os

# Connexion à Redis
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class ConversationMemory:
    def __init__(self, session_id: str):
        self.session_id = f"chat:{session_id}"

    def add_message(self, role: str, content: str):
        """Stocke un message (user ou ai)"""
        msg = json.dumps({"role": role, "content": content})
        # On pousse dans la liste
        redis_client.rpush(self.session_id, msg)
        # On garde seulement les 10 derniers messages (mémoire courte)
        redis_client.ltrim(self.session_id, -10, -1)
        # Expire après 1 heure d'inactivité
        redis_client.expire(self.session_id, 3600)

    def get_history(self):
        """Récupère l'historique formaté pour le LLM"""
        raw_msgs = redis_client.lrange(self.session_id, 0, -1)
        history = []
        for m in raw_msgs:
            data = json.loads(m)
            # On met des marqueurs clairs : "Human:" et "Assistant:"
            role = "Human" if data['role'] == "user" else "Assistant"
            history.append(f"{role}: {data['content']}")
        
        # On renvoie les 6 derniers échanges (suffisant et moins cher)
        return "\n".join(history[-6:])

# Pas de singleton ici, on l'instancie par utilisateur
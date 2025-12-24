import spacy
import os

class NLUEngine:
    def __init__(self, model_path="model_output"):
        print(f"Chargement du modèle depuis {model_path}...")
        # Fallback si le modèle entraîné n'existe pas encore (pour éviter crash au build)
        if os.path.exists(model_path):
            self.nlp = spacy.load(model_path)
        else:
            print("⚠️ Modèle custom non trouvé, chargement du modèle de base.")
            self.nlp = spacy.load("fr_core_news_lg")

    def analyze(self, text: str):
        doc = self.nlp(text)
        
        # 1. Détection d'intention (via textcat)
        intent = "UNKNOWN"
        confidence = 0.0
        
        if "textcat" in self.nlp.pipe_names:
            scores = doc.cats
            # Récupérer l'intent avec le score max
            if scores:
                intent = max(scores, key=scores.get)
                confidence = scores[intent]
        
        # 2. Extraction d'entités (NER)
        entities = []
        for ent in doc.ents:
            entities.append({"label": ent.label_, "text": ent.text})
            
        # Règle simple regex pour Sprint 1 (Numéro de commande)
        import re
        cmd_match = re.search(r'(CMD-\d+)', text)
        if cmd_match:
            entities.append({"label": "ORDER_ID", "text": cmd_match.group(1)})

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities
        }

# Singleton pour réutiliser le modèle
nlu_engine = NLUEngine()

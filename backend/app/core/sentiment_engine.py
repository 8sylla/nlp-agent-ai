from transformers import pipeline
import logging

# On configure les logs pour éviter le spam de warnings TensorFlow/PyTorch
logging.getLogger("transformers").setLevel(logging.ERROR)

class SentimentEngine:
    def __init__(self):
        print("Chargement du modèle d'Analyse de Sentiment...")
        # Modèle très populaire pour le sentiment multilingue (Français, Arabe, Anglais, etc.)
        self.analyzer = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    def analyze(self, text: str):
        """
        Retourne : { "label": "1 star"..."5 stars", "score": float, "is_negative": bool }
        """
        try:
            # Le modèle retourne une liste de dictionnaires [{'label': '1 star', 'score': 0.95}]
            result = self.analyzer(text[:512])[0] # On coupe à 512 caractères pour la performance
            
            label = result['label'] # ex: "1 star", "4 stars"
            stars = int(label.split(' ')[0])
            
            # On considère Négatif si 1 ou 2 étoiles
            is_negative = stars <= 2
            
            return {
                "stars": stars,
                "score": result['score'],
                "is_negative": is_negative
            }
        except Exception as e:
            print(f"Erreur sentiment: {e}")
            return {"stars": 3, "score": 0.0, "is_negative": False}

# Singleton
sentiment_engine = SentimentEngine()

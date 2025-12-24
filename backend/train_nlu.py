import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import os
# import fr_core_news_lg 

# 1. Données d'entraînement MVP (Synthétiques)
TRAIN_DATA = [
    ("Où est ma commande ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je veux suivre le colis CMD-12345", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Mon paquet n'est pas arrivé", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    
    ("Je veux retourner ce produit", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Remboursement svp", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Le produit est cassé, je le renvoie", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    
    ("Le site ne marche pas", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je n'arrive pas à me connecter", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Erreur 404 sur la page paiement", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
]

def train_model(output_dir="model_output"):
    # On charge le modèle français large comme base
    nlp = spacy.load("fr_core_news_lg")

    # print("Chargement du modèle fr_core_news_lg...")
    # nlp = fr_core_news_lg.load()
    
    # Ajout du pipeline de classification de texte (TextCategorizer)
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # Ajouter les labels
    textcat.add_label("TRACK_ORDER")
    textcat.add_label("RETURN")
    textcat.add_label("TECH")

    # Préparation des données
    examples = []
    for text, annots in TRAIN_DATA:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annots))

    # Entraînement (uniquement le component textcat pour ne pas casser le NER existant)
    # Note: En production, on utiliserait 'spacy train' via CLI, ici c'est pour l'exemple scripté
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        print("Début de l'entraînement...")
        for i in range(20): # 20 époques
            random.shuffle(examples)
            losses = {}
            nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
            print(f"Époque {i}, Pertes: {losses}")

    # Sauvegarde
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.to_disk(output_dir)
    print(f"Modèle sauvegardé dans {output_dir}")

if __name__ == "__main__":
    train_model()

    
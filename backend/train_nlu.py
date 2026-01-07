import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import os
# import fr_core_news_lg 

# 1. Données d'entraînement MVP (Synthétiques)
TRAIN_DATA = [
    # =========================================================================
    # INTENT: TRACK_ORDER (Suivi, Livraison, Expédition, Retard)
    # =========================================================================
    ("Où est ma commande ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je veux suivre le colis CMD-12345", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Mon paquet n'est pas arrivé", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Savez-vous quand la commande CMD-987 sera livrée ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Le statut de ma livraison n'a pas bougé depuis 3 jours", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("J'attends toujours mon colis", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Pouvez-vous me donner des nouvelles de l'expédition ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Est-ce que ma commande a été expédiée ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Retard de livraison sur la CMD-1122", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je n'ai pas reçu de mail de confirmation d'envoi", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Le livreur est passé mais je n'étais pas là", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Mon colis est indiqué comme livré mais je n'ai rien", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Suivi CMD-777", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Délais de livraison pour le Sénégal ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("C'est quand que ça arrive ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Il manque un article dans mon colis", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Ou est mon truc ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Tracking number CMD-000", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("La livraison est trop longue", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Expédition en cours ou pas ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Mon colis est bloqué en douane ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Quand est-ce que je reçois ma commande à Casablanca ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Livraison express ou standard ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je n'ai pas de numéro de suivi", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Le transporteur m'a appelé", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Ma commande est-elle partie de l'entrepôt ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Date de livraison estimée", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je suis inquiet pour ma commande CMD-999", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Avez-vous envoyé le colis ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Suivre mon achat", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Toujours en attente de préparation", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Livraison à domicile ou en point relais ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je veux changer la date de livraison", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Mon colis est perdu", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("DHL tracking", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Où en est l'acheminement ?", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Commande non reçue", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Statut CMD-123456", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Retard logistique", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),
    ("Je n'ai rien reçu ce matin", {"cats": {"TRACK_ORDER": 1, "RETURN": 0, "TECH": 0}}),

    # =========================================================================
    # INTENT: RETURN (Retour, Remboursement, Échange, Casse, Insatisfaction)
    # =========================================================================
    ("Je veux retourner ce produit", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Remboursement svp", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Le produit est cassé, je le renvoie", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("La taille ne va pas, je veux changer", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Comment faire un retour ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Je ne suis pas satisfait, rendez-moi mon argent", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Politique de remboursement", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("J'ai reçu le mauvais article", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Annuler ma commande et rembourser", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Le produit ne marche pas, je veux un échange", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Est-ce que les retours sont gratuits ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Délai de rétractation", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Je veux renvoyer la commande CMD-333", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Mon article est arrivé endommagé", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Procédure de renvoi", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Vous m'avez débité mais je veux annuler", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("C'est de la mauvaise qualité, je veux être remboursé", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Ticket de retour", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Combien de temps pour avoir mon argent sur mon compte ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Produit non conforme à la description", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Je me suis trompé de couleur", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Remboursez-moi immédiatement !", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Adresse de retour", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Le colis est arrivé écrasé", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("J'ai changé d'avis, je ne veux plus l'article", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Annulation de commande", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Le vêtement est trop grand", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Il y a un défaut de fabrication", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Je veux faire jouer la garantie satisfait ou remboursé", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Pouvez-vous me faire un avoir ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Erreur dans la commande, je renvoie tout", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Renvoyer un cadeau", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Est-ce que je peux échanger ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Modalités de reprise", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Mon compte bancaire n'a pas reçu le virement de retour", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    ("Je renvoie le colis demain", {"cats": {"TRACK_ORDER": 0, "RETURN": 1, "TECH": 0}}),
    
    # =========================================================================
    # INTENT: TECH (Bug, Site, Compte, Paiement, Connexion)
    # =========================================================================
    ("Le site ne marche pas", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je n'arrive pas à me connecter", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Erreur 404 sur la page paiement", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Mon mot de passe ne fonctionne plus", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Impossible d'ajouter au panier", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("L'application a crashé", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je ne reçois pas vos emails", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Problème d'affichage sur mon téléphone", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Le code promo ne s'applique pas", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Bug lors de la validation", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Ma carte bancaire est refusée par le site", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je n'arrive pas à créer un compte", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Le site est très lent", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Erreur serveur 500", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je ne trouve pas le bouton valider", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Lien mort", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Problème technique", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Mon compte est bloqué", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je ne peux pas changer mon adresse", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Impossible de télécharger ma facture", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("L'écran est blanc quand je clique", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Problème de cookie", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je n'arrive pas à mettre à jour mes infos", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Le paiement PayPal a échoué", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Mon panier se vide tout seul", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je ne reçois pas le code SMS de validation", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Bug sur l'application mobile Android", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Double débit sur ma carte", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Problème d'accès à mon espace client", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Comment supprimer mon compte ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Le site est en maintenance ?", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Message d'erreur bizarre", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Je ne peux pas entrer mon numéro de téléphone", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Le filtre de recherche ne marche pas", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Problème de sécurité", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Le captcha ne fonctionne pas", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Déconnexion intempestive", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Impossible de finaliser l'achat", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
    ("Reset password ne marche pas", {"cats": {"TRACK_ORDER": 0, "RETURN": 0, "TECH": 1}}),
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

    
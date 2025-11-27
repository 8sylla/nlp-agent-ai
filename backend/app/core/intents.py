from enum import Enum
from typing import List, Dict

class Intent(str, Enum):
    """5 intents MVP"""
    TRACK_ORDER = "track_order"
    RETURN_REQUEST = "return_request"
    PRODUCT_INQUIRY = "product_inquiry"
    PAYMENT_ISSUE = "payment_issue"
    GREETING = "greeting"

# Exemples par intent (seed data)
INTENT_EXAMPLES: Dict[str, List[str]] = {
    "track_order": [
        "Où est ma commande ?",
        "Je veux suivre mon colis",
        "Statut de livraison pour ORD-123456",
        "Ma commande arrive quand ?",
        "Numéro de suivi de ma commande",
        "Mon colis n'est pas arrivé",
        "Je n'ai pas reçu ma commande",
        "Quand est-ce que je vais recevoir mon article ?",
        "Tracking de commande ORD-789456",
        "Où se trouve mon paquet actuellement ?",
    ],
    "return_request": [
        "Je veux retourner un article",
        "Comment faire un retour ?",
        "Remboursement possible ?",
        "L'article ne me convient pas",
        "Je voudrais échanger ce produit",
        "Procédure de retour",
        "Retourner ma commande",
        "Article défectueux, je veux un remboursement",
        "Comment renvoyer cet article ?",
        "Je ne suis pas satisfait du produit",
    ],
    "product_inquiry": [
        "Ce produit est disponible en bleu ?",
        "Quelle est la taille de cet article ?",
        "Informations sur ce produit",
        "Est-ce que vous avez ce modèle en stock ?",
        "Caractéristiques techniques de l'article",
        "Ce produit est-il compatible avec ?",
        "Quels sont les matériaux utilisés ?",
        "Dimensions du produit",
        "Y a-t-il une garantie ?",
        "Description détaillée du produit",
    ],
    "payment_issue": [
        "Mon paiement n'est pas passé",
        "Erreur de transaction",
        "Ma carte a été débitée deux fois",
        "Problème avec le paiement",
        "Je n'arrive pas à payer",
        "Facture incorrecte",
        "Montant erroné sur ma facture",
        "Comment payer ma commande ?",
        "Mes coordonnées bancaires ne sont pas acceptées",
        "Remboursement en attente",
    ],
    "greeting": [
        "Bonjour",
        "Salut",
        "Hello",
        "Bonsoir",
        "Coucou",
        "Salutations",
        "Yo",
        "Hey",
        "Bonjour, j'ai besoin d'aide",
        "Salut, tu peux m'aider ?",
    ]
}

# Descriptions des intents
INTENT_DESCRIPTIONS = {
    "track_order": "L'utilisateur veut suivre sa commande ou connaître le statut de livraison",
    "return_request": "L'utilisateur souhaite retourner un article ou demander un remboursement",
    "product_inquiry": "L'utilisateur pose des questions sur un produit (disponibilité, caractéristiques)",
    "payment_issue": "L'utilisateur rencontre un problème de paiement ou de facturation",
    "greeting": "Salutation ou message de bienvenue de l'utilisateur"
}

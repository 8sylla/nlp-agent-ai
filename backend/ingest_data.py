from app.core.database import init_db, get_db_connection
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# Données factices (FAQ)
# Données FAQ riches et contextuelles (Vector Knowledge Base)
FAQ_DATA = [
    # --- 📦 LIVRAISON & EXPÉDITION ---
    "Nous livrons dans tout le Sénégal (Dakar, Thiès, Saint-Louis) sous 24 à 48h ouvrées.",
    "La livraison au Maroc est assurée par Amana Express ou Aramex selon votre ville.",
    "Pour la Côte d'Ivoire, nous proposons la livraison à domicile à Abidjan et en point relais pour l'intérieur du pays.",
    "Les frais de port sont offerts pour toute commande supérieure à 100€ (ou équivalent en monnaie locale).",
    "Pour suivre votre commande, utilisez le numéro commençant par CMD sur notre page de suivi dédiée.",
    "Si votre colis est indiqué comme livré mais que vous n'avez rien reçu, vérifiez auprès de vos voisins ou du gardien.",
    "Les livraisons internationales hors Afrique sont gérées exclusivement par DHL Express.",
    "En cas d'absence lors de la livraison, le livreur tentera un second passage le lendemain.",
    "Nous ne livrons pas les boîtes postales, une adresse physique complète est obligatoire.",
    "Le délai de livraison pour le Cameroun et le Gabon est de 5 à 7 jours ouvrés.",
    "Les retards liés aux inspections douanières ne sont pas de notre responsabilité.",
    "Vous recevrez un SMS de confirmation le matin de la livraison avec le numéro du chauffeur.",

    # --- 💰 PAIEMENT & DOUANES ---
    "Nous acceptons le paiement à la livraison (Cash on Delivery) uniquement pour Casablanca, Dakar et Abidjan.",
    "Les paiements par Orange Money et Wave sont acceptés pour le Sénégal et la Côte d'Ivoire.",
    "Pour le Maroc, vous pouvez payer par virement bancaire ou carte CMI.",
    "Les frais de douane sont inclus pour les pays du Maghreb.",
    "Pour les pays de la zone CEDEAO, la TVA est appliquée selon le taux en vigueur dans le pays de destination.",
    "Les paiements par carte bancaire (Visa, Mastercard) sont sécurisés par le protocole 3D Secure.",
    "Nous n'acceptons pas les paiements par chèque ni par Western Union.",
    "En cas de paiement refusé, veuillez contacter votre banque pour vérifier le plafond de votre carte.",
    "La facture est envoyée automatiquement par email après la validation de la commande.",
    "Les prix affichés incluent toutes les taxes locales pour les clients particuliers.",

    # --- 🔄 RETOURS & REMBOURSEMENTS ---
    "Les retours sont gratuits sous 30 jours pour tous les produits non ouverts et dans leur emballage d'origine.",
    "Si vous avez ouvert le produit mais qu'il ne vous plaît pas, des frais de reconditionnement de 15% s'appliquent.",
    "Le remboursement se fait sous 5 jours ouvrés sur le moyen de paiement d'origine après réception du retour.",
    "Pour les paiements effectués à la livraison, le remboursement se fera par virement bancaire ou bon d'achat.",
    "Les produits d'hygiène (écouteurs intra-auriculaires, rasoirs) ne sont ni repris ni échangés s'ils sont déballés.",
    "Si le produit arrive cassé, vous devez nous envoyer une photo sous 48h pour un échange immédiat.",
    "Les frais de retour sont à notre charge si le produit est défectueux ou s'il y a une erreur de notre part.",
    "L'étiquette de retour prépayée est disponible dans votre espace client.",
    "Les remboursements sur compte Mobile Money (Orange/Wave) sont instantanés après validation du retour.",
    "La garantie satisfait ou remboursé est valable 14 jours pour les articles en solde.",

    # --- 🛠️ GARANTIE & SAV ---
    "Tous nos produits électroniques neufs sont garantis 2 ans constructeur.",
    "Les produits reconditionnés bénéficient d'une garantie commerciale de 6 mois.",
    "La garantie couvre les pannes matérielles mais pas la casse accidentelle ni l'oxydation.",
    "Pour faire jouer la garantie, contactez le support avec votre numéro de série.",
    "Nous disposons de centres de réparation agréés à Casablanca et Dakar.",
    "Le délai moyen de réparation est de 10 à 15 jours ouvrés.",
    "Si le produit n'est pas réparable, nous procédons à un échange à neuf.",
    "Les accessoires (câbles, chargeurs) sont garantis 1 an.",

    # --- 👤 COMPTE & PROMO ---
    "Vous pouvez modifier votre adresse de livraison tant que la commande n'est pas expédiée.",
    "Pour supprimer votre compte et vos données personnelles, envoyez une demande à privacy@notresite.com.",
    "Le programme de fidélité vous donne 1 point pour chaque euro dépensé.",
    "Les codes promo ne sont pas cumulables avec les offres en cours.",
    "Si vous avez oublié votre mot de passe, cliquez sur 'Mot de passe oublié' pour recevoir un lien de réinitialisation.",
    "L'inscription à la newsletter vous offre -10% sur votre première commande.",
    "Vous pouvez parrainer des amis et gagner 5€ par filleul actif.",

    # --- 📞 CONTACT & HORAIRES ---
    "Le service client est ouvert 24h/24 et 7j/7 pour les urgences via le chat en ligne.",
    "Notre support téléphonique est disponible de 8h à 20h (GMT).",
    "Vous pouvez nous contacter sur WhatsApp au +212 6 00 00 00 00.",
    "Nous répondons aux emails sous 24h ouvrées.",
    "Notre siège social pour l'Afrique est basé à Casablanca, Maroc.",
    "Pour les réclamations graves, demandez à parler à un superviseur via le chat."
]

def ingest():
    print("Initialisation de la DB...")
    init_db()
    
    print("Chargement du modèle...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    print(f"Insertion de {len(FAQ_DATA)} documents...")
    for text in FAQ_DATA:
        # Création du vecteur
        embedding = model.encode(text).tolist()
        
        # Insertion SQL
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (text, embedding)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Ingestion terminée avec succès !")

if __name__ == "__main__":
    ingest()

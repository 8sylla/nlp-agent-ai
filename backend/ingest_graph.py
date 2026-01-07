import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_groq import ChatGroq

# Configuration Neo4j
os.environ["NEO4J_URI"] = "bolt://neo4j:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password1234"

# Données brutes riches et interconnectées
raw_text = [
    """
    -- POLITIQUES GÉNÉRALES & LIVRAISON AFRIQUE --
    Politique de Retour : Les produits électroniques peuvent être retournés sous 30 jours s'ils sont non ouverts.
    Si le produit est défectueux ou cassé à l'arrivée, le délai de retour est étendu à 60 jours.
    Les remboursements sont effectués sous 5 jours ouvrés après réception du colis.
    La livraison est gratuite pour toute commande supérieure à 100€ à destination du Maroc, du Sénégal et de la Côte d'Ivoire.
    Pour les autres pays d'Afrique, la livraison est assurée par DHL Express en 3 à 5 jours.
    Les frais de douane sont inclus pour les commandes vers le Maghreb, mais à la charge du client pour l'Afrique de l'Ouest.

    -- ÉCOSYSTÈME APPLE --
    L'iPhone 15 est un smartphone haut de gamme fabriqué par Apple.
    L'iPhone 15 utilise un port de charge USB-C.
    L'iPhone 15 est compatible avec le chargeur MagSafe.
    L'iPhone 15 Pro Max est la version grand format fabriquée par Apple.
    L'iPhone 15 Pro Max dispose d'une garantie constructeur spéciale de 2 ans.
    Les AirPods Pro 2 sont des écouteurs sans fil fabriqués par Apple.
    Les AirPods Pro 2 sont compatibles avec tous les iPhone via Bluetooth.
    Le MacBook Air M2 est un ordinateur portable fabriqué par Apple.
    Le MacBook Air M2 dispose de 2 ports USB-C Thunderbolt.
    Le Chargeur Rapide 20W est fabriqué par Apple.
    Le Chargeur Rapide 20W est compatible avec l'iPhone 15 et l'iPhone 14.

    -- ÉCOSYSTÈME SAMSUNG --
    Le Samsung Galaxy S24 Ultra est un smartphone fabriqué par Samsung.
    Le Galaxy S24 Ultra est livré avec un stylet S-Pen.
    Le Galaxy S24 Ultra utilise un port USB-C.
    Les Galaxy Buds 2 Pro sont des écouteurs fabriqués par Samsung.
    Les Galaxy Buds 2 Pro sont compatibles avec le Galaxy S24 et l'iPhone 15 (fonctionnalités limitées sur iPhone).
    La montre Galaxy Watch 6 est fabriquée par Samsung.
    La Galaxy Watch 6 n'est pas compatible avec l'iPhone.

    -- ACCESSOIRES UNIVERSELS & CÂBLES --
    Le Câble USB-C Tressé est un accessoire fabriqué par Belkin.
    Le Câble USB-C Tressé est compatible avec l'iPhone 15, le Galaxy S24 Ultra et le MacBook Air M2.
    Le Câble USB-C Tressé a une garantie à vie.
    La Batterie Externe Anker 10000mAh est compatible avec tous les smartphones USB-C.
    L'Adaptateur Secteur Universel est compatible avec les prises électriques au Maroc et au Sénégal.

    -- ÉCOSYSTÈME GAMING --
    La PlayStation 5 (PS5) est une console de jeux fabriquée par Sony.
    La PS5 est livrée avec une manette DualSense.
    La Manette DualSense est compatible avec la PS5 et le PC.
    Le Casque Pulse 3D est fabriqué par Sony.
    Le Casque Pulse 3D est compatible avec la PS5 et la PS4.
    Les jeux PS5 ne sont pas compatibles avec la PS4.
    Les jeux PS4 sont compatibles avec la PS5 (rétrocompatibilité).
    """
]

def ingest_graph():
    print("🔌 Connexion à Neo4j...")
    graph = Neo4jGraph()

    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "groq":
        print("⚡ Ingestion via GROQ...")
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0
        )
    else:
        print("🤖 Ingestion via GOOGLE...")
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=0
        )
    
    llm_transformer = LLMGraphTransformer(llm=llm)

    print("🧠 Analyse du texte (Gemini)...")
    documents = [Document(page_content=text) for text in raw_text]
    
    # Transformation
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
    except Exception as e:
        print(f"❌ Erreur Extraction LLM: {e}")
        return

    # --- AFFICHAGE DU RÉSUMÉ AVANT INSERTION ---
    nodes = graph_documents[0].nodes
    rels = graph_documents[0].relationships
    
    print(f"\n RÉSUMÉ DE L'EXTRACTION :")
    print(f"   • Nœuds trouvés : {len(nodes)}")
    print(f"   • Relations trouvées : {len(rels)}")
    
    print("\n APERÇU DES RELATIONS (5 premières) :")
    for i, r in enumerate(rels[:5]):
        print(f"   ({r.source.id}) --[{r.type}]--> ({r.target.id})")

    print("\n Sauvegarde dans Neo4j...")
    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()
    
    # --- VÉRIFICATION FINALE ---
    print("\n VÉRIFICATION EN BASE :")
    count_query = "MATCH (n) RETURN count(n) as count"
    result = graph.query(count_query)
    print(f"   🏆 Total Nœuds en base : {result[0]['count']}")
    
    print("\n Ingestion terminée avec succès !")

if __name__ == "__main__":
    ingest_graph()
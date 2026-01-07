class OrderService:
    def __init__(self):
        # Base de données factice
        self.orders = {
            # --- CAS CLASSIQUES : LIVRAISON ---
            "CMD-100": {"status": "En cours de livraison", "date": "2025-05-20", "items": "iPhone 15 Pro Max (256Go)", "city": "Casablanca", "total": "1400€"},
            "CMD-101": {"status": "Livré", "date": "2025-05-18", "items": "Samsung Galaxy S24 Ultra", "city": "Dakar", "total": "1200€"},
            "CMD-102": {"status": "En préparation", "date": "2025-05-21", "items": "MacBook Air M2", "city": "Abidjan", "total": "1100€"},
            "CMD-103": {"status": "Expédié", "date": "2025-05-19", "items": "Sony WH-1000XM5", "city": "Tunis", "total": "350€"},
            "CMD-104": {"status": "Livré", "date": "2025-05-10", "items": "Machine à café Nespresso", "city": "Rabat", "total": "150€"},
            "CMD-105": {"status": "En attente de prise en charge", "date": "2025-05-22", "items": "PS5 Slim Édition Standard", "city": "Alger", "total": "550€"},
            
            # --- CAS GRAPHRAG (Produits liés) ---
            "CMD-200": {"status": "Livré", "date": "2025-04-15", "items": "Câble USB-C Tressé (2m)", "city": "Marrakech", "total": "25€"},
            "CMD-201": {"status": "En cours de livraison", "date": "2025-05-20", "items": "Coque iPhone 15 MagSafe", "city": "Douala", "total": "45€"},
            "CMD-202": {"status": "Livré", "date": "2025-01-10", "items": "Chargeur Apple 20W", "city": "Lomé", "total": "29€"},

            # --- CAS PROBLÉMATIQUES (Annulation, Retours) ---
            "CMD-900": {"status": "Annulé (Rupture de stock)", "date": "2025-05-01", "items": "Carte Graphique RTX 4090", "city": "Paris", "total": "1800€"},
            "CMD-901": {"status": "Annulé (Demande client)", "date": "2025-05-05", "items": "AirPods Pro 2", "city": "Lyon", "total": "279€"},
            "CMD-902": {"status": "Retourné (Remboursé)", "date": "2025-04-20", "items": "Ecran Dell 27 pouces", "city": "Tanger", "total": "300€"},
            "CMD-903": {"status": "Retourné (En attente inspection)", "date": "2025-05-15", "items": "Drone DJI Mini 4", "city": "Agadir", "total": "800€"},
            "CMD-904": {"status": "Paiement Refusé", "date": "2025-05-22", "items": "iPad Air", "city": "Bamako", "total": "700€"},

            # --- CAS COMPLEXES (Douane, Point Relais) ---
            "CMD-300": {"status": "Bloqué en Douane", "date": "2025-05-12", "items": "Laptop Gaming MSI", "city": "Dakar", "total": "2500€"},
            "CMD-301": {"status": "Disponible en Point Relais", "date": "2025-05-21", "items": "Montre Garmin Fenix 7", "city": "Yaoundé", "total": "600€"},
            "CMD-302": {"status": "Adresse incorrecte (Retour expéditeur)", "date": "2025-05-18", "items": "Clavier Mécanique Keychron", "city": "Cotonou", "total": "120€"},

            # --- COMMANDES RÉCENTES (Pour démos "live") ---
            "CMD-123": {"status": "En cours de livraison (Arrivée ce soir)", "date": "2025-05-21", "items": "iPhone 15", "city": "Casablanca", "total": "960€"},
            "CMD-456": {"status": "Livré dans la boîte aux lettres", "date": "2025-05-20", "items": "Livre: 'Apprendre le GraphRAG'", "city": "Fès", "total": "35€"},
            "CMD-789": {"status": "En préparation", "date": "2025-05-22", "items": "Aspirateur Robot Roborock", "city": "Oran", "total": "450€"},
            
            # --- PETITS PANIERS ---
            "CMD-001": {"status": "Livré", "date": "2025-05-10", "items": "Lot de 3 Chaussettes", "city": "Tunis", "total": "15€"},
            "CMD-002": {"status": "Expédié", "date": "2025-05-19", "items": "Carte SD 128Go", "city": "Sfax", "total": "20€"},
            "CMD-003": {"status": "En préparation", "date": "2025-05-21", "items": "Bouteille d'eau réutilisable", "city": "Alger", "total": "10€"},
        }

    def get_order_status(self, order_id: str):
        # Nettoyage basique (si l'utilisateur tape "CMD123" ou "cmd-123")
        clean_id = order_id.upper().replace(" ", "").strip()
        
        if "-" not in clean_id and clean_id.startswith("CMD"):
            clean_id = clean_id.replace("CMD", "CMD-")

        order = self.orders.get(clean_id)
        
        if not order:
            return None
            
        # Réponse riche et formatée
        return (
            f"📦 **Commande {clean_id}**\n"
            f"• **Statut :** {order['status']}\n"
            f"• **Articles :** {order['items']}\n"
            f"• **Destination :** {order['city']}\n"
            f"• **Date :** {order['date']}\n"
            f"• **Total :** {order['total']}"
        )
    
order_service = OrderService()

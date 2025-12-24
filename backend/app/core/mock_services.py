class OrderService:
    def __init__(self):
        # Base de données factice
        self.orders = {
            "CMD-123": {"status": "En cours de livraison", "date": "2023-12-20", "items": "iPhone 15"},
            "CMD-456": {"status": "Livré", "date": "2023-12-10", "items": "Casque Sony"},
            "CMD-999": {"status": "Annulé", "date": "2023-12-01", "items": "Machine à café"}
        }

    def get_order_status(self, order_id: str):
        # Nettoyage basique (si l'utilisateur tape "CMD123" ou "cmd-123")
        clean_id = order_id.upper().replace(" ", "")
        
        order = self.orders.get(clean_id)
        if not order:
            return None
            
        return f"Votre commande {clean_id} ({order['items']}) est actuellement : **{order['status']}** (Date : {order['date']})."

order_service = OrderService()

from backend.app.core.database import get_db_connection

conn = get_db_connection()
cur = conn.cursor()
# On ajoute une colonne 'feedback' qui peut être NULL (pas de vote), 1 (Positif), 0 (Négatif)
cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS feedback INT;")
conn.commit()
print("Base de données mise à jour !")

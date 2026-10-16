
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

connection_settings = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "company_rag"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD"),
}

try:
    with psycopg.connect(**connection_settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            print("PostgreSQL:", cur.fetchone()[0])

            cur.execute(
                "SELECT extversion FROM pg_extension "
                "WHERE extname = 'vector'"
            )
            result = cur.fetchone()

            if result:
                print("pgvector version:", result[0])
            else:
                print("pgvector is not enabled in this database")

    print("Database connection successful!")

except Exception as error:
    print("Database connection failed:", error)

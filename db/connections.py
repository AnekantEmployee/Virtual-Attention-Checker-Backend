import asyncpg
import os


async def get_database_connection():
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    try:
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

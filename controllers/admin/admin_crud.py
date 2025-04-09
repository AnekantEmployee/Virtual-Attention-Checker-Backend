from datetime import datetime

from db import User
from utils.passwords import hash_password
from db.connections import get_database_connection


async def create_admin(user: dict):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        if not await check_admin_by_email(user["email"]):
            raise Exception("Admin with this email already exists")

        await conn.execute(
            "INSERT INTO users (username, password, email, created_at, role) VALUES ($1, $2, $3, $4, $5)",
            user["username"],
            hash_password(user["password"]),
            user["email"].lower(),
            datetime.now(),
            "admin",
        )
        return user

    except Exception as e:
        print(e)
        raise Exception(f"Error creating user: {e}")

    finally:
        if conn:
            await conn.close()


async def check_admin_by_email(user_email: str) -> bool:
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        row = await conn.fetchrow("SELECT 1 FROM users WHERE email=$1", user_email)
        return row is None
    except Exception as e:
        raise Exception(f"Error checking user by email: {e}")
    finally:
        await conn.close()

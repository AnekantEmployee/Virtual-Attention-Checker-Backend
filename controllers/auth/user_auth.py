from utils.passwords import verify_password
from db.connections import get_database_connection


async def user_login(user: dict):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        email_response = await check_validity_by_email(user["email"].lower())
        if email_response:
            if verify_password(user["password"], email_response["password"]):
                email_response.pop("password")
                return email_response

        raise Exception("Invalid Credentials")

    except Exception as e:
        raise Exception(f"Error logging in user: {e}")

    finally:
        if conn:
            await conn.close()


async def check_validity_by_email(user_email: str) -> bool:
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        # User check
        row = await conn.fetchrow("SELECT * FROM users WHERE email=$1", user_email)
        if row:
            return dict(row)

        return None
    except Exception as e:
        raise Exception(f"Error checking user by email: {e}")
    finally:
        await conn.close()

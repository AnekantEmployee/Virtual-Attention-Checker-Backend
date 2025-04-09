from datetime import datetime

from db import User
from utils.passwords import hash_password
from db.connections import get_database_connection


async def create_user(user: User):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        if not await check_user_by_email(user["email"]):
            raise Exception("User with this email already exists")

        await conn.execute(
            "INSERT INTO users (username, password, email, created_at, role) VALUES ($1, $2, $3, $4, $5)",
            user["username"],
            hash_password(user["password"]),
            user["email"].lower(),
            datetime.now(),
            "user",
        )
        return user

    except Exception as e:
        raise Exception(f"Error creating user: {e}")

    finally:
        if conn:
            await conn.close()


async def read_users():
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        rows = await conn.fetch("SELECT * FROM users")
        user_data = []
        for user in rows:
            user = dict(user)
            user.pop("password")
            if user["role"] == "user":
                user_data.append(user)

        return user_data

    except Exception as e:
        raise Exception(f"Error reading users: {e}")

    finally:
        await conn.close()


async def update_images(user_id: int, user_images: dict):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        if not await get_user_by_id(user_id):
            raise Exception("User not found")

        # Check if the user exists
        row = await conn.fetchrow(
            "SELECT * FROM userimages WHERE user_id = $1", user_id
        )
        if not row:
            # Update the user in the database
            await conn.execute(
                """
                INSERT INTO userimages
                (user_id, image_url1, image_url2, image_url3, image_url4, image_url5, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                user_id,
                user_images["image_url1"],
                user_images["image_url2"],
                user_images["image_url3"],
                user_images["image_url4"],
                user_images["image_url5"],
                datetime.now(),
            )
            return user_images

        # Update the user in the database
        await conn.execute(
            """
            UPDATE userimages
            SET image_url1 = $2, image_url2 = $3, image_url3 = $4, image_url4 = $5, image_url5 = $6
            WHERE user_id = $1
            """,
            user_id,
            user_images["image_url1"],
            user_images["image_url2"],
            user_images["image_url3"],
            user_images["image_url4"],
            user_images["image_url5"],
        )
        return user_images
    except Exception as e:
        raise Exception(f"Error updating user: {e}")
    finally:
        if conn:
            await conn.close()


async def get_user_by_id(user_id: int):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        row = await conn.fetchrow("SELECT * FROM users WHERE id=$1", user_id)
        if row and row["role"] == "user":
            row = dict(row)
            row.pop("password")
            return row
        else:
            raise Exception("User not found")

    except Exception as e:
        raise Exception(f"Error fetching user: {e}")

    finally:
        if conn:
            await conn.close()


async def get_user_details_user_side(user_id: int):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        user_data = await get_user_by_id(user_id)
        if user_data:
            user_images = await conn.fetchrow(
                "SELECT * FROM userimages WHERE user_id=$1", user_id
            )
            if user_images:
                user_images = dict(user_images)
                user_images.pop("id")
                user_images.pop("user_id")
                user_data["images"] = user_images
                return user_data

            return user_data
        else:
            raise Exception("User not found")

    except Exception as e:
        raise Exception(f"Error fetching user: {e}")

    finally:
        if conn:
            await conn.close()


async def check_user_by_email(user_email: str) -> bool:
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


async def update_user(user_id: int, user: User):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        # Check if the user exists
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        if not row:
            raise Exception("User not found")

        # Update the user in the database
        await conn.execute(
            """
            UPDATE users
            SET username = $1, email = $2, password = $3
            WHERE id = $4
            """,
            user["username"],
            user["email"].lower(),
            hash_password(user["password"]),
            user_id,
        )
        return user
    except Exception as e:
        raise Exception(f"Error updating user: {e}")
    finally:
        if conn:
            await conn.close()


async def delete_user(user_id: int):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        if not await get_user_by_id(user_id):
            raise Exception("User Doesn't Exists in the Db")

        await conn.execute("DELETE FROM users WHERE id=$1", user_id)
        return {"detail": "User Deleted Successfully"}
    except Exception as e:
        raise Exception(f"Error deleting user: {e}")

    finally:
        if conn:
            await conn.close()

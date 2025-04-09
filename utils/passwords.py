import bcrypt


def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()  # Generate a salt
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)  # Hash the password
    return hashed_password.decode("utf-8")  # Return the hashed password as a string.


def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a password against a bcrypt hash."""
    hashed_password_bytes = hashed_password.encode("utf-8")
    password_bytes = password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)

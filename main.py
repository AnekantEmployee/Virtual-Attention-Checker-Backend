from fastapi import FastAPI
from dotenv import load_dotenv

from routes import user_routes
from routes import auth_routes
from routes import admin_routes
from routes import meeting_routes


load_dotenv()

app = FastAPI()


app.include_router(auth_routes, prefix="/api/auth", tags=["auth"])
app.include_router(user_routes, prefix="/api/users", tags=["users"])
app.include_router(admin_routes, prefix="/api/admin", tags=["admin"])
app.include_router(meeting_routes, prefix="/api/meetings", tags=["meetings"])


# uvicorn main:app --reload --host localhost --port 5000

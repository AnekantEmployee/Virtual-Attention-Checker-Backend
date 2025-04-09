from fastapi import APIRouter, HTTPException

from db import User
from controllers import create_admin


router = APIRouter()


@router.post("/create", response_model=User)
async def create_admin_endpoint(user: dict):
    try:
        response = await create_admin(dict(user))
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi import APIRouter, HTTPException

from controllers import user_login


router = APIRouter()


@router.post("/login", response_model=dict)
async def login_user(user: dict):
    try:
        response = await user_login(user)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

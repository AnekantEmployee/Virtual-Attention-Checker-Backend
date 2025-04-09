from fastapi import APIRouter, HTTPException

from db import User
from controllers import read_users
from controllers import update_user
from controllers import delete_user
from controllers import create_user
from controllers import update_images
from controllers import get_user_by_id
from controllers import get_user_details_user_side


router = APIRouter()


@router.post("/create", response_model=dict)
async def create_user_endpoint(user: dict):
    try:
        response = await create_user(dict(user))
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get-all", response_model=list[dict])
async def get_all_users():
    try:
        return await read_users()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get", response_model=dict)
async def get_user(user_id: int):
    try:
        user = await get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/full-details", response_model=dict)
async def get_full_user_details(user_id: int):
    try:
        user = await get_user_details_user_side(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/update", response_model=dict)
async def update_user_endpoint(user_id: int, user: dict):
    try:
        return await update_user(user_id, dict(user))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/update-images", response_model=dict)
async def update_images_employee(user_id: int, user_images: dict):
    try:
        return await update_images(user_id, dict(user_images))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/delete")
async def delete_user_endpoint(user_id: int):
    try:
        return await delete_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

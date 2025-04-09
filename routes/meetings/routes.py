from fastapi import APIRouter, HTTPException

from db import Meeting
from controllers import get_meeting
from controllers import read_meetings
from controllers import update_meeting
from controllers import delete_meeting
from controllers import create_meeting


router = APIRouter()


@router.post("/create", response_model=Meeting)
async def create_meeting_endpoint(meeting: Meeting):
    try:
        return await create_meeting(meeting)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get-all", response_model=list[dict])
async def read_meetings_endpoint():
    try:
        return await read_meetings()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get", response_model=dict)
async def get_single_meeting(meeting_id: int) -> dict:
    try:
        return await get_meeting(meeting_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/update", response_model=Meeting)
async def update_meeting_endpoint(meeting_id: int, meeting: Meeting):
    try:
        return await update_meeting(meeting_id, meeting)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/delete")
async def delete_meeting_endpoint(meeting_id: int):
    try:
        return await delete_meeting(meeting_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

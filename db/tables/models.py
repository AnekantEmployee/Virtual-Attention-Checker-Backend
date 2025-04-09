from typing import Optional
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field


class User(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    email: str
    password: str
    role: str
    created_at: Optional[datetime] = Field(default=datetime.now())


class Meeting(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str
    start_time: datetime
    meeting_link: HttpUrl
    admin_id: int
    attendees: list[int]
    created_at: Optional[datetime] = Field(default=datetime.now())


class UserImage(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int
    image_url1: HttpUrl
    image_url2: HttpUrl
    image_url3: HttpUrl
    image_url4: HttpUrl
    image_url5: HttpUrl
    created_at: Optional[datetime] = Field(default=datetime.now())


class MeetingAttendee(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int
    meeting_id: int

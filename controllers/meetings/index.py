from datetime import datetime

from db import Meeting
from .attendee import add_attendees
from .attendee import get_attendees
from .attendee import update_attendees
from db.connections import get_database_connection


async def create_meeting(meeting: Meeting):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        if meeting.created_at is None:
            meeting.created_at = datetime.now()

        response = await conn.fetchrow(
            "INSERT INTO meetings (title, description, start_time, meeting_link, admin_id, created_at) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
            meeting.title,
            meeting.description,
            meeting.start_time,
            str(meeting.meeting_link),
            int(meeting.admin_id),
            meeting.created_at,
        )
        meeting_id = list(dict(response).values())[0]
        attendee_response = await add_attendees(meeting_id, meeting.attendees)
        return meeting

    except Exception as e:
        print(e)
        raise Exception(f"Error creating meeting: {e}")

    finally:
        if conn:
            await conn.close()


async def read_meetings():
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        rows = await conn.fetch("SELECT * FROM meetings")
        rows = [dict(row) for row in rows]
        for row in rows:
            attendee_list = await get_attendees(row["id"])
            row["attendee_list"] = attendee_list

        return rows

    except Exception as e:
        raise Exception(f"Error reading meetings: {e}")

    finally:
        if conn:
            await conn.close()


async def get_meeting(meeting_id: int):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        meeting = await conn.fetchrow("SELECT * FROM meetings WHERE id=$1", meeting_id)
        if not meeting:
            raise Exception(f"Meeting with id {meeting_id} not found")

        meeting = dict(meeting)
        attendee_list = await get_attendees(meeting_id)
        meeting["attendee_list"] = attendee_list
        return meeting

    except Exception as e:
        raise Exception(f"Error updating meeting: {e}")

    finally:
        if conn:
            await conn.close()


async def update_meeting(meeting_id: int, meeting: Meeting):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        meeting_link_str = str(meeting.meeting_link)
        await update_attendees(meeting_id, meeting.attendees)
        await conn.execute(
            "UPDATE meetings SET title=$1, description=$2, start_time=$3, meeting_link=$4, admin_id=$6 WHERE id=$5",
            meeting.title,
            meeting.description,
            meeting.start_time,
            meeting_link_str,
            meeting_id,
            meeting.admin_id,
        )
        return meeting

    except Exception as e:
        raise Exception(f"Error updating meeting: {e}")

    finally:
        if conn:
            await conn.close()


async def delete_meeting(meeting_id: int):
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        if not await get_meeting(meeting_id):
            raise Exception("Meeting Doesn't Exists in the Db")

        await conn.execute("DELETE FROM meetings WHERE id=$1", meeting_id)
        return {"detail": "Meeting Deleted Successfully"}

    except Exception as e:
        raise Exception(f"Error deleting meeting: {e}")

    finally:
        if conn:
            await conn.close()

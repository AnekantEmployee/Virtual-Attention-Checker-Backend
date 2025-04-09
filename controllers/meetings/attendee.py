import traceback
from db.connections import get_database_connection


async def add_attendees(meeting_id: int, attendees: list[int]) -> None:
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        list_of_attendees = [(user_id, meeting_id) for user_id in attendees]

        await conn.executemany(
            "INSERT INTO meetingattendees (user_id, meeting_id) VALUES ($1, $2)",
            list_of_attendees,
        )
        return attendees
    except Exception as e:
        print(e)
        raise Exception(f"Error adding attendees: {e}")

    finally:
        if conn:
            await conn.close()


async def get_attendees(meeting_id: int) -> list[int]:
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        atttendee_data = await conn.fetch(
            "SELECT * FROM meetingattendees WHERE meeting_id = $1",
            meeting_id,
        )
        return [dict(attendee)["user_id"] for attendee in atttendee_data]
    except Exception as e:
        print(e)
        raise Exception(f"Error adding attendees: {e}")

    finally:
        if conn:
            await conn.close()


async def update_attendees(meeting_id: int, attendees: list[int]) -> None:
    conn = await get_database_connection()
    if conn is None:
        raise Exception("Database connection failed")
    try:
        list_of_attendees = [(user_id, meeting_id) for user_id in attendees]
        user_ids = ", ".join([str(attendee) for attendee in attendees])

        await conn.execute(
            f"""DELETE FROM meetingattendees WHERE user_id NOT IN ({user_ids}) AND meeting_id = {meeting_id}""",
        )

        await conn.executemany(
            """INSERT INTO meetingattendees (user_id, meeting_id) VALUES ($1, $2) ON CONFLICT (user_id, meeting_id) DO NOTHING""",
            list_of_attendees,
        )

        return attendees
    except Exception as e:
        print(traceback.print_exception(e))
        raise Exception(f"Error adding attendees: {e}")

    finally:
        if conn:
            await conn.close()

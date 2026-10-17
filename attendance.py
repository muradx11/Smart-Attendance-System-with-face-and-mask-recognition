from datetime import datetime
from database import get_connection


# -------------------------
# Start Attendance Session
# -------------------------

def start_session(total_students):

    connection = get_connection()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor = connection.execute(
        """
        INSERT INTO sessions
        (start_time, total_students, present_count, status)
        VALUES (?, ?, ?, ?)
        """,
        (start_time, total_students, 0, "running")
    )

    session_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return session_id


# -------------------------
# Mark Attendance
# -------------------------

def mark_attendance(enrollment, session_id):

    connection = get_connection()

    # Find student
    student = connection.execute(
        """
        SELECT id, name, roll_number
        FROM students
        WHERE roll_number = ?
        """,
        (enrollment,)
    ).fetchone()

    if student is None:
        connection.close()
        return False, "Student not found"

    # Check duplicate attendance in same session
    existing = connection.execute(
        """
        SELECT id
        FROM attendance
        WHERE student_id = ?
        AND session_id = ?
        """,
        (student["id"], session_id)
    ).fetchone()

    if existing:
        connection.close()
        return False, "Attendance already marked"

    # Current date/time
    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Insert attendance
    connection.execute(
        """
        INSERT INTO attendance
        (student_id, date, time, status, session_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            student["id"],
            date,
            time,
            "Present",
            session_id
        )
    )

    # Update present count
    connection.execute(
        """
        UPDATE sessions
        SET present_count = present_count + 1
        WHERE id = ?
        """,
        (session_id,)
    )

    connection.commit()
    connection.close()

    return True, f"{student['name']} marked Present"


# -------------------------
# Finish Session
# -------------------------

def finish_session(session_id):

    connection = get_connection()

    end_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    connection.execute(
        """
        UPDATE sessions
        SET end_time = ?,
            status = ?
        WHERE id = ?
        """,
        (
            end_time,
            "completed",
            session_id
        )
    )

    connection.commit()
    connection.close()


# -------------------------
# Get Session Attendance
# -------------------------

def get_session_attendance(session_id):

    connection = get_connection()

    records = connection.execute(
        """
        SELECT
            students.roll_number,
            students.name,
            attendance.date,
            attendance.time,
            attendance.status
        FROM attendance
        JOIN students
        ON attendance.student_id = students.id
        WHERE attendance.session_id = ?
        ORDER BY students.id
        """,
        (session_id,)
    ).fetchall()

    connection.close()

    return records


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":

    session_id = start_session(63)

    print("Session started:", session_id)

    success, message = mark_attendance(
        "2400101460",
        session_id
    )

    print(message)

    success, message = mark_attendance(
        "2400101460",
        session_id
    )

    print(message)

    records = get_session_attendance(session_id)

    print("\nAttendance:")

    for record in records:
        print(dict(record))

    finish_session(session_id)

    print("\nSession completed.")
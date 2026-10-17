import sqlite3

DATABASE = "attendance.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    # -------------------------
    # Students table
    # -------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            face_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -------------------------
    # Attendance table
    # -------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            session_id INTEGER,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    # -------------------------
    # Sessions table
    # -------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            duration INTEGER,
            total_students INTEGER DEFAULT 0,
            present_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending'
        )
    """)

    connection.commit()
    connection.close()


def add_student(roll_number, name, face_data=None):

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO students
            (roll_number, name, face_data)
            VALUES (?, ?, ?)
            """,
            (roll_number, name, face_data)
        )

        connection.commit()

    except sqlite3.IntegrityError:

        print(
            f"Student with roll number "
            f"{roll_number} already exists."
        )

    finally:
        connection.close()


def get_all_students():

    connection = get_connection()

    students = connection.execute(
        """
        SELECT *
        FROM students
        ORDER BY serial_no
        """
    ).fetchall()

    connection.close()

    return students


if __name__ == "__main__":

    initialize_database()

    print("Database initialized successfully.")
    if __name__ == "__main__":
     initialize_database()

    students = get_all_students()

    print("\nTotal students in database:", len(students))
    print("\nStudents:\n")

    for index, student in enumerate(students, start=1):
        print(
            f"{index}. "
            f"{student['roll_number']} | "
            f"{student['name']} | "
            f"Face Data: "
            f"{student['face_data'] is not None}"
        )
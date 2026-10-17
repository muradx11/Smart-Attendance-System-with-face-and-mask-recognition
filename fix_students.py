import sqlite3

DATABASE = "attendance.db"


# FINAL CORRECT 63 STUDENTS
correct_students = [
    ("2400104387", "AQUIB AFTAB"),
    ("2400102142", "ASAD RAZA"),
    ("2400104402", "AUSAF ABID"),
    ("2400104329", "DILSHAD ALAM"),
    ("2400100900", "FARIS HALIM"),
    ("2400104293", "MD SAJID"),
    ("2400104747", "MOHAMMAD REHAN"),
    ("2400102113", "MOHAMMAD WAISH"),
    ("2400101782", "MOHAMMAD YASIR ZIYA"),
    ("2400101652", "MOHAMMAD YUSUF"),
    ("2400101031", "MOHAMMAD YUSUF"),
    ("2400101352", "MOHAMMED ANAS SIDDIQUI"),
    ("2400102620", "MOHAMMED ARQAM NADEEM"),
    ("2400100589", "MOHAMMED AYAN KHAN"),
    ("2400100554", "MOHD ALI"),
    ("2400101620", "MOHD ALIMUL HASAN ABIDI"),
    ("2400101127", "MOHD ALTAMASH KHAN"),
    ("2400100727", "MOHD AMAAN"),
    ("2400102879", "MOHD ANADIL RAZA"),
    ("2400103302", "MOHD ARSHAD KHAN"),
    ("2400100640", "MOHD AYAN KHAN"),
    ("2400102370", "MOHD DANIYAL"),
    ("2400102197", "MOHD FAHAD KHAN"),
    ("2400101308", "MOHD FAIZ KHAN"),
    ("2400103540", "MOHD FARAZ"),
    ("2400100373", "MOHD FARHAN"),
    ("2400101075", "MOHD HAMZA"),
    ("2400101776", "MOHD HAMZA HASHMI"),
    ("2400100781", "MOHD JASIM HUSAIN ALI KHAN"),
    ("2400100091", "MOHD KAZIM"),
    ("2400100756", "MOHD MAAZ ARIF"),
    ("2400100628", "MOHD MOHSIN"),
    ("2400103014", "MOHD MOHSIN"),
    ("2400100274", "MOHD NADEEM"),
    ("2400102471", "MOHD NASIR"),
    ("2400101721", "MOHD SAMEER ANSARI"),
    ("2400101805", "MOHD SHAD"),
    ("2400100703", "MOHD SHAFIUDDIN"),
    ("2400102880", "MOHD SUHAIL ANSARI"),
    ("2400101382", "MOHD UZAIR"),
    ("2400100447", "MOHD WAQUAS"),
    ("2400101986", "MOHD ZAID"),
    ("2400100329", "MOHD ZUBAIR"),
    ("2400103939", "MOHD HAMZA KHAN"),
    ("2400101584", "MUAZ ABDUL RAB"),
    ("2400101460", "MURAD KHAN"),
    ("2400100269", "MUSKAN SINGH"),
    ("2400101033", "NABIHA WASIM"),
    ("2400101859", "NABIRA AFAQUE RIZWI"),
    ("2400103715", "NAIMA BASHIR"),
    ("2400100417", "NAVED AHMAD"),
    ("2400101658", "NAWAAL AYUB"),
    ("2400102568", "NEHAL ZAFAR"),
    ("2400102373", "NEMAL KHAN"),
    ("2400100798", "SAIF ALI"),
    ("2400102141", "SAJID ALI"),
    ("2400102230", "SAMEER MAHMOOD"),
    ("2400104298", "YASHIKA RAAJ"),
    ("2500104789", "ASIF AHMAD"),
    ("2200100851", "KAMRAN KHAN"),
    ("2200101134", "MD AKRAMULLAH"),
    ("2500103458", "MD SHAHID ALI"),
    ("2200104511B", "ZAID ABRAR"),
]


def fix_students():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    # Add serial_no column if it does not already exist
    cursor.execute("PRAGMA table_info(students)")
    columns = [column[1] for column in cursor.fetchall()]

    if "serial_no" not in columns:
        cursor.execute(
            """
            ALTER TABLE students
            ADD COLUMN serial_no INTEGER
            """
        )

    correct_enrollments = [
        enrollment
        for enrollment, name in correct_students
    ]

    # Update existing students / insert missing students
    for serial_no, (enrollment, name) in enumerate(
        correct_students,
        start=1
    ):

        cursor.execute(
            """
            SELECT id
            FROM students
            WHERE roll_number = ?
            """,
            (enrollment,)
        )

        student = cursor.fetchone()

        if student:

            cursor.execute(
                """
                UPDATE students
                SET
                    name = ?,
                    serial_no = ?
                WHERE roll_number = ?
                """,
                (name, serial_no, enrollment)
            )

        else:

            cursor.execute(
                """
                INSERT INTO students
                (roll_number, name, serial_no)
                VALUES (?, ?, ?)
                """,
                (enrollment, name, serial_no)
            )

    # Delete extra students
    placeholders = ",".join(
        ["?"] * len(correct_enrollments)
    )

    cursor.execute(
        f"""
        DELETE FROM students
        WHERE roll_number NOT IN ({placeholders})
        """,
        correct_enrollments
    )

    deleted_count = cursor.rowcount

    connection.commit()

    # Verify
    cursor.execute(
        """
        SELECT
            serial_no,
            roll_number,
            name,
            face_data
        FROM students
        ORDER BY serial_no
        """
    )

    students = cursor.fetchall()

    connection.close()

    print()
    print("========================================")
    print("       STUDENT DATABASE FIXED")
    print("========================================")
    print()

    print("Deleted extra students:", deleted_count)
    print("Final student count:", len(students))

    print()
    print("Students:")
    print()

    for student in students:

        face_status = (
            "TRUE"
            if student[3] is not None
            else "FALSE"
        )

        print(
            f"{student[0]}. "
            f"{student[1]} | "
            f"{student[2]} | "
            f"Face Data: {face_status}"
        )

    print()

    if len(students) == 63:
        print("SUCCESS: Exactly 63 students.")
    else:
        print(
            f"WARNING: Found {len(students)} students."
        )

    print()


if __name__ == "__main__":
    fix_students()
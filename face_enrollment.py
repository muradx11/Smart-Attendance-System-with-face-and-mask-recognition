import cv2
import json

from database import get_connection

import recognition


# ============================================================
# SETTINGS
# ============================================================

# Maximum allowed head tilt (up/down) for a capture to be
# allowed. Kept in sync with recognition.MAX_RECOGNITION_PITCH
# so enrollment and recognition agree on "looking straight".
MAX_ENROLLMENT_PITCH = 12

# Maximum allowed head turn (left/right) for a capture.
MAX_ENROLLMENT_YAW = 12

# Guide box size, as a fraction of the frame width/height.
GUIDE_BOX_WIDTH_RATIO = 0.35
GUIDE_BOX_HEIGHT_RATIO = 0.55


def draw_guide_box(frame):

    height, width = frame.shape[:2]

    box_width = int(width * GUIDE_BOX_WIDTH_RATIO)
    box_height = int(height * GUIDE_BOX_HEIGHT_RATIO)

    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    return x1, y1, x2, y2


def save_face_data(enrollment, encoding):

    connection = get_connection()

    connection.execute(
        """
        UPDATE students
        SET face_data = ?
        WHERE roll_number = ?
        """,
        (
            json.dumps(encoding.flatten().tolist()),
            enrollment
        )
    )

    connection.commit()
    connection.close()


def enroll_student(enrollment):

    camera = cv2.VideoCapture(0)

    print()
    print("Face Enrollment")
    print("----------------")
    print(f"Enrollment: {enrollment}")
    print()
    print("Camera starting...")
    print("Keep your face inside the box and look directly at the camera.")
    print("Press SPACE to capture.")
    print("Press Q to cancel.")

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Mirror the feed (same as app.py) so enrollment feels
        # natural - move left, see yourself move left.
        frame = cv2.flip(frame, 1)

        # Reuses recognition.py's MediaPipe landmark extraction,
        # guide box helpers, and pose (pitch/yaw) computation -
        # same functions recognition.py itself uses, so "looking
        # straight" means the same thing in both places.

        current_face, pitch_angle, yaw_angle = recognition.get_face_data(frame)

        box = draw_guide_box(frame)
        box_x1, box_y1, box_x2, box_y2 = box

        frame_height, frame_width = frame.shape[:2]

        in_box = False
        face_bounds = None

        if current_face is not None:

            face_bounds = recognition.get_face_bounds(
                current_face,
                frame_width,
                frame_height
            )

            in_box = recognition.is_face_within_box(
                face_bounds,
                box
            )

        pitch_ok = (
            pitch_angle is not None
            and abs(pitch_angle) <= MAX_ENROLLMENT_PITCH
        )

        yaw_ok = (
            yaw_angle is not None
            and abs(yaw_angle) <= MAX_ENROLLMENT_YAW
        )

        ready_to_capture = (
            current_face is not None
            and in_box
            and pitch_ok
            and yaw_ok
        )

        if ready_to_capture:
            box_color = (0, 255, 0)
        elif current_face is not None:
            box_color = (0, 165, 255)
        else:
            box_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (box_x1, box_y1),
            (box_x2, box_y2),
            box_color,
            2
        )

        if ready_to_capture:

            cv2.putText(
                frame,
                "FACE DETECTED - Press SPACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif current_face is not None and not in_box:

            cv2.putText(
                frame,
                "MOVE YOUR FACE INTO THE BOX",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )

        elif current_face is not None and not pitch_ok:

            cv2.putText(
                frame,
                "LOOK MORE DIRECTLY AT THE CAMERA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {recognition.format_value(pitch_angle)} "
                f"(max {MAX_ENROLLMENT_PITCH})",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

        elif current_face is not None and not yaw_ok:

            cv2.putText(
                frame,
                "TURN TO FACE THE CAMERA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                f"Yaw: {recognition.format_value(yaw_angle)} "
                f"(max {MAX_ENROLLMENT_YAW})",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Face Enrollment",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32 and ready_to_capture:

            encoding = recognition.get_face_encoding(frame, face_bounds)

            if encoding is None:

                print()
                print("Could not compute a face encoding - try again.")
                continue

            save_face_data(
                enrollment,
                encoding
            )

            print()
            print("Face enrolled successfully!")
            print(f"Enrollment: {enrollment}")

            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    enrollment = input(
        "Enter student enrollment: "
    ).strip()

    enroll_student(enrollment)
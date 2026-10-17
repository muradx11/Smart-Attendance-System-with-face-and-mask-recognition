import threading
from io import BytesIO
from collections import deque

import cv2
import openpyxl
from openpyxl.styles import Font
from flask import Flask, render_template, Response, jsonify, send_file

from database import initialize_database, get_all_students
from attendance import (
    start_session,
    finish_session,
    mark_attendance,
    get_session_attendance,
)

import recognition


app = Flask(__name__)

initialize_database()

# ============================================================
# Load once at startup (per requirement: "MediaPipe landmarker
# is initialized once" / "enrolled faces are loaded once").
# recognition.py's landmarker is already created at import
# time inside that module - importing it here does that.
# ============================================================

enrolled_faces = recognition.load_enrolled_faces()


# ============================================================
# Simple global session state.
#
# This is intentionally a plain dict + one lock instead of
# Redis/Celery/etc - fine for a single-camera college project.
# It holds everything recognition.py's main() used to keep as
# local variables, so the SAME recognition logic can run
# per-frame inside the Flask video stream.
# ============================================================

session_lock = threading.Lock()

session_state = {
    "active": False,
    "session_id": None,
    "total_students": 0,
    "marked_students": set(),
    "candidate_enrollment": None,
    "candidate_count": 0,
    "confirmed_student": None,
    # Set when a session ENDS (not when it starts). Stays set even
    # after a new session begins, so "Export Last Session" always
    # has something to download until the NEXT session also ends.
    "last_completed_session_id": None,
    # Motion-based liveness (informational only, never blocks).
    "previous_landmarks": None,
    "motion_scores": deque(maxlen=recognition.MOTION_WINDOW_SIZE),
}

# ============================================================
# Latest live recognition status, for the "Detected Student" /
# "Mask" / "Confidence" panels on the dashboard. This exists
# because the on-screen text burned into the video frame gets
# cropped by the camera box's CSS (object-fit: cover) - this
# JSON endpoint is not, so it's a reliable way to actually see
# what's being detected in real time.
# ============================================================

latest_status = {
    "state": "idle",
    "name": None,
    "enrollment": None,
    "distance": None,
    "gap": None,
    "candidate_count": 0,
    "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
    "mask_detected": None,
    "mask_color_distance": None,
    "mask_texture_std": None,
    "liveness_state": "idle",
    "motion_score": None,
}


def reset_confirmation_state():

    session_state["candidate_enrollment"] = None
    session_state["candidate_count"] = 0
    session_state["confirmed_student"] = None


def mask_status_fields(mask_info):
    """Small helper so every latest_status.update() call includes
    the mask fields consistently, whether or not a face/mask
    reading is currently available."""

    if mask_info is None:
        return {
            "mask_detected": None,
            "mask_color_distance": None,
            "mask_texture_std": None,
        }

    return {
        "mask_detected": mask_info["mask_detected"],
        "mask_color_distance": mask_info["color_distance"],
        "mask_texture_std": mask_info["texture_std"],
    }


def update_motion_liveness(current_face):
    """Updates the rolling motion window from the current frame's
    landmarks (or clears it if there's no face this frame) and
    returns (liveness_state, latest_motion_score). Purely
    informational/gating - never affects recognition or
    attendance directly, only the confirmation-counter gate.
    Must be called with session_lock already held."""

    if current_face is None:

        session_state["motion_scores"].clear()
        session_state["previous_landmarks"] = None

        return "idle", None

    motion_score = recognition.compute_motion_score(
        current_face,
        session_state["previous_landmarks"]
    )

    if motion_score is not None:
        session_state["motion_scores"].append(motion_score)

    session_state["previous_landmarks"] = current_face

    scores = session_state["motion_scores"]

    if len(scores) < 5:
        return "checking", (max(scores) if scores else None)

    elevated_count = sum(
        1 for score in scores
        if score >= recognition.MOTION_LIVE_THRESHOLD
    )

    liveness_state = (
        "live"
        if elevated_count >= recognition.MOTION_MIN_ELEVATED_FRAMES
        else "static"
    )

    return liveness_state, max(scores)


def liveness_status_fields(liveness_state, motion_score):
    """Matches the mask_status_fields() pattern for consistency."""

    return {
        "liveness_state": liveness_state,
        "motion_score": motion_score,
    }


def draw_mask_indicator(frame, mask_info, frame_height):
    """Draws a small, always-present "Mask: ON/OFF" line near the
    bottom-left of the frame. Purely informational - never
    affects recognition or attendance."""

    if mask_info is None:
        return

    if mask_info["mask_detected"]:
        text = "Mask: ON"
        color = (0, 200, 200)
    else:
        text = "Mask: OFF"
        color = (200, 200, 200)

    cv2.putText(
        frame, text, (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )


# ============================================================
# Per-frame recognition processing
#
# This mirrors the loop body of recognition.py's main(), but
# reads/writes session_state instead of local variables, and
# only marks attendance once per student per session (via
# marked_students) instead of relying solely on a per-streak
# flag.
# ============================================================

def process_recognition(frame):

    current_face, pitch_angle, yaw_angle = recognition.get_face_data(frame)

    frame_height, frame_width = frame.shape[:2]

    box = recognition.draw_guide_box(frame)
    box_x1, box_y1, box_x2, box_y2 = box

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

    # Identity now comes from the SFace deep embedding, not the
    # raw landmarks - needs face_bounds (computed above) to know
    # what to crop.
    current_encoding = recognition.get_face_encoding(frame, face_bounds)

    student = recognition.recognize_face(
        current_encoding,
        enrolled_faces
    )

    pitch_ok = (
        pitch_angle is not None
        and abs(pitch_angle) <= recognition.MAX_RECOGNITION_PITCH
    )

    yaw_ok = (
        yaw_angle is not None
        and abs(yaw_angle) <= recognition.MAX_RECOGNITION_YAW
    )

    position_ok = (
        current_face is not None
        and in_box
        and pitch_ok
        and yaw_ok
    )

    if position_ok:
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

    pitch_text = (
        recognition.format_value(pitch_angle)
        if pitch_angle is not None
        else "n/a"
    )

    yaw_text = (
        recognition.format_value(yaw_angle)
        if yaw_angle is not None
        else "n/a"
    )

    # ------------------------------------------------------------
    # Mask detection - informational only, runs independently of
    # position_ok/accepted/rejected. Never gates recognition or
    # attendance in any way.
    # ------------------------------------------------------------

    mask_info = recognition.detect_mask(frame, face_bounds)
    draw_mask_indicator(frame, mask_info, frame_height)

    with session_lock:

        liveness_state, motion_score = update_motion_liveness(current_face)

        if student:

            current_enrollment = student["enrollment"]

            if position_ok:

                if student["accepted"] and current_enrollment in session_state["marked_students"]:

                    # Already marked earlier this session - no need
                    # to make them wait through the confirmation
                    # countdown again, just tell them plainly so
                    # they know they can step away.
                    reset_confirmation_state()

                    cv2.putText(
                        frame, "ALREADY MARKED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2
                    )

                    cv2.putText(
                        frame,
                        f"{student['name']} ({student['enrollment']})",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2
                    )

                    if face_bounds is not None:

                        label_x = int(face_bounds[2]) + 15
                        label_x = min(label_x, max(frame_width - 230, 0))

                        label_y = int(face_bounds[1]) + 30

                        cv2.putText(
                            frame,
                            student["name"],
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
                        )

                        cv2.putText(
                            frame,
                            "ALREADY MARKED",
                            (label_x, label_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2
                        )

                    latest_status.update({
                        "state": "already_marked",
                        "name": student["name"],
                        "enrollment": student["enrollment"],
                        "distance": student["distance"],
                        "gap": student["gap"],
                        "candidate_count": 0,
                        "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                        **mask_status_fields(mask_info),
                        **liveness_status_fields(liveness_state, motion_score),
                    })

                elif student["accepted"]:

                    if liveness_state == "live":

                        # ------------------------------------------------
                        # 15/25-frame identity confirmation.
                        # ------------------------------------------------

                        if current_enrollment == session_state["candidate_enrollment"]:

                            session_state["candidate_count"] += 1

                        else:

                            session_state["candidate_enrollment"] = current_enrollment
                            session_state["candidate_count"] = 1
                            session_state["confirmed_student"] = None

                        if session_state["candidate_count"] >= recognition.REQUIRED_CONFIRMATIONS:

                            session_state["confirmed_student"] = student

                            # Only mark attendance once per student per
                            # session, no matter how many frames stay
                            # confirmed or how many times they re-approach.
                            if current_enrollment not in session_state["marked_students"]:

                                success_mark, message = mark_attendance(
                                    current_enrollment,
                                    session_state["session_id"]
                                )

                                if success_mark:
                                    session_state["marked_students"].add(current_enrollment)
                                    print(message)

                        is_recognized = (
                            session_state["confirmed_student"]
                            and session_state["candidate_count"] >= recognition.REQUIRED_CONFIRMATIONS
                        )

                        if is_recognized:

                            cv2.putText(
                                frame, "RECOGNIZED", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                            )

                            cv2.putText(
                                frame,
                                f"{student['name']} ({student['enrollment']})",
                                (20, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
                            )

                        else:

                            cv2.putText(
                                frame, "VERIFYING...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                            )

                            cv2.putText(
                                frame,
                                f"{student['name']} ({student['enrollment']})",
                                (20, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                            )

                            cv2.putText(
                                frame,
                                f"{recognition.REQUIRED_CONFIRMATIONS}-frame confirmation: "
                                f"{session_state['candidate_count']}/{recognition.REQUIRED_CONFIRMATIONS}",
                                (20, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                            )

                        cv2.putText(
                            frame,
                            f"Distance: {recognition.format_value(student['distance'])}",
                            (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

                        cv2.putText(
                            frame,
                            f"Gap: {recognition.format_value(student['gap'])}",
                            (20, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

                        # ------------------------------------------------
                        # Face-anchored label: name + status printed to
                        # the RIGHT of the actual detected face, moving
                        # with it, so the student can read their status
                        # without looking away from their own face.
                        # ------------------------------------------------

                        if face_bounds is not None:

                            label_color = (0, 255, 0) if is_recognized else (0, 255, 255)

                            label_x = int(face_bounds[2]) + 15
                            label_x = min(label_x, max(frame_width - 230, 0))

                            label_y = int(face_bounds[1]) + 30

                            cv2.putText(
                                frame,
                                student["name"],
                                (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2
                            )

                            cv2.putText(
                                frame,
                                "PRESENT" if is_recognized else "VERIFICATION...",
                                (label_x, label_y + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2
                            )

                        latest_status.update({
                            "state": "recognized" if is_recognized else "verifying",
                            "name": student["name"],
                            "enrollment": student["enrollment"],
                            "distance": student["distance"],
                            "gap": student["gap"],
                            "candidate_count": session_state["candidate_count"],
                            "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                            **mask_status_fields(mask_info),
                            **liveness_status_fields(liveness_state, motion_score),
                        })

                    else:

                        # ------------------------------------------------
                        # Anti-spoofing gate: don't advance the identity
                        # confirmation counter until the motion-liveness
                        # check reports a real, moving face - blocks a
                        # static photo/phone/screen held up to the
                        # camera. Self-recovering: as soon as natural
                        # movement is seen again, this clears on its own.
                        # ------------------------------------------------

                        cv2.putText(
                            frame, "CONFIRMING LIVENESS...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2
                        )

                        cv2.putText(
                            frame,
                            f"{student['name']} ({student['enrollment']})",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2
                        )

                        if face_bounds is not None:

                            label_x = int(face_bounds[2]) + 15
                            label_x = min(label_x, max(frame_width - 230, 0))

                            label_y = int(face_bounds[1]) + 30

                            cv2.putText(
                                frame,
                                student["name"],
                                (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
                            )

                            cv2.putText(
                                frame,
                                "CONFIRMING...",
                                (label_x, label_y + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2
                            )

                        latest_status.update({
                            "state": "confirming_liveness",
                            "name": student["name"],
                            "enrollment": student["enrollment"],
                            "distance": student["distance"],
                            "gap": student["gap"],
                            "candidate_count": 0,
                            "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                            **mask_status_fields(mask_info),
                            **liveness_status_fields(liveness_state, motion_score),
                        })

                elif (
                    current_enrollment == session_state["candidate_enrollment"]
                    and session_state["candidate_count"] > 0
                ):

                    # ------------------------------------------------
                    # GRACE CASE: same person we're already counting
                    # confirmations for, but this single frame got
                    # rejected (e.g. a natural blink briefly changes
                    # eye-landmark shape, since eyes are part of the
                    # stable landmark subset used for identity
                    # matching). Don't wipe out the streak for one
                    # noisy frame - just hold and wait for the next.
                    # ------------------------------------------------

                    cv2.putText(
                        frame, "VERIFYING...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"{student['name']} ({student['enrollment']})",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"{recognition.REQUIRED_CONFIRMATIONS}-frame confirmation: "
                        f"{session_state['candidate_count']}/{recognition.REQUIRED_CONFIRMATIONS}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )

                    if face_bounds is not None:

                        label_x = int(face_bounds[2]) + 15
                        label_x = min(label_x, max(frame_width - 230, 0))

                        label_y = int(face_bounds[1]) + 30

                        cv2.putText(
                            frame,
                            student["name"],
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )

                        cv2.putText(
                            frame,
                            "VERIFICATION...",
                            (label_x, label_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                        )

                    latest_status.update({
                        "state": "verifying",
                        "name": student["name"],
                        "enrollment": student["enrollment"],
                        "distance": student["distance"],
                        "gap": student["gap"],
                        "candidate_count": session_state["candidate_count"],
                        "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                        **mask_status_fields(mask_info),
                        **liveness_status_fields(liveness_state, motion_score),
                    })

                else:

                    reset_confirmation_state()

                    cv2.putText(
                        frame, "BEST CANDIDATE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"{student['name']} ({student['enrollment']})",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Distance: {recognition.format_value(student['distance'])}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Gap: {recognition.format_value(student['gap'])}",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Threshold: {recognition.MATCH_THRESHOLD:.3f}",
                        (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Min Gap: {recognition.MIN_DISTANCE_GAP:.3f}",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch_text} / Yaw: {yaw_text}",
                        (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame, "REJECTED", (20, 255),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

                    latest_status.update({
                        "state": "rejected",
                        "name": student["name"],
                        "enrollment": student["enrollment"],
                        "distance": student["distance"],
                        "gap": student["gap"],
                        "candidate_count": 0,
                        "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                        **mask_status_fields(mask_info),
                        **liveness_status_fields(liveness_state, motion_score),
                    })

            else:

                reset_confirmation_state()

                if not in_box:

                    cv2.putText(
                        frame, "MOVE INTO THE BOX", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                    )

                    latest_status.update({
                        "state": "move_into_box",
                        "name": student["name"],
                        "enrollment": student["enrollment"],
                        "distance": student["distance"],
                        "gap": student["gap"],
                        "candidate_count": 0,
                        "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                        **mask_status_fields(mask_info),
                        **liveness_status_fields(liveness_state, motion_score),
                    })

                else:

                    if not pitch_ok:

                        cv2.putText(
                            frame, "LOOK MORE DIRECTLY AT CAMERA", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                        )

                    else:

                        cv2.putText(
                            frame, "TURN TO FACE THE CAMERA", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                        )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch_text} (max {recognition.MAX_RECOGNITION_PITCH})",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {yaw_text} (max {recognition.MAX_RECOGNITION_YAW})",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )

                    latest_status.update({
                        "state": "look_straight",
                        "name": student["name"],
                        "enrollment": student["enrollment"],
                        "distance": student["distance"],
                        "gap": student["gap"],
                        "candidate_count": 0,
                        "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                        **mask_status_fields(mask_info),
                        **liveness_status_fields(liveness_state, motion_score),
                    })

        else:

            reset_confirmation_state()

            cv2.putText(
                frame, "NO FACE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

            latest_status.update({
                "state": "no_face",
                "name": None,
                "enrollment": None,
                "distance": None,
                "gap": None,
                "candidate_count": 0,
                "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
                **mask_status_fields(mask_info),
                **liveness_status_fields(liveness_state, motion_score),
            })

    return frame


def generate_frames():

    camera = cv2.VideoCapture(0)

    try:
        while True:

            success, frame = camera.read()

            if not success:
                break

            # Mirror the feed so it feels natural to look at (like
            # a selfie camera/mirror) - move left, see yourself
            # move left. Everything downstream (guide box,
            # landmarks, encoding, text) works on this flipped
            # frame consistently.
            frame = cv2.flip(frame, 1)

            with session_lock:
                active = session_state["active"]

            if active:
                frame = process_recognition(frame)

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
    finally:
        camera.release()


@app.route("/")
def dashboard():

    students = get_all_students()

    return render_template(
        "dashboard.html",
        students=students
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start_session", methods=["POST"])
def start_attendance_session():

    with session_lock:

        if session_state["active"]:

            return jsonify({
                "status": "already_active",
                "session_id": session_state["session_id"],
                "total_students": session_state["total_students"]
            })

        total_students = len(get_all_students())

        session_id = start_session(total_students)

        session_state["active"] = True
        session_state["session_id"] = session_id
        session_state["total_students"] = total_students
        session_state["marked_students"] = set()

        reset_confirmation_state()

        latest_status.update({
            "state": "idle",
            "name": None,
            "enrollment": None,
            "distance": None,
            "gap": None,
            "candidate_count": 0,
            "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
            "mask_detected": None,
            "mask_color_distance": None,
            "mask_texture_std": None,
            "liveness_state": "idle",
            "motion_score": None,
        })

    return jsonify({
        "status": "started",
        "session_id": session_id,
        "total_students": total_students
    })


@app.route("/end_session", methods=["POST"])
def end_attendance_session():

    with session_lock:

        if not session_state["active"]:
            return jsonify({"status": "not_active"})

        session_id = session_state["session_id"]

        session_state["active"] = False
        session_state["session_id"] = None
        session_state["last_completed_session_id"] = session_id

        reset_confirmation_state()

        latest_status.update({
            "state": "idle",
            "name": None,
            "enrollment": None,
            "distance": None,
            "gap": None,
            "candidate_count": 0,
            "required_confirmations": recognition.REQUIRED_CONFIRMATIONS,
            "mask_detected": None,
            "mask_color_distance": None,
            "mask_texture_std": None,
            "liveness_state": "idle",
            "motion_score": None,
        })

    finish_session(session_id)

    return jsonify({"status": "ended"})


@app.route("/live_status")
def live_status():

    with session_lock:
        status_copy = dict(latest_status)

    return jsonify(status_copy)


@app.route("/attendance_status")
def attendance_status():

    with session_lock:

        total = session_state["total_students"] or len(get_all_students())
        present_enrollments = list(session_state["marked_students"])
        present = len(present_enrollments)
        absent = max(total - present, 0)

    return jsonify({
        "total": total,
        "present": present,
        "absent": absent,
        "present_enrollments": present_enrollments
    })


def build_attendance_workbook(session_id):
    """Builds an .xlsx with every enrolled student - Present rows
    get their real date/time from get_session_attendance(), and
    everyone else is filled in as Absent."""

    all_students = get_all_students()
    present_records = get_session_attendance(session_id)

    present_by_enrollment = {
        record["roll_number"]: record
        for record in present_records
    }

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Attendance"

    headers = ["Enrollment No", "Name", "Status", "Date", "Time"]
    sheet.append(headers)

    for cell in sheet[1]:
        cell.font = Font(bold=True)

    for student in all_students:

        roll_number = student["roll_number"]
        name = student["name"]

        record = present_by_enrollment.get(roll_number)

        if record:
            sheet.append([
                roll_number, name, "Present",
                record["date"], record["time"]
            ])
        else:
            sheet.append([roll_number, name, "Absent", "", ""])

    for column_cells in sheet.columns:

        length = max(
            (len(str(cell.value)) for cell in column_cells if cell.value is not None),
            default=0
        )

        sheet.column_dimensions[column_cells[0].column_letter].width = length + 4

    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    return output


@app.route("/export_session")
def export_session():

    with session_lock:
        session_id = session_state["last_completed_session_id"]

    if session_id is None:
        return jsonify({"status": "no_session"}), 404

    workbook_file = build_attendance_workbook(session_id)

    return send_file(
        workbook_file,
        as_attachment=True,
        download_name=f"attendance_session_{session_id}.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        )
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)
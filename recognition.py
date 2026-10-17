from attendance import start_session, mark_attendance, finish_session

import cv2
import json
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from database import get_connection


MODEL_PATH = "models/face_landmarker.task"

# SFace face-recognition model (128-d deep embedding).
#
# Download it yourself (this sandbox can't reach the file due to
# a network restriction, but your own machine will be fine):
#   https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
# Click "Download" on that page (NOT "raw" - GitHub's raw link
# just returns a Git-LFS pointer, not the actual file), and save
# it as models/face_recognition_sface_2021dec.onnx, right next to
# your existing models/face_landmarker.task.
SFACE_MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"


# ============================================================
# SETTINGS
# ============================================================

# SFace's distance is a completely different scale than the old
# landmark-distance method (0.001-0.25 range). These are OpenCV's
# own documented starting values for the L2-norm distance - the
# live distance/gap are exposed via latest_status as always, so
# tune these from real data rather than guessing further.
MATCH_THRESHOLD = 1.128

MIN_DISTANCE_GAP = 0.15

# Same student must be recognized this many times before we
# consider the recognition stable.
REQUIRED_CONFIRMATIONS = 25

# How often (in frames) to print diagnostic info to the terminal.
DIAGNOSTIC_PRINT_EVERY_N_FRAMES = 15

# Maximum allowed head tilt (up/down) for a frame to count
# towards recognition/confirmation.
MAX_RECOGNITION_PITCH = 12

# Maximum allowed head TURN (left/right) for a frame to count
# towards recognition/confirmation.
MAX_RECOGNITION_YAW = 12

# Guide box size, as a fraction of the frame width/height.
GUIDE_BOX_WIDTH_RATIO = 0.35
GUIDE_BOX_HEIGHT_RATIO = 0.55

# Extra margin (as a fraction of the detected face box) included
# when cropping the face for the SFace embedding, so the crop
# isn't razor-tight around just the landmark cloud.
FACE_CROP_MARGIN = 0.20


# ============================================================
# MASK DETECTION (heuristic, informational only)
#
# No pretrained model - compares the color/texture of the
# lower-face region (mouth/chin, always covered by any mask)
# against a reference skin patch between the eyes (never
# covered by any mask). A mask is usually a flat, distinctly
# colored fabric, so it shows up as either a clear color
# difference or unusually low texture variation (skin/mouth
# detail vs a flat piece of cloth).
#
# This is PURELY informational - it never blocks or affects
# recognition/attendance.
# ============================================================

MASK_COLOR_DIFF_THRESHOLD = 25
MASK_LOW_TEXTURE_THRESHOLD = 18


def detect_mask(frame, face_bounds):
    """
    Returns a dict with mask_detected (bool) and the raw
    diagnostic numbers, or None if face_bounds isn't available.
    """

    if face_bounds is None:
        return None

    min_x, min_y, max_x, max_y = face_bounds
    height = max_y - min_y

    if height <= 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    x1 = int(min_x + 0.15 * (max_x - min_x))
    x2 = int(max_x - 0.15 * (max_x - min_x))

    ref_y1 = int(min_y + 0.30 * height)
    ref_y2 = int(min_y + 0.45 * height)

    lower_y1 = int(min_y + 0.68 * height)
    lower_y2 = int(max_y)

    x1 = max(0, min(x1, frame_width))
    x2 = max(0, min(x2, frame_width))
    ref_y1 = max(0, min(ref_y1, frame_height))
    ref_y2 = max(0, min(ref_y2, frame_height))
    lower_y1 = max(0, min(lower_y1, frame_height))
    lower_y2 = max(0, min(lower_y2, frame_height))

    if x2 <= x1 or ref_y2 <= ref_y1 or lower_y2 <= lower_y1:
        return None

    ref_patch = frame[ref_y1:ref_y2, x1:x2]
    lower_patch = frame[lower_y1:lower_y2, x1:x2]

    if ref_patch.size == 0 or lower_patch.size == 0:
        return None

    ref_hsv = cv2.cvtColor(ref_patch, cv2.COLOR_BGR2HSV)
    lower_hsv = cv2.cvtColor(lower_patch, cv2.COLOR_BGR2HSV)

    ref_mean = ref_hsv.reshape(-1, 3).mean(axis=0)
    lower_mean = lower_hsv.reshape(-1, 3).mean(axis=0)

    hue_diff = abs(float(ref_mean[0]) - float(lower_mean[0]))
    hue_diff = min(hue_diff, 180 - hue_diff)

    sat_diff = abs(float(ref_mean[1]) - float(lower_mean[1]))

    color_distance = hue_diff + sat_diff * 0.5

    texture_std = float(lower_hsv[:, :, 2].std())

    mask_detected = (
        color_distance > MASK_COLOR_DIFF_THRESHOLD
        or texture_std < MASK_LOW_TEXTURE_THRESHOLD
    )

    return {
        "mask_detected": mask_detected,
        "color_distance": color_distance,
        "texture_std": texture_std,
    }


# ============================================================
# LIVENESS (informational motion check)
#
# Passive, non-blocking signal: a real face shows tiny natural
# movement frame-to-frame (blinking, breathing, small shifts);
# a printed photo or phone screen held up to the camera stays
# almost perfectly static. This NEVER blocks recognition or
# attendance - it's purely a display hint. (The earlier active
# blink-challenge attempt caused more problems than it solved
# by gating the actual attendance flow - this doesn't.)
# ============================================================

# How many recent frames the rolling window looks at.
MOTION_WINDOW_SIZE = 15

# A single frame's non-rigid residual above this counts as
# "real movement" (e.g. a blink). Starting value - the raw
# motion_score is exposed via latest_status, so tune it from
# real numbers the same way every other threshold here was.
MOTION_LIVE_THRESHOLD = 0.0008

# At least this many frames within the window must show real
# movement to count as "Live". Using a small COUNT instead of
# an average means a single blink (which spans a few frames)
# is picked up immediately, instead of getting diluted into a
# 15-frame average and taking a long time to register.
MOTION_MIN_ELEVATED_FRAMES = 2


def compute_motion_score(current_landmarks, previous_landmarks):
    """
    Measures NON-RIGID deformation between two consecutive
    frames' landmarks - movement that ISN'T explained by simple
    translation/rotation of the whole face. A shaky hand holding
    up a photo/phone moves everything together (pure rigid
    motion), which this removes via Kabsch alignment before
    measuring - so hand-shake alone no longer registers as
    "alive". A real face has genuine internal movement (blink,
    subtle expression change) that survives rigid alignment;
    a flat photo/screen does not, no matter how much it shakes.
    """

    if current_landmarks is None or previous_landmarks is None:
        return None

    current_xyz = current_landmarks[:, :3].astype(np.float64)
    previous_xyz = previous_landmarks[:, :3].astype(np.float64)

    if current_xyz.shape != previous_xyz.shape:
        return None

    current_centered = current_xyz - current_xyz.mean(axis=0, keepdims=True)
    previous_centered = previous_xyz - previous_xyz.mean(axis=0, keepdims=True)

    # Best-fit rotation aligning "previous" onto "current"
    # (Kabsch algorithm) - same technique used earlier for
    # pose-robust identity comparison.
    covariance = previous_centered.T @ current_centered

    U, S, Vt = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = Vt.T @ correction @ U.T

    aligned_previous = previous_centered @ rotation.T

    # Residual left over after removing the best possible rigid
    # motion - this is the non-rigid component (real facial
    # movement), not hand-shake/camera-shake.
    residual = float(
        np.mean(
            np.linalg.norm(current_centered - aligned_previous, axis=1)
        )
    )

    return residual


# ============================================================
# MediaPipe Face Landmarker
#
# Still used for: face detection, pitch/yaw (pose gating), the
# guide box / face bounding box, and mask detection. NOT used
# for identity anymore - that's SFace now (below).
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# SFace face recognizer (deep embedding, identity matching)
# ============================================================

face_recognizer = cv2.FaceRecognizerSF_create(
    SFACE_MODEL_PATH,
    ""
)


# ============================================================
# Load enrolled faces
#
# face_data now stores a flat 128-number SFace embedding
# (JSON list), not landmark points. Re-enroll everyone after
# switching to this - old landmark-based face_data won't work
# with SFace's distance metric.
# ============================================================

def load_enrolled_faces():

    connection = get_connection()

    students = connection.execute(
        """
        SELECT roll_number, name, face_data
        FROM students
        WHERE face_data IS NOT NULL
        """
    ).fetchall()

    connection.close()

    enrolled_faces = []

    for student in students:

        try:

            face_data = json.loads(
                student["face_data"]
            )

            encoding = np.array(
                face_data,
                dtype=np.float32
            ).reshape(1, -1)

            # SFace embeddings are always 128 numbers. If this is
            # some other length, it's almost certainly stale data
            # from the OLD landmark-based method (this student
            # hasn't been re-enrolled yet with enroll.py). Skip it
            # instead of letting it crash the whole video stream
            # when compared against a real 128-d live encoding.
            if encoding.shape[1] != 128:

                print(
                    f"Skipping {student['name']} "
                    f"({student['roll_number']}): face_data has "
                    f"{encoding.shape[1]} values, expected 128 - "
                    f"needs re-enrollment with enroll.py."
                )

                continue

            enrolled_faces.append({
                "enrollment": student["roll_number"],
                "name": student["name"],
                "face_data": encoding
            })

        except Exception as error:

            print(
                f"Could not load face data for "
                f"{student['name']}: {error}"
            )

    return enrolled_faces


# ============================================================
# Extract face landmarks + head pose
#
# Still returns landmarks (used for face_bounds/mask/guide box)
# plus pitch and yaw. Identity no longer comes from here - see
# get_face_encoding() below.
# ============================================================

def get_face_data(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None, None

    landmarks = result.face_landmarks[0]

    face_data = []

    for landmark in landmarks:

        face_data.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    face_data = np.array(
        face_data,
        dtype=np.float32
    )

    pitch_angle = None
    yaw_angle = None

    if result.facial_transformation_matrixes:

        matrix = result.facial_transformation_matrixes[0]

        rotation = matrix[:3, :3]

        pitch_angle = np.degrees(
            np.arctan2(
                rotation[2, 1],
                rotation[2, 2]
            )
        )

        yaw_angle = np.degrees(
            np.arctan2(
                -rotation[2, 0],
                np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
            )
        )

    return face_data, pitch_angle, yaw_angle


# ============================================================
# Guide box helpers (positioning gate)
# ============================================================

def draw_guide_box(frame):

    height, width = frame.shape[:2]

    box_width = int(width * GUIDE_BOX_WIDTH_RATIO)
    box_height = int(height * GUIDE_BOX_HEIGHT_RATIO)

    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    return x1, y1, x2, y2


def get_face_bounds(face_points, width, height):

    xs = [point[0] * width for point in face_points]
    ys = [point[1] * height for point in face_points]

    return min(xs), min(ys), max(xs), max(ys)


def is_face_within_box(face_bounds, box, margin=15):

    face_min_x, face_min_y, face_max_x, face_max_y = face_bounds
    box_x1, box_y1, box_x2, box_y2 = box

    return (
        face_min_x >= box_x1 - margin
        and face_max_x <= box_x2 + margin
        and face_min_y >= box_y1 - margin
        and face_max_y <= box_y2 + margin
    )


# ============================================================
# Face encoding (SFace) - THIS is identity now
#
# Crops the face (with a bit of margin) using the bounding box
# already computed from MediaPipe's landmarks, resizes it to
# SFace's expected 112x112 input, and extracts a 128-d feature.
#
# NOTE: this does a simple crop+resize, not SFace's "official"
# 5-point alignCrop(). It's simpler and avoids needing to map
# MediaPipe landmarks to YuNet's exact 5-point format, at some
# cost to accuracy. If accuracy needs another push later, this
# is the place to add proper 5-point alignment.
# ============================================================

def get_face_encoding(frame, face_bounds):

    if face_bounds is None:
        return None

    min_x, min_y, max_x, max_y = face_bounds

    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return None

    margin_x = width * FACE_CROP_MARGIN
    margin_y = height * FACE_CROP_MARGIN

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, int(min_x - margin_x))
    y1 = max(0, int(min_y - margin_y))
    x2 = min(frame_width, int(max_x + margin_x))
    y2 = min(frame_height, int(max_y + margin_y))

    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    aligned = cv2.resize(face_crop, (112, 112))

    feature = face_recognizer.feature(aligned)

    return feature


# ============================================================
# Compare two SFace embeddings (L2 distance - lower = closer)
# ============================================================

def compare_faces(current_encoding, enrolled_encoding):

    distance = face_recognizer.match(
        current_encoding,
        enrolled_encoding,
        cv2.FaceRecognizerSF_FR_NORM_L2
    )

    return float(distance)


# ============================================================
# Face Recognition (DIAGNOSTIC MODE)
#
# Same shape/behavior as before - always returns the best
# candidate + diagnostics when a face + enrolled students
# exist, even if rejected, so the caller can show WHY.
# ============================================================

def recognize_face(
    current_encoding,
    enrolled_faces
):

    if current_encoding is None:
        return None

    if not enrolled_faces:
        return None

    results = []

    for student in enrolled_faces:

        distance = compare_faces(
            current_encoding,
            student["face_data"]
        )

        results.append({
            "enrollment": student["enrollment"],
            "name": student["name"],
            "distance": distance
        })

    results.sort(
        key=lambda x: x["distance"]
    )

    best_match = results[0]

    has_second_candidate = len(results) > 1

    second_best_distance = (
        results[1]["distance"]
        if has_second_candidate
        else float("inf")
    )

    distance_gap = (
        second_best_distance - best_match["distance"]
        if has_second_candidate
        else float("inf")
    )

    distance_ok = best_match["distance"] < MATCH_THRESHOLD

    if has_second_candidate:
        gap_ok = distance_gap >= MIN_DISTANCE_GAP
    else:
        gap_ok = True

    accepted = distance_ok and gap_ok

    if not distance_ok:
        reason = "distance_too_high"
    elif not gap_ok:
        reason = "gap_too_small"
    else:
        reason = "accepted"

    return {
        "enrollment": best_match["enrollment"],
        "name": best_match["name"],
        "distance": best_match["distance"],
        "second_best_distance": second_best_distance,
        "gap": distance_gap,
        "accepted": accepted,
        "reason": reason
    }


def format_value(value):

    if value == float("inf"):
        return "inf"

    return f"{value:.3f}"


# ============================================================
# Main (standalone tester - unchanged UI/flow, updated only to
# use the SFace encoding instead of raw landmarks for identity)
# ============================================================

def main():

    enrolled_faces = load_enrolled_faces()

    print()
    print(
        f"Enrolled students: {len(enrolled_faces)}"
    )

    for student in enrolled_faces:

        print(
            f"- {student['enrollment']} "
            f"| {student['name']}"
        )

    if not enrolled_faces:

        print()
        print("No enrolled faces found.")
        return

    session_id = start_session(63)

    print()
    print(
        f"Attendance session started: {session_id}"
    )

    print()
    print("Starting camera...")
    print("Press Q to exit.")
    print()
    print(
        f"DIAGNOSTIC MODE | "
        f"MATCH_THRESHOLD={MATCH_THRESHOLD} | "
        f"MIN_DISTANCE_GAP={MIN_DISTANCE_GAP}"
    )
    print()

    camera = cv2.VideoCapture(0)

    candidate_enrollment = None
    candidate_count = 0

    confirmed_student = None
    attendance_marked = False

    frame_count = 0

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Mirror the feed (same as app.py/enroll.py) - move left,
        # see yourself move left.
        frame = cv2.flip(frame, 1)

        frame_count += 1

        current_face, pitch_angle, yaw_angle = get_face_data(frame)

        frame_height, frame_width = frame.shape[:2]

        box = draw_guide_box(frame)
        box_x1, box_y1, box_x2, box_y2 = box

        in_box = False
        face_bounds = None

        if current_face is not None:

            face_bounds = get_face_bounds(
                current_face,
                frame_width,
                frame_height
            )

            in_box = is_face_within_box(
                face_bounds,
                box
            )

        pitch_ok = (
            pitch_angle is not None
            and abs(pitch_angle) <= MAX_RECOGNITION_PITCH
        )

        yaw_ok = (
            yaw_angle is not None
            and abs(yaw_angle) <= MAX_RECOGNITION_YAW
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

        current_encoding = get_face_encoding(frame, face_bounds)

        student = recognize_face(
            current_encoding,
            enrolled_faces
        )

        if student:

            current_enrollment = student["enrollment"]

            pitch_text = (
                format_value(pitch_angle)
                if pitch_angle is not None
                else "n/a"
            )

            yaw_text = (
                format_value(yaw_angle)
                if yaw_angle is not None
                else "n/a"
            )

            if frame_count % DIAGNOSTIC_PRINT_EVERY_N_FRAMES == 0:

                print(
                    f"[DIAG] {student['name']} "
                    f"({student['enrollment']}) | "
                    f"best={format_value(student['distance'])} | "
                    f"second_best={format_value(student['second_best_distance'])} | "
                    f"gap={format_value(student['gap'])} | "
                    f"threshold={MATCH_THRESHOLD} | "
                    f"min_gap={MIN_DISTANCE_GAP} | "
                    f"accepted={student['accepted']} | "
                    f"reason={student['reason']} | "
                    f"pitch={pitch_text} | yaw={yaw_text}"
                )

            if position_ok:

                if student["accepted"]:

                    if current_enrollment == candidate_enrollment:

                        candidate_count += 1

                    else:

                        candidate_enrollment = current_enrollment
                        candidate_count = 1
                        confirmed_student = None
                        attendance_marked = False

                    if candidate_count >= REQUIRED_CONFIRMATIONS:

                        confirmed_student = student

                        if not attendance_marked:

                            success_mark, message = mark_attendance(
                                student["enrollment"],
                                session_id
                            )

                            if success_mark:
                                print(message)

                            attendance_marked = True

                    if confirmed_student and candidate_count >= REQUIRED_CONFIRMATIONS:

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
                            f"{REQUIRED_CONFIRMATIONS}-frame confirmation: "
                            f"{candidate_count}/{REQUIRED_CONFIRMATIONS}",
                            (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

                    cv2.putText(
                        frame,
                        f"Distance: {format_value(student['distance'])}",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Gap: {format_value(student['gap'])}",
                        (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                else:

                    candidate_enrollment = None
                    candidate_count = 0
                    confirmed_student = None
                    attendance_marked = False

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
                        f"Distance: {format_value(student['distance'])}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Gap: {format_value(student['gap'])}",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Threshold: {MATCH_THRESHOLD:.3f}",
                        (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Min Gap: {MIN_DISTANCE_GAP:.3f}",
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

            else:

                candidate_enrollment = None
                candidate_count = 0
                confirmed_student = None
                attendance_marked = False

                if not in_box:

                    cv2.putText(
                        frame, "MOVE INTO THE BOX", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                    )

                elif not pitch_ok:

                    cv2.putText(
                        frame, "LOOK MORE DIRECTLY AT CAMERA", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch_text} (max {MAX_RECOGNITION_PITCH})",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )

                else:

                    cv2.putText(
                        frame, "TURN TO FACE THE CAMERA", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {yaw_text} (max {MAX_RECOGNITION_YAW})",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )

        else:

            candidate_enrollment = None
            candidate_count = 0
            confirmed_student = None
            attendance_marked = False

            cv2.putText(
                frame, "NO FACE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            finish_session(
                session_id
            )

            print()
            print(
                "Attendance session completed."
            )

            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
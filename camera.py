import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Camera
camera = cv2.VideoCapture(0)


# Face Detector
base_options = python.BaseOptions(
    model_asset_path="models/face_detector.tflite"
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.6
)

detector = vision.FaceDetector.create_from_options(options)


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect faces
        result = detector.detect(mp_image)

        height, width, _ = frame.shape

        for detection in result.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            # Keep coordinates inside frame
            x = max(0, x)
            y = max(0, y)

            w = min(w, width - x)
            h = min(h, height - y)

            # Face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                frame,
                "Face Detected",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Encode frame
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
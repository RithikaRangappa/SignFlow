import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------
# 1. Load the hand landmark model
# ----------------------------------

MODEL_PATH = "models/hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ----------------------------------
# 2. Open the webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    landmarker.close()
    exit()

print("Hand tracking started!")
print("Press 'q' to quit.")


# ----------------------------------
# 3. Define hand connections
# ----------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ----------------------------------
# 4. Process video frames
# ----------------------------------

start_time = time.monotonic()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Mirror the camera feed
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame into a MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Create timestamp for video processing
    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    # Detect hands
    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ----------------------------------
    # 5. Draw landmarks
    # ----------------------------------

    height, width, _ = frame.shape

    detected_hands = len(results.hand_landmarks)

    for hand_landmarks in results.hand_landmarks:

        # Draw landmark points
        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw connections between landmarks
        for start_idx, end_idx in HAND_CONNECTIONS:

            start_landmark = hand_landmarks[start_idx]
            end_landmark = hand_landmarks[end_idx]

            start_point = (
                int(start_landmark.x * width),
                int(start_landmark.y * height)
            )

            end_point = (
                int(end_landmark.x * width),
                int(end_landmark.y * height)
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 0, 0),
                2
            )


    # ----------------------------------
    # 6. Display detection information
    # ----------------------------------

    cv2.putText(
        frame,
        f"Hands detected: {detected_hands}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ----------------------------------
    # 7. Display the video
    # ----------------------------------

    cv2.imshow(
        "SignFlow - Hand Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ----------------------------------
# 8. Clean up
# ----------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Hand tracking stopped.")
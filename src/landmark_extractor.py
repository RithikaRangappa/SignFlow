import cv2
import mediapipe as mp
import time
import numpy as np

from preprocessing import extract_two_hand_features
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------
# 1. Load the MediaPipe model
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
# 2. Open webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")

    landmarker.close()

    exit()


start_time = time.monotonic()

print("SignFlow fixed-slot landmark extractor started!")
print("Press S to inspect the feature vector.")
print("Press Q to quit.")


# ----------------------------------
# 3. Process webcam frames
# ----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror webcam for a selfie-style display
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Generate timestamp
    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    # Detect hands
    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ----------------------------------
    # 4. Extract fixed-slot features
    # ----------------------------------

    feature_vector = extract_two_hand_features(
        results
    )

    hand_count = len(results.hand_landmarks)


    # ----------------------------------
    # 5. Display hand information
    # ----------------------------------

    left_present = int(feature_vector[126])
    right_present = int(feature_vector[127])

    cv2.putText(
        frame,
        f"Hands detected: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Left hand: {left_present}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Right hand: {right_present}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Feature shape: {feature_vector.shape}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ----------------------------------
    # 6. Display webcam
    # ----------------------------------

    cv2.imshow(
        "SignFlow - Fixed Hand Slots",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # ----------------------------------
    # 7. Inspect features using S
    # ----------------------------------

    if key == ord("s"):

        print("\n--- SignFlow Feature Data ---")

        print("Feature vector shape:")
        print(feature_vector.shape)

        print("\nLeft hand detected:")
        print(left_present)

        print("\nRight hand detected:")
        print(right_present)

        print("\nFirst 10 left-hand features:")
        print(feature_vector[:10])

        print("\nFirst 10 right-hand features:")
        print(feature_vector[63:73])

        print("\nPresence indicators:")
        print(feature_vector[126:128])


    # ----------------------------------
    # 8. Quit
    # ----------------------------------

    if key == ord("q"):
        break


# ----------------------------------
# 9. Clean up
# ----------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Fixed-slot landmark extraction stopped.")
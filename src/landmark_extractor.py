import cv2
import mediapipe as mp
import time
import numpy as np

from preprocessing import normalize_hand
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 1. Load the model

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


# 2. Open webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    landmarker.close()
    exit()

start_time = time.monotonic()


# 3. Process frames

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # 4. Extract and normalise detected hand landmarks

    feature_vector = []

    for hand_landmarks in results.hand_landmarks:

        normalized_landmarks = normalize_hand(
            hand_landmarks
        )

        feature_vector.extend(
            normalized_landmarks.flatten()
        )


    # Pad missing data with zeros

    while len(feature_vector) < 126:

        feature_vector.append(0.0)


    feature_vector = np.array(
        feature_vector[:126],
        dtype=np.float32
    )


    # 5. Display information

    hand_count = len(results.hand_landmarks)

    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Feature shape: {feature_vector.shape}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # 6. Print data when S is pressed

    cv2.imshow(
        "SignFlow - Landmark Data",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        print("\n--- Landmark Feature Vector ---")

        print(feature_vector)

        print("\nFeature vector shape:")
        print(feature_vector.shape)

        print("\nNumber of detected hands:")
        print(hand_count)

        print("\nMinimum value:")
        print(np.min(feature_vector))

        print("\nMaximum value:")
        print(np.max(feature_vector))

        print("\nFirst three values (wrist):")
        print(feature_vector[:3])


    if key == ord("q"):
        break


# 7. Clean up

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Landmark extraction stopped.")
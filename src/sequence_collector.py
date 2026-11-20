import cv2
import mediapipe as mp
import time
import numpy as np

from collections import deque
from pathlib import Path

from preprocessing import normalize_hand
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
# 2. Set up sequence parameters
# ----------------------------------

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ----------------------------------
# 3. Open webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")

    landmarker.close()

    exit()


start_time = time.monotonic()

print("SignFlow sequence collector started!")
print("Press S to save a 30-frame sequence.")
print("Press Q to quit.")


# ----------------------------------
# 4. Process webcam frames
# ----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror the webcam
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
    # 5. Extract normalised landmarks
    # ----------------------------------

    feature_vector = []

    for hand_landmarks in results.hand_landmarks:

        normalized_landmarks = normalize_hand(
            hand_landmarks
        )

        feature_vector.extend(
            normalized_landmarks.flatten()
        )


    # Pad or truncate to 126 features

    while len(feature_vector) < FEATURE_COUNT:

        feature_vector.append(0.0)


    feature_vector = np.array(
        feature_vector[:FEATURE_COUNT],
        dtype=np.float32
    )


    # Add current frame to sequence buffer

    sequence_buffer.append(feature_vector)


    # ----------------------------------
    # 6. Display information
    # ----------------------------------

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
        f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    # ----------------------------------
    # 7. Display webcam
    # ----------------------------------

    cv2.imshow(
        "SignFlow - Sequence Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # ----------------------------------
    # 8. Save sequence when S is pressed
    # ----------------------------------

    if key == ord("s"):

        if len(sequence_buffer) == SEQUENCE_LENGTH:

            sequence = np.array(
                sequence_buffer,
                dtype=np.float32
            )

            print("\nSequence captured!")

            print("Sequence shape:")
            print(sequence.shape)

            print("\nTotal values:")
            print(sequence.size)


            # Create output directory

            output_dir = Path("data/processed")

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # Save sequence

            output_path = (
                output_dir / "demo_sequence.npy"
            )

            np.save(
                output_path,
                sequence
            )

            print("\nSaved to:")
            print(output_path)


        else:

            print(
                "Please wait until 30 frames are collected."
            )


    # ----------------------------------
    # 9. Quit
    # ----------------------------------

    if key == ord("q"):
        break


# ----------------------------------
# 10. Clean up
# ----------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Sequence collector stopped.")
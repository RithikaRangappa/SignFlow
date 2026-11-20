
import cv2
import mediapipe as mp
import time
import numpy as np

from collections import deque
from pathlib import Path

from preprocessing import extract_two_hand_features
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------
# 1. Load MediaPipe model
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
# 2. Sequence parameters
# ----------------------------------

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 128

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
# 4. Process frames
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
    # 5. Extract fixed-slot features
    # ----------------------------------

    feature_vector = extract_two_hand_features(
        results
    )

    # Add current frame to the rolling buffer
    sequence_buffer.append(feature_vector)


    # ----------------------------------
    # 6. Display information
    # ----------------------------------

    hand_count = len(results.hand_landmarks)

    left_present = int(feature_vector[126])
    right_present = int(feature_vector[127])

    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Left: {left_present} Right: {right_present}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
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
    # 8. Save sequence
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

            print("Total values:")
            print(sequence.size)


            # Create directory
            output_dir = Path("data/processed")

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # Save demo sequence
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

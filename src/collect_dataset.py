
import cv2
import mediapipe as mp
import time
import numpy as np
import csv

from collections import deque
from pathlib import Path

from preprocessing import extract_two_hand_features
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------------
# 1. Get sign label
# ----------------------------------

sign_label = input(
    "Enter the sign label you want to collect: "
).strip().upper()

if not sign_label:
    print("Label cannot be empty.")
    exit()

if not sign_label.replace("_", "").isalnum():
    print("Use only letters, numbers, and underscores.")
    exit()


# ----------------------------------
# 2. Load MediaPipe
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
# 3. Sequence parameters
# ----------------------------------

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 128

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ----------------------------------
# 4. Output directories
# ----------------------------------

output_dir = Path(
    "data/processed/sign_samples"
) / sign_label

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

metadata_path = Path(
    "data/metadata/sign_samples.csv"
)

metadata_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Find next available sample number
sample_number = 1

while (
    output_dir / f"{sign_label}_{sample_number:03d}.npy"
).exists():

    sample_number += 1


# ----------------------------------
# 5. Open webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")
    landmarker.close()
    exit()


start_time = time.monotonic()

save_message = ""
save_message_time = 0

print("\nSignFlow labelled dataset collector started!")
print(f"Current label: {sign_label}")
print("Click the webcam window before pressing S.")
print("Press S to save. Press Q to quit.")


# ----------------------------------
# 6. Webcam processing
# ----------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
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


    # ----------------------------------
    # 7. Extract features
    # ----------------------------------

    feature_vector = extract_two_hand_features(
        results
    )

    sequence_buffer.append(feature_vector)

    left_present = int(feature_vector[126])
    right_present = int(feature_vector[127])


    # ----------------------------------
    # 8. Display information
    # ----------------------------------

    cv2.putText(
        frame,
        f"Label: {sign_label}",
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

    cv2.putText(
        frame,
        "Press S to save | Q to quit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Display save message temporarily
    if time.time() - save_message_time < 3:

        cv2.putText(
            frame,
            save_message,
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ----------------------------------
    # 9. Show webcam
    # ----------------------------------

    cv2.imshow(
        "SignFlow - Labelled Data Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # ----------------------------------
    # 10. Keyboard handling
    # ----------------------------------

    if key != 255:

        print("Key detected:", key)

    # Accept either uppercase or lowercase S
    if key == ord("s") or key == ord("S"):

        print("\nSave key detected!")

        if len(sequence_buffer) != SEQUENCE_LENGTH:

            print(
                f"Buffer incomplete: "
                f"{len(sequence_buffer)}/{SEQUENCE_LENGTH}"
            )

            save_message = "Buffer incomplete!"
            save_message_time = time.time()

            continue


        sequence = np.array(
            sequence_buffer,
            dtype=np.float32
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT
        ):

            print("Unexpected sequence shape:", sequence.shape)

            save_message = "Unexpected shape!"
            save_message_time = time.time()

            continue


        # ----------------------------------
        # 11. Save sequence
        # ----------------------------------

        sample_name = (
            f"{sign_label}_{sample_number:03d}.npy"
        )

        sample_path = output_dir / sample_name

        if sample_path.exists():

            print("File already exists:", sample_path)

            save_message = "File already exists!"
            save_message_time = time.time()

            continue


        try:

            np.save(
                sample_path,
                sequence
            )

            file_exists = metadata_path.exists()

            with open(
                metadata_path,
                "a",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                writer = csv.writer(csv_file)

                if not file_exists:

                    writer.writerow(
                        [
                            "label",
                            "sample_name",
                            "relative_path",
                            "frames",
                            "features_per_frame"
                        ]
                    )

                writer.writerow(
                    [
                        sign_label,
                        sample_name,
                        sample_path.as_posix(),
                        SEQUENCE_LENGTH,
                        FEATURE_COUNT
                    ]
                )


            print("\nSAMPLE SAVED!")
            print("Label:", sign_label)
            print("Sample:", sample_name)
            print("Shape:", sequence.shape)
            print("Location:", sample_path)


            save_message = "SAMPLE SAVED!"
            save_message_time = time.time()

            sample_number += 1


        except Exception as error:

            print("Save failed:", error)

            save_message = "SAVE FAILED!"
            save_message_time = time.time()


    # ----------------------------------
    # 12. Quit
    # ----------------------------------

    if key == ord("q") or key == ord("Q"):
        break


# ----------------------------------
# 13. Cleanup
# ----------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("\nDataset collection session ended.") 
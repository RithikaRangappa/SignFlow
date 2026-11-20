import numpy as np


def normalize_hand(landmarks):
    """
    Convert 21 hand landmarks into wrist-relative,
    scale-normalised coordinates.

    Returns an array of shape (21, 3).
    """

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Make the wrist the origin
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    # Calculate the maximum XY distance
    xy_distances = np.linalg.norm(
        landmarks[:, :2],
        axis=1
    )

    scale = np.max(xy_distances)

    # Normalise coordinates
    if scale > 0:
        landmarks[:, :2] /= scale
        landmarks[:, 2] /= scale

    return landmarks


def extract_two_hand_features(results):
    """
    Create a fixed 128-feature representation.

    63 values: Left hand
    63 values: Right hand
    1 value: Left hand presence
    1 value: Right hand presence
    """

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    left_present = 0.0
    right_present = 0.0

    for i, hand_landmarks in enumerate(results.hand_landmarks):

        # Determine whether this is the left or right hand
        handedness = results.handedness[i][0].category_name

        # Normalise landmarks
        normalized_landmarks = normalize_hand(
            hand_landmarks
        )

        flattened = normalized_landmarks.flatten()

        # Assign the landmarks to a fixed slot
        if handedness == "Right":
            left_hand = flattened
            left_present = 1.0

        elif handedness == "Left":
            right_hand = flattened
            right_present = 1.0

    # Combine the two hands and presence indicators
    feature_vector = np.concatenate(
        [
            left_hand,
            right_hand,
            np.array(
                [left_present, right_present],
                dtype=np.float32
            )
        ]
    )

    return feature_vector

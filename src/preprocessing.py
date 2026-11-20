import numpy as np


def normalize_hand(landmarks):
    """
    Convert 21 hand landmarks into wrist-relative coordinates.

    Input:
        landmarks: 21 landmarks, each containing x, y, z

    Output:
        NumPy array of shape (21, 3)
    """

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Use wrist landmark (index 0) as the origin
    wrist = landmarks[0].copy()

    # Subtract wrist coordinates from all landmarks
    landmarks = landmarks - wrist

    # Calculate hand scale using the maximum XY distance
    xy_distances = np.linalg.norm(
        landmarks[:, :2],
        axis=1
    )

    scale = np.max(xy_distances)

    # Avoid division by zero
    if scale > 0:
        landmarks[:, :2] = landmarks[:, :2] / scale

        # Scale depth using the same scale factor
        landmarks[:, 2] = landmarks[:, 2] / scale

    return landmarks


if __name__ == "__main__":

    print("SignFlow preprocessing module")
    print("Wrist-relative landmark normalisation is ready.")
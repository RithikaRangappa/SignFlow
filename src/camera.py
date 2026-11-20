import cv2
import time

# Open the laptop webcam
cap = cv2.VideoCapture(0)

# Check whether the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started!")
print("Press 'q' to quit.")

# Variables for calculating FPS
previous_time = time.time()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check whether the frame was captured
    if not ret:
        print("Error: Could not read frame.")
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Display FPS on the video
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display the webcam feed
    cv2.imshow("SignFlow - Webcam", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Webcam closed.")
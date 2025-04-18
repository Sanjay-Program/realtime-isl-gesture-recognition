import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection model
hands = mp_hands.Hands(
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.8,  # Confidence threshold for detection
    min_tracking_confidence=0.8  # Confidence threshold for tracking
)

# Function to calculate hand orientation (360 degrees)
def calculate_hand_orientation(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Calculate the angle of the hand
    dx = middle_finger_mcp.x - wrist.x
    dy = middle_finger_mcp.y - wrist.y
    angle = np.degrees(np.arctan2(dy, dx))  # Angle in degrees

    # Normalize angle to [0, 360)
    angle = (angle + 360) % 360
    return angle

# Function to remove background and focus on the hand
def remove_background(frame, landmarks):
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Get hand landmarks and create a convex hull
    points = []
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        points.append((x, y))
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply the mask to the frame
    hand_only = cv2.bitwise_and(frame, frame, mask=mask)
    return hand_only

# Open the camera
cap = cv2.VideoCapture(0)

# Set a smaller frame size for faster processing
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Initialize variables for combined bounding box
    x_min_combined, y_min_combined = frame_width, frame_height
    x_max_combined, y_max_combined = 0, 0

    # Initialize a blank canvas for combined hand-only regions
    combined_hands = np.zeros_like(frame)

    # If hands are detected, focus on the hands
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Label the hand (Hand 1 or Hand 2)
            cv2.putText(frame, f"Hand {i + 1}", (x_min, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate hand orientation
            angle = calculate_hand_orientation(hand_landmarks.landmark)
            cv2.putText(frame, f"Angle: {angle:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Remove background and focus on the hand
            hand_only = remove_background(frame, hand_landmarks.landmark)

            # Combine the hand-only regions into a single frame
            combined_hands = cv2.bitwise_or(combined_hands, hand_only)

            # Update the combined bounding box
            x_min_combined = min(x_min_combined, x_min)
            y_min_combined = min(y_min_combined, y_min)
            x_max_combined = max(x_max_combined, x_max)
            y_max_combined = max(y_max_combined, y_max)

    # Crop the frame to focus on the combined region of both hands
    if x_min_combined < x_max_combined and y_min_combined < y_max_combined:
        hand_crop = frame[y_min_combined:y_max_combined, x_min_combined:x_max_combined]

        # Display the cropped hand region
        if hand_crop.size != 0:
            cv2.imshow("Focused Hand Region (Original)", hand_crop)

    # Display the original frame with bounding boxes
    cv2.imshow("Hand Detection", frame)

    # Display the combined hand-only regions
    cv2.imshow("Combined Hands (Background Removed)", combined_hands)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
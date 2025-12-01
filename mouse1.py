import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Variables to keep track of drag state and previous coordinates
drag_active = False

# Capture video from webcam
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)  # Flip horizontally for natural control
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks for index and middle finger tips
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]

            # Convert landmark positions to screen coordinates
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # If two fingers are close together, activate left click drag
            if abs(index_y - middle_y) < 20 and not drag_active:
                # Start dragging
                pyautogui.mouseDown(button="left")
                drag_active = True
                print("Drag started")

            # If index finger is raised alone, stop dragging
            elif abs(index_y - middle_y) > 50 and drag_active:
                # Stop dragging
                pyautogui.mouseUp(button="left")
                drag_active = False
                print("Drag stopped")

            # Map index finger position to screen coordinates to move the mouse
            if drag_active or abs(index_y - middle_y) > 50:
                screen_x = np.interp(index_x, (0, w), (0, screen_width))
                screen_y = np.interp(index_y, (0, h), (0, screen_height))
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Draw landmarks on frame
            draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the video frame
    cv2.imshow("Hand Tracking for Mouse Control", frame)

    # Break loop on 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


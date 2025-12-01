import mediapipe as mp
import cv2
import pyautogui
import numpy as np
import os

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Screen dimensions for scaling
screen_width, screen_height = pyautogui.size()

# Video capture
cap = cv2.VideoCapture(0)

# Constants to track state
clicking = False

# Function to detect if fingers are raised
def fingers_up(hand_landmarks):
    # List of finger status (0 = down, 1 = up)
    finger_status = [0] * 5

    # Check the y-coordinate of the index and middle finger tip and pip joints to see if they are raised
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        finger_status[1] = 1  # Index finger is up
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        finger_status[2] = 1  # Middle finger is up

    return finger_status

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for natural interaction
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and get hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates for the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            # Check which fingers are up
            fingers = fingers_up(hand_landmarks)

            # Single finger up (Index finger): Move the mouse
            if fingers[1] == 1 and fingers[2] == 0:
                pyautogui.moveTo(x, y)

            # Two fingers up (Index and middle fingers): Left click or drag
            elif fingers[1] == 1 and fingers[2] == 1:
                if not clicking:
                    clicking = True
                    pyautogui.mouseDown()
                pyautogui.moveTo(x, y)

            # Reset click status when fingers are down
            else:
                if clicking:
                    clicking = False
                    pyautogui.mouseUp()

    # Display the frame with hand landmarks
    cv2.imshow("Hand Tracking Mouse Control", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


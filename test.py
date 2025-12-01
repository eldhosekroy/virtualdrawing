import mediapipe as mp
import cv2
import numpy as np
import time

# Constants for tool selection and dimensions
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Function to determine the selected tool based on x-coordinate
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"  # Change to fluid simulation
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

# Function to check if index finger is raised
def index_raised(yi, y9):
    return (y9 - yi) > 40

# Initialize MediaPipe Hands and drawing tools
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Load tools image
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

# Mask for drawing
mask = np.ones((480, 640), dtype='uint8') * 255

# Initialize fluid simulation effect screen
fluid_screen = np.zeros((480, 640, 3), dtype='uint8')  # Blank screen for fluid effect

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Current tool selected:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            if curr_tool == "draw":
                # Fluid simulation effect
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    # Draw circles with fading effect on the fluid screen
                    color = (255, 0, 0)  # Blue color for fluid
                    cv2.circle(fluid_screen, (x, y), 20, color, -1)
                    mask[y-10:y+10, x-10:x+10] = 0  # Update mask to show fluid effect

                prevx, prevy = x, y

            elif curr_tool == "line":
                # Line tool logic
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                # Rectangle tool logic
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "circle":
                # Circle tool logic
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)
                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (0, 255, 0), thick)
                        var_inits = False

            elif curr_tool == "erase":
                # Eraser tool logic
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)

    # Blend the fluid effect with the main frame
    fluid_effect = cv2.bitwise_and(fluid_screen, fluid_screen, mask=mask)
    combined_frame = cv2.addWeighted(frm, 0.5, fluid_effect, 0.5, 0)

    # Overlay tools on the combined frame
    combined_frame[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, combined_frame[:max_y, ml:max_x], 0.3, 0)

    cv2.putText(combined_frame, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Fluid Simulation Paint App", combined_frame)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break


import mediapipe as mp
import cv2
import numpy as np
import time
import os

# ------------------ constants / state ------------------
ml = 150
max_x, max_y = 250 + ml, 50  # toolbar region: x in [ml, max_x), y in [0, max_y)
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = None, None

# ------------------ helper functions ------------------
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

def index_raised(yi, y9):
    return (y9 - yi) > 40

# ------------------ mediapipe setup ------------------
mp_hands = mp.solutions.hands
hand_landmark = mp_hands.Hands(min_detection_confidence=0.6,
                               min_tracking_confidence=0.6,
                               max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ------------------ toolbar fallback image (not used for overlay anymore) ------------------
# We'll draw the toolbar directly on the frame to avoid ROI mismatch issues.
# But keep a simple image for potential use.
tools_img = None
if os.path.exists("tools.png"):
    tools_img = cv2.imread("tools.png")
    if tools_img is None:
        tools_img = None

# ------------------ capture ------------------
source = "https://10.168.59.75:8080/video"
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open any video source.")

mask = None

# ------------------ main loop ------------------
while True:
    ret, frm = cap.read()
    if not ret or frm is None:
        if cv2.waitKey(10) == 27:
            break
        continue

    frm = cv2.flip(frm, 1)
    h, w = frm.shape[:2]

    # create/rescale mask if needed
    if mask is None or mask.shape[:2] != (h, w) or mask.dtype != np.uint8:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        prevx, prevy = None, None
        var_inits = False

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    results = hand_landmark.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frm, handLms, mp_hands.HAND_CONNECTIONS)
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # toolbar selection region guard (we'll clamp later too)
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()
                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad = max(1, rad - 1)
                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("[INFO] tool set to:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            xi = int(handLms.landmark[12].x * w)
            yi = int(handLms.landmark[12].y * h)
            y9 = int(handLms.landmark[9].y * h)

            if curr_tool == "draw":
                if index_raised(yi, y9):
                    if prevx is None or prevy is None:
                        prevx, prevy = x, y
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = None, None

            elif curr_tool == "line":
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
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    r = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                    cv2.circle(frm, (xii, yii), r, (255, 255, 0), thick)
                else:
                    if var_inits:
                        r = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                        cv2.circle(mask, (xii, yii), r, 0, thick)
                        var_inits = False

            elif curr_tool == "erase":
                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)

    # Ensure mask type/size
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Safe bitwise_and
    try:
        op = cv2.bitwise_and(frm, frm, mask=mask)
    except cv2.error as e:
        print("[ERROR] bitwise_and failed:", e)
        mask = cv2.convertScaleAbs(mask).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        op = cv2.bitwise_and(frm, frm, mask=mask)

    # preserve some channels as before
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    # ------------------ Robust toolbar drawing (guaranteed visible) ------------------
        # ------------------ Bigger & cleaner toolbar ------------------
    # define a larger toolbar height
    toolbar_height = int(h * 0.12)  # 12% of frame height
    y1 = 0
    y2 = min(h, toolbar_height)
    x1 = 0
    x2 = min(w, 600)  # increase width for bigger icons

    # dark background + bright border
    cv2.rectangle(frm, (x1, y1), (x2 - 1, y2 - 1), (30, 30, 30), -1)
    cv2.rectangle(frm, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)

    # 5 tool icons
    num_tools = 5
    cell_w = (x2 - x1) // num_tools
    icon_y = y1 + int(toolbar_height * 0.55)

    for t in range(num_tools):
        cx = x1 + t * cell_w + cell_w // 2
        # separator lines
        if t > 0:
            cv2.line(frm, (x1 + t * cell_w, y1), (x1 + t * cell_w, y2), (0, 180, 180), 1)

        # draw colorful icons (no text)
        if t == 0:  # Line
            cv2.line(frm, (cx - 35, icon_y), (cx + 35, icon_y), (0, 255, 255), 4)
        elif t == 1:  # Rectangle
            cv2.rectangle(frm, (cx - 30, icon_y - 20), (cx + 30, icon_y + 20), (0, 255, 255), 4)
        elif t == 2:  # Draw (curve)
            pts = np.array([[cx - 35, icon_y + 10],
                            [cx - 10, icon_y - 25],
                            [cx + 10, icon_y + 25],
                            [cx + 35, icon_y - 15]], np.int32)
            cv2.polylines(frm, [pts], False, (0, 255, 255), 4)
        elif t == 3:  # Circle
            cv2.circle(frm, (cx, icon_y), 28, (0, 255, 255), 4)
        elif t == 4:  # Eraser
            cv2.rectangle(frm, (cx - 25, icon_y - 25), (cx + 25, icon_y + 25), (0, 100, 255), -1)
            cv2.rectangle(frm, (cx - 25, icon_y - 25), (cx + 25, icon_y + 25), (255, 255, 0), 2)

    # highlight current tool
    sel_idx = {"line":0, "rectangle":1, "draw":2, "circle":3, "erase":4}.get(curr_tool, -1)
    if 0 <= sel_idx < num_tools:
        sx1 = x1 + sel_idx * cell_w + 2
        sx2 = x1 + (sel_idx + 1) * cell_w - 2
        cv2.rectangle(frm, (sx1, y1 + 2), (sx2, y2 - 2), (0, 120, 255), 3)
    # ------------------ end toolbar drawing ------------------

    # ------------------ end toolbar drawing ------------------

    cv2.putText(frm, curr_tool, (min(w-200, ml+120), min(h-10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("paint app", frm)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'):
        mask[:] = 255
        prevx, prevy = None, None
        var_inits = False

cap.release()
cv2.destroyAllWindows()


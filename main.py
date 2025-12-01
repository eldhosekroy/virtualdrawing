import mediapipe as mp
import cv2
import numpy as np
import time

# ------------------ constants / state ------------------
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = None, None

# Undo stack
undo_stack = []
UNDO_LIMIT = 20

# Gesture edge trackers
last_index_raised = False  # for freehand/erase rising edge

# ------------------ helper functions ------------------
def index_raised(yi, y9):
    return (y9 - yi) > 40

def tool_index_from_y(y, toolbar_y1, toolbar_h, num_tools=7):
    if y < toolbar_y1 or y >= toolbar_y1 + toolbar_h:
        return -1
    rel_y = y - toolbar_y1
    cell_h = max(1, toolbar_h // num_tools)
    idx = int(rel_y // cell_h)
    if idx < 0:
        idx = 0
    if idx >= num_tools:
        idx = num_tools - 1
    return idx

def tool_name_from_index(idx):
    return {0: "undo", 1: "all-clear", 2: "line", 3: "rectangle", 4: "draw", 5: "circle", 6: "erase"}.get(idx, "select tool")

def push_undo(mask):
    global undo_stack
    if mask is None:
        return
    if len(undo_stack) >= UNDO_LIMIT:
        undo_stack.pop(0)
    undo_stack.append(mask.copy())

def undo(mask):
    global undo_stack
    if not undo_stack:
        return None
    restored = undo_stack.pop()
    return restored

# ------------------ mediapipe setup ------------------
mp_hands = mp.solutions.hands
hand_landmark = mp_hands.Hands(min_detection_confidence=0.6,
                               min_tracking_confidence=0.6,
                               max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ------------------ capture ------------------
source = "https://10.140.217.200:8080/video"
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source.")

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

    # toolbar geometry (vertical left)
    toolbar_width = min(100, w//5)
    toolbar_x1 = 0
    toolbar_x2 = toolbar_x1 + toolbar_width
    toolbar_y1 = 0
    toolbar_y2 = h
    num_tools = 7
    cell_h = toolbar_y2 // num_tools

    # create/rescale mask if needed
    if mask is None or mask.shape[:2] != (h, w) or mask.dtype != np.uint8:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        prevx, prevy = None, None
        var_inits = False
        undo_stack = []

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    results = hand_landmark.process(rgb)

    current_index_raised = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frm, handLms, mp_hands.HAND_CONNECTIONS)

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # selection logic for vertical toolbar
            sel_idx = tool_index_from_y(y, toolbar_y1, toolbar_y2, num_tools)
            inside_toolbar_horizontally = (x >= toolbar_x1 and x < toolbar_x2)

            if sel_idx != -1 and inside_toolbar_horizontally:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()
                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad = max(1, rad - 1)
                if (ptime - ctime) > 0.8:
                    curr_tool = tool_name_from_index(sel_idx)
                    print("[INFO] Tool set to:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            xi = int(handLms.landmark[12].x * w)
            yi = int(handLms.landmark[12].y * h)
            y9 = int(handLms.landmark[9].y * h)

            current_index_raised = index_raised(yi, y9)

            # freehand draw
            if curr_tool == "draw":
                if current_index_raised and not last_index_raised:
                    push_undo(mask)
                if current_index_raised:
                    if prevx is None or prevy is None:
                        prevx, prevy = x, y
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = None, None

            # erase
            elif curr_tool == "erase":
                if current_index_raised and not last_index_raised:
                    push_undo(mask)
                if current_index_raised:
                    cv2.circle(frm, (x, y), 30, (0,0,0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)

            # line
            elif curr_tool == "line":
                if current_index_raised:
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            # rectangle
            elif curr_tool == "rectangle":
                if current_index_raised:
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            # circle
            elif curr_tool == "circle":
                if current_index_raised:
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    r = int(((xii - x)**2 + (yii - y)**2)**0.5)
                    cv2.circle(frm, (xii, yii), r, (255,255,0), thick)
                else:
                    if var_inits:
                        r = int(((xii - x)**2 + (yii - y)**2)**0.5)
                        cv2.circle(mask, (xii, yii), r, 0, thick)
                        var_inits = False

            # handle Undo/All-Clear gestures
            if curr_tool == "undo":
                restored = undo(mask)
                if restored is not None:
                    mask = restored.copy()
                    prevx, prevy = None, None
                    var_inits = False
                curr_tool = "select tool"

            elif curr_tool == "all-clear":
                push_undo(mask)
                mask[:] = 255
                prevx, prevy = None, None
                var_inits = False
                curr_tool = "select tool"

            last_index_raised = current_index_raised

    # Apply mask
    try:
        op = cv2.bitwise_and(frm, frm, mask=mask)
    except cv2.error:
        mask = mask.astype(np.uint8)
        op = cv2.bitwise_and(frm, frm, mask=mask)

    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    # Draw vertical toolbar
    cv2.rectangle(frm, (toolbar_x1, toolbar_y1), (toolbar_x2-1, toolbar_y2-1), (30,30,30), -1)
    cv2.rectangle(frm, (toolbar_x1, toolbar_y1), (toolbar_x2-1, toolbar_y2-1), (0,255,255), 3)
    for t in range(num_tools):
        cy = toolbar_y1 + t * cell_h + cell_h//2
        cx = toolbar_x1 + toolbar_width//2
        if t > 0:
            cv2.line(frm, (toolbar_x1, toolbar_y1 + t*cell_h), (toolbar_x2, toolbar_y1 + t*cell_h), (0,150,150), 2)
        # simple icons
        if t == 0:  # undo
            cv2.putText(frm, "U", (cx-15, cy+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        elif t == 1:  # all-clear
            cv2.putText(frm, "AC", (cx-20, cy+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        elif t == 2: cv2.line(frm, (cx-25, cy), (cx+25, cy), (0,255,255), 6)
        elif t == 3: cv2.rectangle(frm, (cx-25, cy-15), (cx+25, cy+15), (0,255,255), 6)
        elif t == 4: pts = np.array([[cx-25, cy+10],[cx-10, cy-15],[cx+10, cy+15],[cx+25, cy-10]], np.int32); cv2.polylines(frm,[pts],False,(0,255,255),6)
        elif t == 5: cv2.circle(frm, (cx, cy), 20, (0,255,255), 6)
        elif t == 6: cv2.rectangle(frm, (cx-20, cy-20), (cx+20, cy+20), (0,100,255), -1)

    # Highlight current tool
    sel_idx = {"line":2,"rectangle":3,"draw":4,"circle":5,"erase":6}.get(curr_tool,-1)
    if 0 <= sel_idx < num_tools:
        cy1 = toolbar_y1 + sel_idx*cell_h + 6
        cy2 = toolbar_y1 + (sel_idx+1)*cell_h - 6
        cv2.rectangle(frm, (toolbar_x1+6, cy1), (toolbar_x2-6, cy2), (0,120,255), 4)

    cv2.imshow("paint app", frm)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        mask[:] = 255
        prevx, prevy = None, None
        var_inits = False
    elif key == ord('a'):
        push_undo(mask)
        mask[:] = 255
        prevx, prevy = None, None
        var_inits = False
    elif key == ord('u'):
        restored = undo(mask)
        if restored is not None:
            mask = restored.copy()
            prevx, prevy = None, None
            var_inits = False

cap.release()
cv2.destroyAllWindows()


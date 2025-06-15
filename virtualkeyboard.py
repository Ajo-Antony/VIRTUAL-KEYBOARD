import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M", "Backspace", "Space"]]

typed_text = ""
last_click_time = 0

# Button class
class Button:
    def __init__(self, pos, text, size=[40, 40]):
        self.pos = pos
        self.text = text
        self.size = size

# Detect pinch gesture (click)
def detect_click(lmList):
    if len(lmList) > 8:
        x1, y1 = lmList[8][1], lmList[8][2]  # Index tip
        x2, y2 = lmList[4][1], lmList[4][2]  # Thumb tip
        distance = np.hypot(x2 - x1, y2 - y1)
        return distance < 30
    return False

# Draw keyboard
def draw_keyboard(img, button_list):
    overlay = img.copy()
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)
        font_scale = 0.7 if len(button.text) <= 2 else 0.5
        cv2.putText(overlay, button.text, (x + 5, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

# Fixed keyboard position
anchor_x, anchor_y = 50, 150

# Create fixed-position button list
button_list = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        key_width = 90 if key in ["Space", "Backspace"] else 40
        btn = Button([anchor_x + j * 50, anchor_y + i * 60], key, [key_width, 40])
        button_list.append(btn)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    all_lmLists = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            all_lmLists.append(lmList)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Draw the keyboard
    img = draw_keyboard(img, button_list)

    # Check interaction for each hand
    for lmList in all_lmLists:
        if len(lmList) < 9:
            continue

        x, y = lmList[8][1], lmList[8][2]  # Index tip
        cv2.circle(img, (x, y), 10, (0, 255, 255), cv2.FILLED)

        for button in button_list:
            bx, by = button.pos
            bw, bh = button.size

            if bx < x < bx + bw and by < y < by + bh:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                if detect_click(lmList) and time.time() - last_click_time > 0.5:
                    key = button.text
                    if key == "Backspace":
                        typed_text = typed_text[:-1]
                    elif key == "Space":
                        typed_text += " "
                    else:
                        typed_text += key
                    last_click_time = time.time()

    # Display typed text at the top
    cv2.rectangle(img, (20, 20), (w - 20, 70), (255, 255, 255), -1)
    cv2.putText(img, typed_text[-50:], (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Two-Hand Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()

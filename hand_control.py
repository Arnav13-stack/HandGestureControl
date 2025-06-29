import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import subprocess
import time
import math

# Initialize
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

click_down = False
fist_action_done = False

def get_landmark_pos(landmark, frame_shape):
    h, w, _ = frame_shape
    return int(landmark.x * w), int(landmark.y * h)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Finger detection
            finger_tips = [8, 12, 16, 20]
            finger_joints = [6, 10, 14, 18]
            fingers_extended = []

            for tip, joint in zip(finger_tips, finger_joints):
                tip_y = hand_landmarks.landmark[tip].y
                joint_y = hand_landmarks.landmark[joint].y
                fingers_extended.append(tip_y < joint_y)

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            index_x, index_y = get_landmark_pos(index_tip, frame.shape)
            middle_x, middle_y = get_landmark_pos(middle_tip, frame.shape)

            all_fingers_up = all(fingers_extended)
            all_fingers_down = not any(fingers_extended)

            # === All fingers up → Move mouse ===
            if all_fingers_up:
                screen_x = np.interp(index_x, [0, frame.shape[1]], [0, screen_w])
                screen_y = np.interp(index_y, [0, frame.shape[0]], [0, screen_h])
                pyautogui.moveTo(screen_x, screen_y)
                cv2.putText(frame, "All Fingers Up - Moving Cursor", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # === Fist - Minimize current window ===
            if all_fingers_down:
                cv2.putText(frame, "Fist - Minimize Window", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                if not fist_action_done:
                    pyautogui.hotkey('alt', 'space')
                    time.sleep(0.1)
                    pyautogui.press('n')
                    fist_action_done = True
            else:
                fist_action_done = False

            # === Click - Index & Middle touch ===
            distance = np.hypot(middle_x - index_x, middle_y - index_y)
            if distance < 40:
                if not click_down:
                    pyautogui.click()
                    click_down = True
                    cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
            else:
                click_down = False

            # === Scroll - Index + Middle Extended ===
            if fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
                scroll_y = middle_y - index_y
                if abs(scroll_y) > 15:
                    direction = -1 if scroll_y > 0 else 1
                    pyautogui.scroll(direction * 40)
                    cv2.putText(frame, "Scrolling", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            # === Volume Control - Pinch with Thumb + Index ===
            thumb_x, thumb_y = get_landmark_pos(thumb_tip, frame.shape)
            pinch_distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if pinch_distance < 100:
                cv2.putText(frame, "Volume Control", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 0), 2)
                if pinch_distance < 40:
                    pyautogui.press("volumedown")
                elif pinch_distance > 70:
                    pyautogui.press("volumeup")

            # === Close Window - Tilt Left/Right ===
            wrist_x = hand_landmarks.landmark[0].x
            mid_x = hand_landmarks.landmark[9].x
            tilt = wrist_x - mid_x

            if abs(tilt) > 0.2:
                direction = "Right Tilt" if tilt < 0 else "Left Tilt"
                cv2.putText(frame, f"Palm Tilt - {direction} - Close App", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
                pyautogui.hotkey('alt', 'f4')
                time.sleep(1)

    cv2.imshow("Hand Gesture Controller", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

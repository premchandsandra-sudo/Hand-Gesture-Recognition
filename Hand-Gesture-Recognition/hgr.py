import cv2
import math
from collections import deque
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def fingers_up(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers = [0, 0, 0, 0, 0]

    thumb_tip = lm[TIP_IDS[0]]
    thumb_ip  = lm[PIP_IDS[0]]
    if handedness_label == "Right":
        fingers[0] = 1 if thumb_tip.x < thumb_ip.x - 0.02 else 0
    else:
        fingers[0] = 1 if thumb_tip.x > thumb_ip.x + 0.02 else 0

    for i in range(1, 5):
        if lm[TIP_IDS[i]].y < lm[PIP_IDS[i]].y - 0.02:
            fingers[i] = 1

    return fingers

def classify_gesture(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    f = fingers_up(hand_landmarks, handedness_label)
    up_count = sum(f)

    if up_count == 0:
        return "Fist (0)"
    if up_count == 5:
        return "Open Palm (5)"
    if up_count == 2 and f[1] and f[2]:
        return "Victory"
    if up_count == 1 and f[1]:
        return "One (Index)"
    if up_count == 1 and f[0]:
        wrist = lm[0]
        thumb_tip = lm[4]
        if thumb_tip.y < wrist.y - 0.04:
            return "Thumbs Up"
        elif thumb_tip.y > wrist.y + 0.04:
            return "Thumbs Down"
        else:
            return "Thumb (Side)"
    if up_count == 3 and f[1] and f[2] and f[3]:
        return "Three"
    if up_count == 4 and not f[0] and all(f[1:]):
        return "Four"
    if dist(lm[4], lm[8]) < 0.05:
        return "OK"
    if f[1] and f[4] and not f[2] and not f[3]:
        return "Rock"

    return f"{up_count} finger(s)"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found. Try index 1 or 2.")
        return

    label_history = deque(maxlen=7)
    count_history = deque(maxlen=7)

    cv2.namedWindow("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            h, w = frame.shape[:2]

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    gesture = classify_gesture(hand_landmarks, label)
                    fingers = fingers_up(hand_landmarks, label)
                    count = sum(fingers)

                    label_history.append(gesture)
                    count_history.append(count)

                    stable_label = max(set(label_history), key=label_history.count)
                    stable_count = max(set(count_history), key=count_history.count)

                    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                    cv2.rectangle(frame, (x1 - 6, y1 - 40), (x2 + 6, y1 - 8), (0, 0, 0), -1)
                    cv2.putText(frame, f"{label} | {stable_count} | {stable_label}",
                                (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

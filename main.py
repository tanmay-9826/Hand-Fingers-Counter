import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Finger tip IDs
tips = [4, 8, 12, 16, 20]

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            finger_count = 0
            lm = hand_landmarks.landmark
            hand_label = handedness.classification[0].label # Left or Right

            # Draw hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           # Thumb logic
            if hand_label == "Right":
                # In a mirrored view, the "Right" hand thumb points LEFT
                if lm[tips[0]].x < lm[tips[0] - 2].x:
                    finger_count += 1
            else: # Left hand
                # In a mirrored view, the "Left" hand thumb points RIGHT
                if lm[tips[0]].x > lm[tips[0] - 2].x:
                    finger_count += 1

            # Other 4 fingers
            for i in range(1, 5):
                if lm[tips[i]].y < lm[tips[i] - 2].y:
                    finger_count += 1

            total_fingers += finger_count

            # Display per-hand count
            cv2.putText(
                frame,
                f"{hand_label}: {finger_count}",
                (20, 100 if hand_label=="Left" else 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

    # Display total count
    cv2.putText(
        frame,
        f"Total Fingers: {total_fingers}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Both Hands Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
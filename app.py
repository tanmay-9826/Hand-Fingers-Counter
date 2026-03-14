import cv2
import mediapipe as mp
import gradio as gr

# 1. Initialize MediaPipe (done globally so it doesn't reload every frame)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
tips = [4, 8, 12, 16, 20]

# 2. The core processing function
def count_fingers(frame):
    # Gradio sometimes sends an empty frame during initialization
    if frame is None:
        return None
        
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Gradio passes frames as RGB, which MediaPipe loves!
    result = hands.process(frame)
    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            finger_count = 0
            lm = hand_landmarks.landmark
            hand_label = handedness.classification[0].label 

            # Draw hand connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb logic (Mirrored view)
            if hand_label == "Right":
                if lm[tips[0]].x < lm[tips[0] - 2].x:
                    finger_count += 1
            else: # Left hand
                if lm[tips[0]].x > lm[tips[0] - 2].x:
                    finger_count += 1

            # Other 4 fingers
            for i in range(1, 5):
                if lm[tips[i]].y < lm[tips[i] - 2].y:
                    finger_count += 1

            total_fingers += finger_count

            # Display per-hand count
            # Note: Colors are in RGB format for Gradio, so (255, 0, 0) is Red
            cv2.putText(
                frame, f"{hand_label}: {finger_count}",
                (20, 100 if hand_label=="Left" else 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )

    # Display total count
    cv2.putText(
        frame, f"Total Fingers: {total_fingers}",
        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
    )

    return frame

# 3. Build the Gradio Interface
demo = gr.Interface(
    fn=count_fingers,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs="image",
    live=True,               # <--- This makes it run in real-time
    flagging_mode="never",  # <--- This removes the confusing Flag button
    title="Real-Time Both Hands Finger Counter",
    description="Click the 'Record' button below the dark box to turn on your camera!"
)

if __name__ == "__main__":
    demo.launch()
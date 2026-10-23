import os
# Use PyTorch as the Keras backend (TensorFlow is not installed).
# Must be set before importing keras.
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import torch
import mediapipe as mp
import numpy as np
import pyttsx3
import keras   # standalone Keras with PyTorch backend
from sklearn.preprocessing import LabelEncoder
import pickle

# -------------------------------
# Phase 1: Hand Detection (MediaPipe)
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Phase 2 & 3: Load AI Detector (Deep Learning Model)
# -------------------------------
MODEL_PATH = "sign_detector.h5"
ENCODER_PATH = "label_encoder.pkl"

model: keras.Model | None = None
encoder: LabelEncoder | None = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
else:
    print("⚠️ Trained model not found. Run: python train_model.py")
    raise SystemExit(1)

# -------------------------------
# Phase 4: Recognition → Text
# -------------------------------
def recognize_sign(landmarks: list[list[float]]) -> str | None:
    if not landmarks or model is None:
        return None
    data = np.array(landmarks, dtype=np.float32).flatten().reshape(1, -1)
    prediction = model.predict(data, verbose=0)
    sign_text = encoder.inverse_transform([int(np.argmax(prediction))])[0]
    return sign_text

# -------------------------------
# Phase 5: Text → Speech
# -------------------------------
engine = pyttsx3.init()

def speak_text(text: str) -> None:
    if text:
        engine.say(text)
        engine.runAndWait()

# -------------------------------
# Phase 6: Integration (Live Webcam)
# -------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW avoids warnings on Windows

    last_sign = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmarks: list[list[float]] = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize sign
            sign_text = recognize_sign(landmarks)
            if sign_text and sign_text != last_sign:
                cv2.putText(frame, sign_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                speak_text(sign_text)
                last_sign = sign_text
        else:
            last_sign = None

        cv2.imshow("Sign Language Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()

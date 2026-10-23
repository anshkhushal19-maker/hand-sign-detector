"""
Sign Language Recognition - Model Training Script
==================================================
This script:

  1. COLLECTS hand-landmark samples from your webcam for each sign class.
  2. TRAINS a neural network (Keras with the PyTorch backend) on those samples.
  3. SAVES the trained model (sign_detector.h5) and label encoder
     (label_encoder.pkl) so that script.py can perform live recognition.

Classes trained: Hello, Yes, No, ThankYou
"""

import os
# Use PyTorch as the Keras backend (TensorFlow is not installed).
os.environ["KERAS_BACKEND"] = "torch"

import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Configuration
# -------------------------------
CLASSES = ["Hello", "Yes", "No", "ThankYou"]
SAMPLES_PER_CLASS = 100       # samples to capture per class (raise for better accuracy)
EPOCHS = 60
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
MODEL_PATH = "sign_detector.h5"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = "landmark_data.pkl"   # cache of collected data

NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 2      # x and y per landmark -> 42 features

# -------------------------------
# MediaPipe setup
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
# Phase A: Data Collection
# -------------------------------
def collect_data() -> tuple[np.ndarray, np.ndarray]:
    """Capture hand-landmark samples from the webcam for each class.

    Controls:
      SPACE  -> save the current hand pose as a sample
      N      -> move to the next class (after enough samples)
      Q      -> quit early and use whatever has been collected
    """
    data: list[np.ndarray] = []
    labels: list[str] = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("=" * 60)
    print("DATA COLLECTION")
    print("=" * 60)
    print("Show the sign to the camera and press SPACE to save a sample.")
    print("After enough samples, press N to go to the next sign.")
    print("Press Q at any time to stop collecting.\n")

    try:
        for class_idx, class_name in enumerate(CLASSES):
            samples = 0
            print(f"\n>>> Now collecting: '{class_name}'  (press N when done / Q to quit)")

            while samples < SAMPLES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)  # mirror view feels natural
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                landmarks = None
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    landmarks = np.array(
                        [[lm.x, lm.y] for lm in hand.landmark], dtype=np.float32
                    ).flatten()

                # On-screen status
                info = f"Class: {class_name}   Samples: {samples}/{SAMPLES_PER_CLASS}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
                if landmarks is not None:
                    cv2.putText(frame, "HAND DETECTED - press SPACE", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "NO HAND - show your hand", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("Collect Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and landmarks is not None:
                    data.append(landmarks)
                    labels.append(class_name)
                    samples += 1
                    print(f"   saved sample {samples}/{SAMPLES_PER_CLASS}")
                elif key == ord('n'):
                    break
                elif key == ord('q'):
                    print("\nQuit requested.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return _package_data(data, labels)
            # end while

            if samples < SAMPLES_PER_CLASS:
                print(f"   (collected {samples} samples for '{class_name}')")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return _package_data(data, labels)


def _package_data(data: list[np.ndarray], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Turn the collected lists into numpy arrays, with mirrored augmentation."""
    if not data:
        raise RuntimeError("No data collected. Please run data collection again.")

    X_raw = np.array(data, dtype=np.float32)          # shape (n, 42)
    y_raw = np.array(labels)

    # Horizontal mirror augmentation (simulate the other hand):
    # flip x coordinates (indices 0, 2, 4, ... -> x = 1 - x)
    X_flip = X_raw.copy()
    X_flip[:, 0::2] = 1.0 - X_flip[:, 0::2]
    X = np.vstack([X_raw, X_flip])
    y = np.concatenate([y_raw, y_raw])

    print(f"\nTotal samples after augmentation: {len(X)} "
          f"(from {len(X_raw)} raw, x2 via mirroring)")
    return X, y


# -------------------------------
# Phase B: Model Building & Training
# -------------------------------
def build_model(num_classes: int) -> keras.Model:
    """Build a small feed-forward network for landmark classification."""
    model = keras.Sequential([
        layers.Input(shape=(FEATURES,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(X: np.ndarray, y_int: np.ndarray) -> keras.Model:
    """Train the model and return the fitted Keras model."""
    model = build_model(num_classes=len(CLASSES))
    model.summary()

    print("\nTraining started...")
    history = model.fit(
        X, y_int,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    # Quick accuracy report
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_val_acc * 100:.2f}%")
    return model


# -------------------------------
# Phase C: Save Artifacts
# -------------------------------
def save_artifacts(model: keras.Model, encoder: LabelEncoder) -> None:
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    print(f"Saved model      -> {MODEL_PATH}")
    print(f"Saved encoder    -> {ENCODER_PATH}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SIGN LANGUAGE MODEL TRAINER")
    print("=" * 60)
    print(f"Classes: {CLASSES}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")

    # 1) Collect data
    X, y = collect_data()

    # 2) Encode string labels -> integers
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)
    print("Label mapping:")
    for i, cls in enumerate(encoder.classes_):
        print(f"   {i} -> {cls}")

    # 3) Train
    model = train_model(X, y_int)

    # 4) Save
    save_artifacts(model, encoder)

    print("\nDone! Now run:  python sign_language_detector.py   to test live recognition.")


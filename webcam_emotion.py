import csv
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==============================
# CONFIGURATION
# ==============================

PROJECT_DIR = Path(r"C:\Users\mpste\CPSC483_Project")

MODEL_PATH = PROJECT_DIR / "models" / "rafdb_emotion_model.keras"
LABEL_PATH = PROJECT_DIR / "models" / "emotion_labels.txt"

LOG_DIR = PROJECT_DIR / "logs"
LOG_PATH = LOG_DIR / "emotion_log.csv"

IMG_SIZE = 224
SMOOTHING_FRAMES = 10
CONFIDENCE_THRESHOLD = 0.40

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD MODEL AND LABELS
# ==============================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    emotion_labels = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

print("Loaded model:", MODEL_PATH)
print("Loaded labels:", emotion_labels)


# ==============================
# FACE DETECTOR
# ==============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade face detector.")


# ==============================
# LOG FILE
# ==============================

new_log_file = not LOG_PATH.exists()

log_file = open(LOG_PATH, "a", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)

if new_log_file:
    log_writer.writerow([
        "timestamp",
        "face_id",
        "emotion",
        "confidence"
    ])


# ==============================
# HELPER FUNCTIONS
# ==============================

def predict_face(face_bgr):
    """
    Takes a cropped BGR face image from OpenCV.
    Returns prediction probabilities.
    """

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

    x = face_rgb.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    probs = model.predict(x, verbose=0)[0]

    return probs


def draw_label(frame, text, x, y):
    """
    Draws a readable text label above the face box.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 8),
        (x + text_width + 8, y),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 6),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )


# ==============================
# WEBCAM LOOP
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1)."
    )

prediction_history = deque(maxlen=SMOOTHING_FRAMES)

print("Webcam started.")
print("Press q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        for face_id, (x, y, w, h) in enumerate(faces):
            face_bgr = frame[y:y + h, x:x + w]

            if face_bgr.size == 0:
                continue

            probs = predict_face(face_bgr)

            prediction_history.append(probs)

            avg_probs = np.mean(prediction_history, axis=0)

            pred_index = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_index])
            emotion = emotion_labels[pred_index]

            if confidence < CONFIDENCE_THRESHOLD:
                label = f"uncertain: {confidence * 100:.1f}%"
            else:
                label = f"{emotion}: {confidence * 100:.1f}%"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            draw_label(frame, label, x, y)

            log_writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                face_id,
                emotion,
                round(confidence, 4)
            ])

            log_file.flush()

        cv2.imshow(
            "RAF-DB Real-Time Facial Expression Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    log_file.close()
    cv2.destroyAllWindows()

    print("Webcam closed.")
    print("Log saved to:", LOG_PATH)
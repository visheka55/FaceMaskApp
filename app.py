import os
import sys
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pygame

# 🔥 Function to handle file paths (important for .exe)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # when running exe
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 🔊 Initialize sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(resource_path("alert.wav"))

# 🤖 Load model
model = load_model(resource_path("mask_detector.model"))

# 😀 Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 📷 Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Resize + normalize
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        prediction = model.predict(face_img, verbose=0)
        mask, withoutMask = prediction[0]

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw result
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # 🔊 Play alert (only if not already playing)
        if label == "No Mask":
            if not pygame.mixer.get_busy():
                alert_sound.play()

    # Show output
    cv2.imshow("Face Mask Detector", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
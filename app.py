import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)

# Load model
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model = load_model(resource_path("mask_detector.model"))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/")
def home():
    return jsonify({"message": "Face Mask Detector API Running!"})

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json["image"]
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            prediction = model.predict(face_img, verbose=0)
            mask, withoutMask = prediction[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            alert = label == "No Mask"

            results.append({
                "label": label,
                "alert": alert,
                "confidence": float(mask)
            })

        return jsonify({
            "faces_detected": len(faces),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)

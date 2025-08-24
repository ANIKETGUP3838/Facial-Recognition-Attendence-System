import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# -----------------------
# Prepare training data
# -----------------------
def prepare_training_data(data_dir="images"):
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces, labels = [], []
    label_dict = {}
    current_label = 0

    data_dir = Path(data_dir)
    if not data_dir.exists():
        return [], [], {}

    for person_dir in data_dir.iterdir():
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        label_dict[current_label] = person_name

        for img_file in person_dir.glob("*.*"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            if len(faces_rects) == 0:
                continue

            for (x, y, w, h) in faces_rects:
                face = gray[y:y+h, x:x+w]
                faces.append(face)
                labels.append(current_label)

        current_label += 1

    return faces, np.array(labels), label_dict


# -----------------------
# Train recognizer
# -----------------------
faces, labels, label_dict = prepare_training_data("images")

if len(faces) == 0:
    recognizer = None
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

# -----------------------
# Attendance File
# -----------------------
attendance_file = "attendance.csv"
if os.path.exists(attendance_file):
    attendance = pd.read_csv(attendance_file)
else:
    attendance = pd.DataFrame(columns=["Name", "Date", "Time"])

# -----------------------
# Streamlit UI
# -----------------------
st.title("📸 Facial Recognition Attendance System")

if recognizer is None:
    st.error("⚠️ No valid faces found in training images. Please add clear face images in `images/` folder.")
else:
    st.subheader("🎥 Capture a photo to mark attendance")
    img_file = st.camera_input("Take a photo")

    if img_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        ).detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces_rects:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 70:  # lower = better
                name = label_dict[label]

                # Mark attendance
                today = datetime.now().strftime("%Y-%m-%d")
                now_time = datetime.now().strftime("%H:%M:%S")

                if not ((attendance["Name"] == name) & (attendance["Date"] == today)).any():
                    new_row = {"Name": name, "Date": today, "Time": now_time}
                    attendance = pd.concat([attendance, pd.DataFrame([new_row])], ignore_index=True)
                    attendance.to_csv(attendance_file, index=False)

                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

# -----------------------
# Show Attendance Table
# -----------------------
st.subheader("📋 Attendance Log")
st.dataframe(attendance)

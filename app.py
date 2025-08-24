import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# -----------------------
# Load known faces
# -----------------------
path = "images"  # folder with subfolders for each person
known_faces = []
known_names = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                known_faces.append(encodings[0])
                person_name = os.path.basename(root)  # folder name = person name
                known_names.append(person_name)

# -----------------------
# Attendance DataFrame
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

mode = st.radio("Choose Camera Mode", ["📷 Use Browser Camera", "💻 Use Local Webcam"])

FRAME_WINDOW = st.image([])

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    if not ((attendance["Name"] == name) & (attendance["Date"] == today)).any():
        new_row = {"Name": name, "Date": today, "Time": now_time}
        global attendance
        attendance = pd.concat([attendance, pd.DataFrame([new_row])], ignore_index=True)
        attendance.to_csv(attendance_file, index=False)

# -----------------------
# Browser Camera (Streamlit Cloud safe)
# -----------------------
if mode == "📷 Use Browser Camera":
    img_file = st.camera_input("Take a photo")
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
                name = known_names[best_match_index]
                mark_attendance(name)

            top, right, bottom, left = face_loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# -----------------------
# Local Webcam (Only works locally)
# -----------------------
elif mode == "💻 Use Local Webcam":
    run = st.checkbox("Start Webcam")
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Cannot access webcam")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_loc in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
                    name = known_names[best_match_index]
                    mark_attendance(name)

                top, right, bottom, left = face_loc
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

# -----------------------
# Show Attendance Log
# -----------------------
st.subheader("📋 Attendance Log")
st.dataframe(attendance)

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
path = "images"  # folder with known faces
known_faces = []
known_names = []

if os.path.exists(path):
    for filename in os.listdir(path):
        img = face_recognition.load_image_file(f"{path}/{filename}")
        encodings = face_recognition.face_encodings(img)
        if encodings:  # avoid error if no face found
            known_faces.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

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

run_camera = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
cap = None

if run_camera:
    cap = cv2.VideoCapture(0)

if cap is not None and run_camera:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_names[best_match_index]

                # Mark attendance if not already done today
                today = datetime.now().strftime("%Y-%m-%d")
                now_time = datetime.now().strftime("%H:%M:%S")

                if not ((attendance["Name"] == name) & (attendance["Date"] == today)).any():
                    new_row = {"Name": name, "Date": today, "Time": now_time}
                    attendance = pd.concat([attendance, pd.DataFrame([new_row])], ignore_index=True)
                    attendance.to_csv(attendance_file, index=False)

            # Draw box + name
            top, right, bottom, left = face_loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# -----------------------
# Show Attendance Table
# -----------------------
st.subheader("📋 Attendance Log")
st.dataframe(attendance)

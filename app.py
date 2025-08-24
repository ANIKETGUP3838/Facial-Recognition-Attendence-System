import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from deepface import DeepFace

# -----------------------
# Load known faces
# -----------------------
path = "images"  # folder with known faces
known_faces = []
known_names = []

for filename in os.listdir(path):
    known_faces.append(os.path.join(path, filename))
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
st.title("📸 Facial Recognition Attendance System (DeepFace)")

run_camera = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run_camera:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Verify face against database
            for idx, db_img in enumerate(known_faces):
                result = DeepFace.verify(rgb_frame, db_img, enforce_detection=False)

                if result["verified"]:
                    name = known_names[idx]

                    # Mark attendance if not already done today
                    today = datetime.now().strftime("%Y-%m-%d")
                    now_time = datetime.now().strftime("%H:%M:%S")

                    if not ((attendance["Name"] == name) & (attendance["Date"] == today)).any():
                        new_row = {"Name": name, "Date": today, "Time": now_time}
                        attendance = pd.concat([attendance, pd.DataFrame([new_row])], ignore_index=True)
                        attendance.to_csv(attendance_file, index=False)

                    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                    break

        except Exception as e:
            st.write("No face detected:", e)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# -----------------------
# Show Attendance Table
# -----------------------
st.subheader("📋 Attendance Log")
st.dataframe(attendance)

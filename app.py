import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------
# Load known faces
# -----------------------
path = "images"  # each subfolder is a person
known_faces = []
known_names = []

for person_name in os.listdir(path):
    person_folder = os.path.join(path, person_name)
    if os.path.isdir(person_folder):
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_faces.append(encodings[0])
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
# Mark Attendance Function
# -----------------------
def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    if not ((attendance["Name"] == name) & (attendance["Date"] == today)).any():
        new_row = {"Name": name, "Date": today, "Time": now_time}
        attendance.loc[len(attendance)] = new_row
        attendance.to_csv(attendance_file, index=False)

# -----------------------
# Video Transformer
# -----------------------
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches and matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)

            # Draw rectangle + name
            top, right, bottom, left = face_loc
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

        return img

# -----------------------
# Streamlit UI
# -----------------------
st.title("📸 Facial Recognition Attendance System")

st.info("Click **Start** to open camera and detect faces.")

webrtc_streamer(
    key="face-recognition",
    video_transformer_factory=FaceRecognitionTransformer
)

st.subheader("📋 Attendance Log")
st.dataframe(attendance)

import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

# ------------------------------
# Paths
# ------------------------------
IMAGE_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"

# Ensure folders exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# ------------------------------
# Helper Functions
# ------------------------------
def train_model():
    faces = []
    labels = []
    label_dict = {}
    current_id = 0

    for person_name in os.listdir(IMAGE_DIR):
        person_folder = os.path.join(IMAGE_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_dict[current_id] = person_name

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_id)

        current_id += 1

    if len(faces) == 0:
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    return recognizer, label_dict

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Check if already marked today
    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        new_row = {"Name": name, "Date": today, "Time": now_time}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📸 Facial Recognition Attendance System")

# Upload new images
st.subheader("👤 Add New Person")
new_name = st.text_input("Enter Name")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file and new_name:
    save_path = os.path.join(IMAGE_DIR, new_name)
    os.makedirs(save_path, exist_ok=True)
    img = Image.open(uploaded_file)
    img.save(os.path.join(save_path, uploaded_file.name))
    st.success(f"Image saved for {new_name}. Retrain the model to update.")

# Train button
if st.button("🔄 Train Model"):
    recognizer, label_dict = train_model()
    if recognizer:
        st.session_state["recognizer"] = recognizer
        st.session_state["label_dict"] = label_dict
        st.success("✅ Model trained successfully!")
    else:
        st.error("❌ No images found. Please upload at least one image.")

# Camera
st.subheader("🎥 Live Camera")
start_cam = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

if start_cam:
    if "recognizer" not in st.session_state or st.session_state["recognizer"] is None:
        st.error("⚠️ Please train the model first.")
    else:
        recognizer = st.session_state["recognizer"]
        label_dict = st.session_state["label_dict"]

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]

                try:
                    label, confidence = recognizer.predict(roi_gray)
                except:
                    continue

                if confidence < 80:  # lower = more strict
                    name = label_dict[label]
                    mark_attendance(name)
                else:
                    name = "Unknown"

                # Draw box and name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

# Show attendance log
st.subheader("📋 Attendance Log")
if os.path.exists(ATTENDANCE_FILE):
    st.dataframe(pd.read_csv(ATTENDANCE_FILE))
else:
    st.info("No attendance recorded yet.")

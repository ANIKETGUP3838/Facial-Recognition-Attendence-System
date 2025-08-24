import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Train recognizer (same as before)
def prepare_training_data(data_dir="images"):
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces, labels, label_dict = [], [], {}
    current_label = 0
    data_dir = Path(data_dir)
    for person_dir in data_dir.iterdir():
        if not person_dir.is_dir(): continue
        person_name = person_dir.name
        label_dict[current_label] = person_name
        for img_file in person_dir.glob("*.*"):
            img = cv2.imread(str(img_file))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector.detectMultiScale(gray, 1.2, 5, minSize=(60,60))
            for (x,y,w,h) in rects:
                face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
                faces.append(face)
                labels.append(current_label)
        current_label += 1
    return faces, np.array(labels), label_dict

faces, labels, label_dict = prepare_training_data("images")
recognizer = None
if len(faces) > 0:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

attendance_file = "attendance.csv"
if os.path.exists(attendance_file):
    attendance = pd.read_csv(attendance_file)
else:
    attendance = pd.DataFrame(columns=["Name","Date","Time","Status"])

st.title("📸 Facial Recognition Attendance (Browser Camera)")

if recognizer is not None:
    img_file = st.camera_input("Take a photo")
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
                .detectMultiScale(gray, 1.2, 5, minSize=(60,60))
        for (x,y,w,h) in rects:
            face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            label, conf = recognizer.predict(face)
            if conf < 70:
                name = label_dict[label]
                today = datetime.now().strftime("%Y-%m-%d")
                now_time = datetime.now().strftime("%H:%M:%S")
                if not ((attendance["Name"]==name)&(attendance["Date"]==today)).any():
                    new_row = {"Name":name,"Date":today,"Time":now_time,"Status":"Present"}
                    attendance = pd.concat([attendance, pd.DataFrame([new_row])], ignore_index=True)
                    attendance.to_csv(attendance_file, index=False)
                color = (0,255,0)
            else:
                name = "Unknown"
                color = (0,0,255)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detection Result")
else:
    st.error("⚠️ No training data found. Please add faces in `images/`.")

st.subheader("📋 Attendance Log")
st.dataframe(attendance)

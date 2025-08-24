import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

st.set_page_config(page_title="Facial Recognition Attendance", page_icon="📸", layout="centered")

# -----------------------
# Paths & files
# -----------------------
IMAGES_DIR = Path("images")          # images/<person_name>/*.jpg
MODEL_PATH = Path("lbph_model.xml")  # saved recognizer
ATTEND_FILE = Path("attendance.csv")

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def list_people():
    """Return sorted list of person names based on subfolders inside images/."""
    if not IMAGES_DIR.exists():
        return []
    return sorted([p.name for p in IMAGES_DIR.iterdir() if p.is_dir()])

def load_haar():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError("OpenCV Haar cascade not found.")
    return cv2.CascadeClassifier(cascade_path)

def prepare_training_data(detector, img_size=(200, 200)):
    """Read images/<person>/*.jpg → faces (aligned to fixed size), labels, name map."""
    faces, labels = [], []
    name_to_id, id_to_name = {}, {}
    current_id = 0

    people = list_people()
    for person in people:
        person_dir = IMAGES_DIR / person
        for img_file in person_dir.glob("*"):
            if not img_file.is_file():
                continue
            # Read image
            data = np.frombuffer(img_file.read_bytes(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect face(s)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            if len(boxes) == 0:
                continue
            # choose largest face
            x, y, w, h = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, img_size)
            # map name → id
            if person not in name_to_id:
                name_to_id[person] = current_id
                id_to_name[current_id] = person
                current_id += 1
            faces.append(face)
            labels.append(name_to_id[person])

    return faces, labels, id_to_name

def save_attendance_row(name: str):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    if ATTEND_FILE.exists():
        df = pd.read_csv(ATTEND_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    # mark once per person per day
    already = ((df["Name"] == name) & (df["Date"] == today)).any()
    if not already:
        new_row = pd.DataFrame([{"Name": name, "Date": today, "Time": now_time}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTEND_FILE, index=False)
    return df

def draw_box_and_name(img_bgr, box, name):
    x, y, w, h = box
    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_bgr, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# -----------------------
# UI
# -----------------------
st.title("📸 Facial Recognition Attendance (OpenCV)")
st.caption("No dlib. No TensorFlow/Keras. Fast install, CPU-only.")

with st.sidebar:
    st.header("Configuration")
    conf_thresh = st.slider("LBPH confidence threshold (lower=better)", min_value=30, max_value=120, value=70, step=1,
                            help="Prediction is accepted if confidence ≤ threshold. Tune based on your data.")
    retrain = st.button("🧠 Train / Retrain model", use_container_width=True)
    st.markdown("---")
    st.markdown("**Training data folder**: `images/<person_name>/*.jpg`")
    if st.button("📁 Create sample folders"):
        (IMAGES_DIR / "person1").mkdir(parents=True, exist_ok=True)
        (IMAGES_DIR / "person2").mkdir(parents=True, exist_ok=True)
        st.success("Created `images/person1` and `images/person2`. Add face photos there.")

# -----------------------
# Train / Load model
# -----------------------
detector = load_haar()
recognizer = cv2.face.LBPHFaceRecognizer_create()

if retrain or not MODEL_PATH.exists():
    people = list_people()
    if len(people) == 0:
        st.warning("No training data found. Create folders under `images/` and add face photos, then click **Train**.")
    else:
        with st.spinner("Training model..."):
            faces, labels, id_to_name = prepare_training_data(detector)
            if len(faces) == 0:
                st.error("No faces detected in your training images. Add clearer, frontal photos and retrain.")
            else:
                recognizer.train(faces, np.array(labels))
                recognizer.write(str(MODEL_PATH))
                # Save label map alongside model
                np.save(MODEL_PATH.with_suffix(".labels.npy"), id_to_name)
                st.success(f"Model trained on {len(set(labels))} person(s), {len(labels)} face(s).")
else:
    # Load model + labels if available
    if MODEL_PATH.exists():
        recognizer.read(str(MODEL_PATH))
        labels_path = MODEL_PATH.with_suffix(".labels.npy")
        if labels_path.exists():
            # np.load returns array of objects (dict), allow_pickle=True is required
            id_to_name = np.load(labels_path, allow_pickle=True).item()
        else:
            id_to_name = {}
    else:
        id_to_name = {}

# -----------------------
# Camera Capture (browser) & Recognition
# -----------------------
st.subheader("🎥 Capture from your browser")
img_input = st.camera_input("Click **Take Photo** to capture a frame")

if img_input is not None and MODEL_PATH.exists():
    # Read uploaded bytes → OpenCV image
    file_bytes = np.asarray(bytearray(img_input.getvalue()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    display = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    recognized_names = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label_id, confidence = recognizer.predict(face)  # lower conf = better
        if confidence <= conf_thresh and label_id in id_to_name:
            name = id_to_name[label_id]
        else:
            name = "Unknown"
        draw_box_and_name(display, (x, y, w, h), f"{name} ({int(confidence)})")
        if name != "Unknown":
            recognized_names.append(name)

    st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), caption="Detection & Recognition", use_column_width=True)

    if len(recognized_names) > 0:
        # mark attendance per unique name
        for name in sorted(set(recognized_names)):
            df = save_attendance_row(name)
        st.success(f"Marked attendance for: {', '.join(sorted(set(recognized_names)))}")
    else:
        st.info("No known faces recognized. Try better lighting / frontal pose or lower the threshold.")

elif img_input is not None and not MODEL_PATH.exists():
    st.warning("Model not trained yet. Add images and click **Train / Retrain model** in the sidebar.")

# -----------------------
# Attendance table
# -----------------------
st.subheader("📋 Attendance Log")
if ATTEND_FILE.exists():
    df = pd.read_csv(ATTEND_FILE)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV", data=df.to_csv(index=False), file_name="attendance.csv", mime="text/csv")
else:
    st.write("No attendance yet.")

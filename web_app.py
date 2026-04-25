import csv
import json
import os
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, redirect, render_template, request, send_file, url_for

ATTENDANCE_FILE = "attendance.csv"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.json"
DATASET_DIR = "dataset"
FACE_SIZE = (200, 200)
SAMPLES_PER_PERSON = 15
UNKNOWN_THRESHOLD = 85.0

app = Flask(__name__)
video_lock = threading.Lock()
cap = None
camera_state = {
    "running": False,
    "auto_stop_after_mark": False,
    "last_event": "Camera is stopped",
}
session_marked_names = set()


def normalize_name(name):
    return " ".join(name.strip().split()).lower()


def ensure_files_and_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time", "Status"])


def get_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load Haar cascade face detector.")
    return detector


def get_recognizer():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("Install opencv-contrib-python so cv2.face is available.")
    return cv2.face.LBPHFaceRecognizer_create()


def load_labels():
    if not os.path.exists(LABELS_FILE):
        return {}
    with open(LABELS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_labels(labels):
    with open(LABELS_FILE, "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2)


def next_label_id(labels):
    if not labels:
        return 1
    return max(int(k) for k in labels.keys()) + 1


def mark_attendance(name):
    normalized_name = normalize_name(name)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if normalized_name in session_marked_names:
        return False

    with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2 and normalize_name(row[0]) == normalized_name and row[1] == date_str:
                session_marked_names.add(normalized_name)
                return False

    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([name, date_str, time_str, "Present"])
    session_marked_names.add(normalized_name)
    return True


def detect_faces(detector, gray_frame):
    # Reduced minNeighbors and minSize to makes detection easier
    return detector.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=3, 
        minSize=(30, 30)
    )


def preprocess_face(gray_frame, rect):
    x, y, w, h = rect
    face_roi = gray_frame[y : y + h, x : x + w]
    return cv2.resize(face_roi, FACE_SIZE)


def train_model(recognizer):
    labels = load_labels()
    if not labels:
        return labels, False

    train_images = []
    train_ids = []

    for label_id, person_name in labels.items():
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith(".png"):
                continue
            img_path = os.path.join(person_dir, file_name)
            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face_img is None:
                continue
            train_images.append(face_img)
            train_ids.append(int(label_id))

    if not train_images:
        return labels, False

    recognizer.train(train_images, np.array(train_ids))
    recognizer.save(MODEL_FILE)
    return labels, True


def ensure_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap


def stop_camera():
    global cap
    with video_lock:
        if cap is not None and cap.isOpened():
            cap.release()
        cap = None
    camera_state["running"] = False


def start_camera(auto_stop_after_mark=True):
    camera = ensure_camera()
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")
    camera_state["running"] = True
    camera_state["auto_stop_after_mark"] = auto_stop_after_mark
    camera_state["last_event"] = "Camera started"


def capture_samples(person_name):
    detector = get_detector()
    recognizer = get_recognizer()
    labels = load_labels()

    existing_label = None
    for label_id, existing_name in labels.items():
        if existing_name.lower() == person_name.lower():
            existing_label = int(label_id)
            person_name = existing_name
            break

    if existing_label is None:
        existing_label = next_label_id(labels)
        labels[str(existing_label)] = person_name
        save_labels(labels)

    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Keep only one sample image per person by clearing old captures first.
    for file_name in os.listdir(person_dir):
        if file_name.lower().endswith(".png"):
            try:
                os.remove(os.path.join(person_dir, file_name))
            except OSError:
                pass

    was_running = camera_state["running"]
    camera = ensure_camera()
    captured = 0

    while captured < SAMPLES_PER_PERSON:
        with video_lock:
            success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)
        if len(faces) == 0:
            continue

        largest = max(faces, key=lambda r: r[2] * r[3])
        face_img = preprocess_face(gray, largest)
        file_path = os.path.join(person_dir, f"{captured + 1:03d}.png")
        cv2.imwrite(file_path, face_img)
        captured += 1

    labels, trained = train_model(recognizer)
    
    # Mark attendance immediately after successful registration
    if trained:
        mark_attendance(person_name)
        camera_state["last_event"] = f"Registered {person_name} and marked attendance"
    
    # Registration should always end with camera stop as requested.
    if was_running or (camera is not None and camera.isOpened()):
        stop_camera()
        camera_state["last_event"] = f"Registered {person_name}. Camera stopped automatically"
    return trained, labels


def generate_frames():
    detector = get_detector()
    recognizer = get_recognizer()
    labels, has_model = train_model(recognizer)

    if not camera_state["running"]:
        return

    camera = ensure_camera()
    if not camera.isOpened():
        camera_state["last_event"] = "Could not open webcam"
        return

    while True:
        if not camera_state["running"]:
            break

        with video_lock:
            success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        for rect in faces:
            x, y, w, h = rect
            face_img = preprocess_face(gray, rect)
            name = "UNKNOWN"
            color = (0, 0, 255)
            confidence_text = ""

            if has_model:
                label_id, confidence = recognizer.predict(face_img)
                if confidence < UNKNOWN_THRESHOLD:
                    name = labels.get(str(label_id), "UNKNOWN")
                    color = (0, 255, 0)
                    confidence_text = f" {confidence:.1f}"
                    was_marked = mark_attendance(name)
                    if was_marked:
                        camera_state["last_event"] = f"Attendance marked for {name}"
                        if camera_state["auto_stop_after_mark"]:
                            stop_camera()
                            break

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y + h - 28), (x + w, y + h), color, cv2.FILLED)
            cv2.putText(
                frame,
                f"{name}{confidence_text}",
                (x + 6, y + h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            frame,
            "Camera will auto-stop after first attendance",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    labels = load_labels()
    attendance = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
            attendance = list(csv.reader(file))[1:]
    return render_template(
        "index.html",
        known_count=len(labels),
        attendance=attendance,
        camera_running=camera_state["running"],
        last_event=camera_state["last_event"],
    )


@app.route("/video_feed")
def video_feed():
    if not camera_state["running"]:
        return ("", 204)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/start", methods=["POST"])
def camera_start():
    auto_stop = request.form.get("auto_stop", "1") == "1"
    try:
        start_camera(auto_stop_after_mark=auto_stop)
    except RuntimeError as ex:
        camera_state["last_event"] = str(ex)
    return redirect(url_for("index"))


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    stop_camera()
    camera_state["last_event"] = "Camera stopped"
    return redirect(url_for("index"))


@app.route("/register", methods=["POST"])
def register():
    person_name = request.form.get("name", "").strip()
    if person_name:
        camera_state["last_event"] = f"Registering {person_name}"
        trained, labels = capture_samples(person_name)
        if trained:
            camera_state["last_event"] = f"Registered {person_name}"
            return redirect(url_for("index"))
        camera_state["last_event"] = "Registration failed. Try again."
    return redirect(url_for("index"))


@app.route("/download")
def download_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        ensure_files_and_dirs()
    return send_file(ATTENDANCE_FILE, as_attachment=True, download_name="attendance.csv")


@app.route("/refresh")
def refresh():
    return redirect(url_for("index"))


if __name__ == "__main__":
    ensure_files_and_dirs()
    debug_mode = os.environ.get("APP_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug_mode, use_reloader=debug_mode)

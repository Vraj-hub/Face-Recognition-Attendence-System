import csv
import json
import os
from datetime import datetime

import cv2
import numpy as np

ATTENDANCE_FILE = "attendance.csv"
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.json"
DATASET_DIR = "dataset"
FACE_SIZE = (200, 200)
SAMPLES_PER_PERSON = 20
UNKNOWN_THRESHOLD = 65.0


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
        raise RuntimeError(
            "OpenCV face module is missing. Install opencv-contrib-python instead of opencv-python."
        )
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
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == date_str:
                return

    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([name, date_str, time_str, "Present"])
    print(f"Attendance marked for {name}")


def detect_faces(detector, gray_frame):
    return detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))


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


def register_person(cap, detector, recognizer):
    person_name = input("Enter name to register: ").strip()
    if not person_name:
        print("Registration cancelled: name is empty.")
        return False

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

    print("Look at camera. Capturing face samples...")
    count = 0
    while count < SAMPLES_PER_PERSON:
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray)

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Live Face Attendance", frame)
            cv2.waitKey(1)
            continue

        largest = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest
        face_img = preprocess_face(gray, largest)

        file_path = os.path.join(person_dir, f"{count + 1:03d}.png")
        cv2.imwrite(file_path, face_img)
        count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Capturing {count}/{SAMPLES_PER_PERSON}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Live Face Attendance", frame)
        cv2.waitKey(100)

    labels, trained = train_model(recognizer)
    if trained:
        print(f"Registered and trained for {person_name}")
        return True

    print("Registration saved, but training failed. Try again.")
    return False


def main():
    ensure_files_and_dirs()

    try:
        detector = get_detector()
        recognizer = get_recognizer()
    except RuntimeError as ex:
        print(ex)
        return

    labels, has_model = train_model(recognizer)

    print("Live Face Recognition Attendance")
    print("Controls: [R] Register person, [Q] Quit")
    print(f"Known people loaded: {len(labels)}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read webcam frame.")
            break

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
                    mark_attendance(name)

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
            "Press R to register | Q to quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Live Face Attendance", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if register_person(cap, detector, recognizer):
                labels = load_labels()
                has_model = os.path.exists(MODEL_FILE)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
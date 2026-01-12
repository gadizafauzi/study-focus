import cv2
import mediapipe as mp
import time
import json
import pygame
from utils import eye_aspect_ratio

# Inisialisasi dari config
with open("config.json") as f:
    config = json.load(f)

EAR_THRESHOLD = config["ear_threshold"]
NON_FOCUS_DURATION = config["non_focus_duration"]
AUDIO_PATH = config["audio_path"]

# Inisialisasi pygame dan MediaPipe
pygame.mixer.init()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
start_non_focus = None
last_status = "focused"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔁 Membalik frame seperti cermin
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    h, w, _ = frame.shape
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [face_landmarks.landmark[i] for i in [263, 387, 385, 362, 380, 373]]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                if start_non_focus is None:
                    start_non_focus = time.time()
                elif time.time() - start_non_focus > NON_FOCUS_DURATION:
                    status = "distracted"
                    if last_status != status:
                        pygame.mixer.music.load(AUDIO_PATH)
                        pygame.mixer.music.play()
                    last_status = status
            else:
                start_non_focus = None
                status = "focused"
                last_status = status

            # Tampilkan status di layar
            cv2.putText(frame, f"Status: {status}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status == "focused" else (0, 0, 255), 2)

    cv2.imshow("Deteksi Fokus Belajar", frame)

    # 🚪 Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






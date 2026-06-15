import cv2
import numpy as np
import mediapipe as mp
import time
import json
import pygame
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
from utils import eye_aspect_ratio
import threading

app = Flask(__name__)
CORS(app)

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

# Global Variables
current_status = "focused"
last_status = "focused"
start_non_focus = None
lock = threading.Lock()

# Pomodoro Variables
POMODORO_WORK_TIME = 25 * 60
POMODORO_BREAK_TIME = 5 * 60
pomodoro_state = "IDLE"  # IDLE, WORK, BREAK
pomodoro_end_time = 0

# Global Analytics Variables
analytics = {
    "focus_time": 0.0,
    "unfocus_time": 0.0,
    "sleep_count": 0,
    "logs": []
}

def generate_frames():
    global current_status, last_status, start_non_focus, analytics, pomodoro_state
    
    cap = None
    last_time_check = time.time()
    
    while True:
        if pomodoro_state != "WORK":
            last_time_check = time.time()
            if cap is not None and cap.isOpened():
                cap.release()
                cap = None
            
            # Placeholder frame during break/idle
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = "Kamera Dimatikan (IDLE)" if pomodoro_state == "IDLE" else "Mode Istirahat (BREAK) - Rileks!"
            cv2.putText(frame, msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            continue
            
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)

        success, frame = cap.read()
        if not success:
            last_time_check = time.time()
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "Gagal mengakses kamera.", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(err_frame, "(Mungkin sedang dipakai Zoom/aplikasi lain?)", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            ret, buffer = cv2.imencode('.jpg', err_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            continue
        else:
            # Membalik frame seperti cermin
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            status = "focused"
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
                    else:
                        start_non_focus = None
                        status = "focused"

            with lock:
                current_time = time.time()
                elapsed = current_time - last_time_check
                last_time_check = current_time

                if current_status == "focused":
                    analytics["focus_time"] += elapsed
                else:
                    analytics["unfocus_time"] += elapsed

                current_status = status
                if last_status != status and status == "distracted":
                    analytics["sleep_count"] += 1
                    timestamp = time.strftime("%I:%M %p")
                    analytics["logs"].insert(0, {
                        "time": timestamp,
                        "trigger": "Mata tertutup / tidak fokus",
                        "action": "Alarm Suara Aktif"
                    })
                    if len(analytics["logs"]) > 5:
                        analytics["logs"].pop()

                    try:
                        pygame.mixer.music.load(AUDIO_PATH)
                        pygame.mixer.music.play()
                    except Exception as e:
                        print("Error memutar audio:", e)
                last_status = status

            # === GAMBAR STATUS DI ATAS VIDEO FRAME (PURE PYTHON) ===
            text = "SEDANG FOKUS" if current_status == "focused" else "TIDAK FOKUS!"
            color = (0, 255, 0) if current_status == "focused" else (0, 0, 255)
            # Buat background hitam semi-transparan untuk teks agar lebih mudah dibaca
            cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Encode frame untuk web (menjadi format jpeg)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame dalam format multipart HTTP
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    global pomodoro_state, pomodoro_end_time
    with lock:
        current_time = time.time()
        time_left = 0
        
        # Check and transition pomodoro state
        if pomodoro_state != "IDLE":
            time_left = int(pomodoro_end_time - current_time)
            if time_left <= 0:
                if pomodoro_state == "WORK":
                    pomodoro_state = "BREAK"
                    pomodoro_end_time = current_time + POMODORO_BREAK_TIME
                    time_left = POMODORO_BREAK_TIME
                elif pomodoro_state == "BREAK":
                    pomodoro_state = "IDLE"
                    pomodoro_end_time = 0
                    time_left = 0

        total = analytics["focus_time"] + analytics["unfocus_time"]
        score = int((analytics["focus_time"] / total * 100)) if total > 0 else 100
        duration_minutes = round(total / 60, 1)
        
        return jsonify({
            "duration": duration_minutes,
            "score": score,
            "sleep_count": analytics["sleep_count"],
            "logs": analytics["logs"],
            "pomodoro_state": pomodoro_state,
            "pomodoro_time_left": time_left
        })

@app.route('/api/pomodoro/start')
def start_pomodoro():
    global pomodoro_state, pomodoro_end_time, analytics
    pomodoro_state = "WORK"
    pomodoro_end_time = time.time() + POMODORO_WORK_TIME
    
    # Reset statistik saat memulai sesi baru
    analytics = {
        "focus_time": 0.0,
        "unfocus_time": 0.0,
        "sleep_count": 0,
        "logs": []
    }
    
    return jsonify({"status": "started", "state": pomodoro_state})

@app.route('/api/pomodoro/stop')
def stop_pomodoro():
    global pomodoro_state, pomodoro_end_time
    with lock:
        pomodoro_state = "IDLE"
        pomodoro_end_time = 0
    return jsonify({"status": "success", "state": pomodoro_state})

if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)

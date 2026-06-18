from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import numpy as np
import mediapipe as mp
from detector.yolo_detector import detect_fall_yolo
from detector.mediapipe_detector import detect_fall_mediapipe, posture_status
from detector.videomae_detector import detect_fall_videomae

import os

# ============================================================
# Phase 2: XGBoost ML detector (F1 94.3)
# ============================================================
try:
    from detector.phase2_detector import Phase2Detector
    phase2 = Phase2Detector()
    PHASE2_AVAILABLE = True
    print("Phase 2 XGBoost detector loaded (F1 94.3 model)")
except Exception as e:
    phase2 = None
    PHASE2_AVAILABLE = False
    print(f"[warning] Phase 2 model load failed: {e}. Using rule-based detection only.")

# Ensemble mode.
# - "and": rule-based AND XGBoost must agree. Lowest false positives.
# - "or": rule-based OR XGBoost can trigger a fall. Higher sensitivity.
# - "ml_only": XGBoost only.
# - "rule_only": rule-based only.
ENSEMBLE_MODE = os.environ.get("FALL_ENSEMBLE_MODE", "or")
XGBOOST_THRESHOLD = float(os.environ.get("FALL_XGBOOST_THR", "0.7"))

# Camera index. 0 is usually the built-in camera, 1 is usually an external USB camera.
CAMERA_INDEX = int(os.environ.get("FALL_CAMERA_INDEX", "0"))

# ============================================================
# Phase 1: activity data logging for behavior-based condition indicators.
# ============================================================
from detector import health_logger
health_logger.init_db()
# The logger thread starts after fall_status is defined.
import threading
import pygame
import time
import os
import json
from datetime import datetime

try:
    import serial
except Exception:
    serial = None

ARDUINO_PORT = os.environ.get("FALL_ARDUINO_PORT", "COM3")
ARDUINO_BAUD = int(os.environ.get("FALL_ARDUINO_BAUD", "9600"))
# "serial" (USB default) or "wifi" (Arduino sends HTTP POST to /arduino/status)
ARDUINO_MODE = os.environ.get("FALL_ARDUINO_MODE", "serial")

# ============================================================
# Spring Boot 연동 설정
# ============================================================
SENIOR_ID   = int(os.environ.get("FALL_SENIOR_ID",   "53"))    # 김숙희 ID
SPRING_URL  = os.environ.get("FALL_SPRING_URL",  "http://localhost:8082")
# 낙상 캡처 이미지 외부 접근 URL용 IP (Flutter 앱에서 이미지 접근 시 사용)
FALL_SERVER_IP = os.environ.get("FALL_SERVER_IP", "172.28.6.250")
import requests as _requests

_spring_alert_cooldown = 0   # 마지막 Spring 알림 전송 시각
SPRING_ALERT_COOLDOWN_SEC = 30  # 30초 쿨다운

def notify_spring_fall(metadata: dict):
    """낙상 감지 시 Spring Boot POST /api/alerts/fall 전송"""
    global _spring_alert_cooldown
    now = time.time()
    if now - _spring_alert_cooldown < SPRING_ALERT_COOLDOWN_SEC:
        return
    _spring_alert_cooldown = now
    try:
        capture_file = metadata.get("image", "")
        image_url = f"http://{FALL_SERVER_IP}:8000/captures/{capture_file}" if capture_file else None
        _requests.post(
            f"{SPRING_URL}/api/alerts/fall",
            json={
                "seniorId":          SENIOR_ID,
                "score":             int(metadata.get("score", 0)),
                "imageUrl":          image_url,
                "imageAccessUrl":    image_url,
                "notifyGuardian":    True,
                "notifyWelfare":     False,
                "escalationRequired": metadata.get("accident_by_stillness", False),
                "fallDetails": {
                    "posture":              metadata.get("posture", "unknown"),
                    "ensembleMode":         metadata.get("ensemble_mode", ""),
                    "accidentByStillness":  metadata.get("accident_by_stillness", False),
                    "xgbProba":             metadata.get("xgb_proba", 0),
                    "cameraFall":           metadata.get("camera_fall", False),
                    "arduinoFall":          metadata.get("arduino_fall", False),
                },
            },
            timeout=5,
        )
        print(f"[spring] fall alert sent → seniorId={SENIOR_ID}")
    except Exception as e:
        print(f"[spring] fall alert failed: {e}")

def sync_activity_to_spring():
    """활동 패턴 데이터를 Spring Boot에 주기적으로 동기화"""
    import json as _json
    endpoints = {
        "today":        lambda: health_logger.calc_today_scores(),
        "baseline":     lambda: health_logger.calc_personal_baseline(days=14),
        "fall-pattern": lambda: health_logger.analyze_fall_pattern(),
        "slots":        lambda: health_logger.get_today_slots(),
        "trend":        lambda: health_logger.get_trend(days=7),
    }
    for key, fn in endpoints.items():
        try:
            data = fn()
            _requests.put(
                f"{SPRING_URL}/api/seniors/{SENIOR_ID}/activity/{key}",
                data=_json.dumps(data, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            print(f"[spring] activity sync '{key}' failed: {e}")
    print(f"[spring] activity synced → seniorId={SENIOR_ID}")

def _activity_sync_loop():
    """10분마다 활동 데이터 Spring 동기화"""
    while True:
        time.sleep(600)
        sync_activity_to_spring()

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")
alert_sound.set_volume(1.0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

fall_status = {
    "fall_detected": False,
    "camera_fall": False,
    "arduino_fall": False,
    "arduino_status": "DISCONNECTED",
    "arduino_port": ARDUINO_PORT,
    "arduino_last_line": None,
    "score": 0,
    "last_capture": None,
    "battery": None,
}
alert_playing = False
fall_start_time = None
latest_frame_bytes = None
latest_frame_lock = threading.Lock()
camera_worker_started = False
camera_worker_lock = threading.Lock()
arduino_worker_started = False
arduino_worker_lock = threading.Lock()

# Store a fall_status snapshot every minute.
health_logger.start_logger_thread(fall_status)

# Separate VideoMAE worker state.
videomae_score_global = 0
videomae_frame_buffer = []
videomae_lock = threading.Lock()

os.makedirs("captures", exist_ok=True)

def videomae_worker():
    global videomae_score_global
    while True:
        with videomae_lock:
            frames = videomae_frame_buffer.copy()
        if len(frames) >= 16:
            score = detect_fall_videomae_from_buffer(frames)
            videomae_score_global = score
        time.sleep(0.5)

def detect_fall_videomae_from_buffer(frames):
    try:
        score = detect_fall_videomae(frames[-1])
        return score
    except:
        return 0

threading.Thread(target=videomae_worker, daemon=True).start()

def play_alert():
    global alert_playing
    if alert_playing:
        return
    alert_playing = True
    alert_sound.play()
    pygame.time.wait(3300)
    alert_playing = False


def start_arduino_worker():
    global arduino_worker_started

    with arduino_worker_lock:
        if arduino_worker_started:
            return
        arduino_worker_started = True

    def worker():
        if serial is None:
            fall_status["arduino_status"] = "SERIAL_NOT_INSTALLED"
            print("[arduino_worker] pyserial is not installed.")
            return

        while True:
            try:
                print(f"[arduino_worker] connecting to {ARDUINO_PORT} @ {ARDUINO_BAUD}")
                with serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1) as ser:
                    time.sleep(2)
                    fall_status["arduino_status"] = "CONNECTED"
                    fall_status["arduino_port"] = ARDUINO_PORT
                    print("[arduino_worker] connected")

                    while True:
                        raw = ser.readline()
                        if not raw:
                            continue

                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue

                        upper_line = line.upper()
                        fall_status["arduino_last_line"] = line

                        if "STATUS:" in upper_line:
                            status = upper_line.split("STATUS:", 1)[1].strip()
                            fall_status["arduino_status"] = status
                            fall_status["arduino_fall"] = status.startswith("FALL")
                        elif upper_line in ("FALL", "NORMAL", "STOPPED", "STARTED"):
                            fall_status["arduino_status"] = upper_line
                            fall_status["arduino_fall"] = upper_line == "FALL"

            except Exception as exc:
                fall_status["arduino_status"] = "DISCONNECTED"
                fall_status["arduino_fall"] = False
                print(f"[arduino_worker] error: {exc}. retrying in 3 seconds...")
                time.sleep(3)

    threading.Thread(target=worker, daemon=True).start()
    print("[arduino_worker] background serial loop started")

_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

def adjust_brightness(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def save_capture(frame, metadata=None):
    # Save one fall-event capture image, sidecar metadata, and DB event.
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    filename = f"captures/fall_{timestamp}.jpg"

    # Save a high-quality still image separately from the stream frame.
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Persist the event immediately so it is not missed between minute snapshots.
    health_logger.log_fall_event(metadata or {}, capture_filename=os.path.basename(filename))

    # Spring Boot에 낙상 알림 전송 (백그라운드)
    threading.Thread(target=notify_spring_fall, args=(metadata or {},), daemon=True).start()

    # Save sidecar metadata next to the image.
    if metadata:
        meta_filename = filename.replace(".jpg", ".json")
        metadata["captured_at"] = now.isoformat()
        metadata["image"] = os.path.basename(filename)
        with open(meta_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    fall_status["last_capture"] = filename
    print(f"[capture] saved {filename}")
    return filename

def generate_frames():
    global fall_start_time, videomae_frame_buffer, latest_frame_bytes
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    print(f"[camera] using index {CAMERA_INDEX} (FALL_CAMERA_INDEX={CAMERA_INDEX})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    last_captured = False
    frame_count = 0

    # Track one capture per fall event.
    last_capture_time = 0
    fall_off_time = None
    CAPTURE_COOLDOWN = 5
    event_max_score = 0
    event_best_frame = None
    event_best_meta = None
    fall_ongoing = False

    # Cache expensive inference results.
    cached_yolo_score = 0
    cached_brightness_frame = None

    # Processing intervals.
    YOLO_INTERVAL = 3
    BRIGHTNESS_INTERVAL = 5
    VIDEOMAE_INTERVAL = 2
    MEDIAPIPE_INTERVAL = 2
    PHASE2_INTERVAL = 3
    JPEG_QUALITY = 70
    cached_xgb_proba = 0.0
    cached_mediapipe_score = 0
    cached_pose_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        # Brightness correction every few frames.
        if frame_count % BRIGHTNESS_INTERVAL == 0 or cached_brightness_frame is None:
            frame = adjust_brightness(frame)
            cached_brightness_frame = frame

        # VideoMAE frame buffer.
        if frame_count % VIDEOMAE_INTERVAL == 0:
            resized = cv2.resize(frame, (224, 224))
            rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            with videomae_lock:
                videomae_frame_buffer.append(rgb_small)
                if len(videomae_frame_buffer) > 16:
                    videomae_frame_buffer.pop(0)

        # MediaPipe Pose inference.
        if frame_count % MEDIAPIPE_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cached_pose_results = pose.process(rgb)
            cached_mediapipe_score = 0
            if cached_pose_results.pose_landmarks:
                cached_mediapipe_score = detect_fall_mediapipe(
                    cached_pose_results.pose_landmarks.landmark,
                    frame.shape[0]
                )
        results = cached_pose_results
        mediapipe_score = cached_mediapipe_score
        # Draw landmarks on every displayed frame for smoother UI.
        if results is not None and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # YOLO inference, cached between frames.
        if frame_count % YOLO_INTERVAL == 0:
            cached_yolo_score = detect_fall_yolo(frame)
        yolo_score = cached_yolo_score

        total_score = mediapipe_score + yolo_score + videomae_score_global

        # Posture-based thresholds reduce false positives while sitting or standing still.
        posture = posture_status.get("posture", "unknown")
        stillness_sec = posture_status.get("stillness_sec", 0)
        abnormal = posture_status.get("abnormal", False)

        # Use posture-specific fall thresholds from the MediaPipe detector.
        from detector.mediapipe_detector import get_threshold_for_posture
        fall_threshold = get_threshold_for_posture(posture)

        # Accident-by-stillness rule:
        # - lying still for 10 seconds can indicate an accident.
        # - sitting/standing still requires abnormal posture too.
        accident_by_stillness = (
            (posture == "lying" and stillness_sec >= 10.0) or
            (posture in ("sitting", "standing") and abnormal and stillness_sec >= 5.0)
        )

        # Rule-based decision.
        rule_based_pos = (total_score >= fall_threshold) or accident_by_stillness

        # ============================================================
        # Phase 2: XGBoost ML decision.
        # ============================================================
        if PHASE2_AVAILABLE and frame_count % PHASE2_INTERVAL == 0:
            try:
                cached_xgb_proba = phase2.predict(frame, results, frame.shape[0])
            except Exception:
                pass
        xgb_proba = cached_xgb_proba
        xgb_pos = xgb_proba >= XGBOOST_THRESHOLD

        # Ensemble decision.
        if ENSEMBLE_MODE == "and":
            ensemble_pos = rule_based_pos and xgb_pos
        elif ENSEMBLE_MODE == "or":
            ensemble_pos = rule_based_pos or xgb_pos
        elif ENSEMBLE_MODE == "ml_only":
            ensemble_pos = xgb_pos
        else:  # rule_only
            ensemble_pos = rule_based_pos

        # Time filter: require the decision to hold briefly.
        if ensemble_pos:
            if fall_start_time is None:
                fall_start_time = time.time()
            elapsed = time.time() - fall_start_time
            fall_detected = elapsed >= 0.5
        else:
            fall_start_time = None
            fall_detected = False

        camera_fall = fall_detected
        arduino_fall = bool(fall_status.get("arduino_fall", False))
        fall_detected = camera_fall or arduino_fall

        fall_status["fall_detected"] = fall_detected
        fall_status["camera_fall"] = camera_fall
        fall_status["score"] = total_score
        fall_status["xgb_proba"] = round(xgb_proba, 3)
        fall_status["ensemble_mode"] = ENSEMBLE_MODE
        fall_status["rule_pos"] = rule_based_pos
        fall_status["xgb_pos"] = xgb_pos
        fall_status["posture"] = posture
        fall_status["stillness_sec"] = stillness_sec
        fall_status["threshold"] = fall_threshold
        fall_status["accident_by_stillness"] = accident_by_stillness
        fall_status["abnormal_posture"] = abnormal
        # Separate camera recognition quality from actual posture ambiguity.
        fall_status["landmark_detected"] = (results is not None and results.pose_landmarks is not None)

        # Fall alert and one-capture-per-event handling.
        now_ts = time.time()

        if fall_detected:
            # Play alert immediately.
            if not alert_playing:
                threading.Thread(target=play_alert, daemon=True).start()

            # Track the best frame while the fall event is active.
            if not fall_ongoing:
                fall_ongoing = True
                event_max_score = total_score
                event_best_frame = frame.copy()
                event_best_meta = {
                    "score": total_score,
                    "threshold": fall_threshold,
                    "posture": posture,
                    "stillness_sec": stillness_sec,
                    "accident_by_stillness": accident_by_stillness,
                    "mediapipe_score": mediapipe_score,
                    "yolo_score": yolo_score,
                    "videomae_score": videomae_score_global,
                    "xgb_proba": round(xgb_proba, 3),
                    "ensemble_mode": ENSEMBLE_MODE,
                    "rule_pos": rule_based_pos,
                    "xgb_pos": xgb_pos,
                    "camera_fall": camera_fall,
                    "arduino_fall": arduino_fall,
                    "arduino_status": fall_status.get("arduino_status"),
                    "arduino_last_line": fall_status.get("arduino_last_line"),
                }
            elif total_score > event_max_score:
                # Replace the event frame if a clearer/higher-score frame appears.
                event_max_score = total_score
                event_best_frame = frame.copy()
                event_best_meta = {
                    "score": total_score,
                    "threshold": fall_threshold,
                    "posture": posture,
                    "stillness_sec": stillness_sec,
                    "accident_by_stillness": accident_by_stillness,
                    "mediapipe_score": mediapipe_score,
                    "yolo_score": yolo_score,
                    "videomae_score": videomae_score_global,
                    "xgb_proba": round(xgb_proba, 3),
                    "ensemble_mode": ENSEMBLE_MODE,
                    "rule_pos": rule_based_pos,
                    "xgb_pos": xgb_pos,
                    "camera_fall": camera_fall,
                    "arduino_fall": arduino_fall,
                    "arduino_status": fall_status.get("arduino_status"),
                    "arduino_last_line": fall_status.get("arduino_last_line"),
                }

            # Capture once per event after cooldown.
            if not last_captured and (now_ts - last_capture_time) > CAPTURE_COOLDOWN:
                threading.Thread(
                    target=save_capture,
                    args=(event_best_frame.copy(), dict(event_best_meta)),
                    daemon=True
                ).start()
                last_captured = True
                last_capture_time = now_ts
        else:
            # End current fall event.
            if fall_ongoing:
                fall_ongoing = False
                fall_off_time = now_ts
                event_max_score = 0
                event_best_frame = None
                event_best_meta = None

            # Allow the next event after normal state is stable or cooldown expires.
            if last_captured:
                event_clearly_ended = (
                    fall_off_time is not None and (now_ts - fall_off_time) > 2.0
                )
                if event_clearly_ended or (now_ts - last_capture_time) > CAPTURE_COOLDOWN:
                    last_captured = False
                    fall_off_time = None

        # Overlay status label.
        if fall_detected:
            if accident_by_stillness and total_score < fall_threshold:
                label = f"ACCIDENT! Still {stillness_sec}s ({total_score}pt)"
            else:
                label = f"FALL DETECTED! ({total_score}/{fall_threshold}pt)"
        else:
            label = f"Normal ({total_score}/{fall_threshold}pt)"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        abn_txt = "ABNORMAL" if abnormal else "OK"
        too_close = posture_status.get("too_close", False)
        close_txt = " [TOO CLOSE]" if too_close else ""
        cv2.putText(frame, f"Posture: {posture}({abn_txt}){close_txt} | Still: {stillness_sec}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"MediaPipe: {mediapipe_score}pt", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"YOLO: {yolo_score}pt", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"VideoMAE: {videomae_score_global}pt", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # Display Phase 2 XGBoost probability.
        if PHASE2_AVAILABLE:
            xgb_color = (0, 0, 255) if xgb_pos else (0, 255, 0)
            cv2.putText(frame, f"XGBoost: {xgb_proba:.2f} [{ENSEMBLE_MODE}]", (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, xgb_color, 2)

        # JPEG encode and publish the latest frame for stream viewers.
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        with latest_frame_lock:
            latest_frame_bytes = frame_bytes
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def stream_latest_frames():
    #     Stream the latest frame produced by the background camera worker.
    while True:
        with latest_frame_lock:
            frame_bytes = latest_frame_bytes

        if frame_bytes is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Camera worker starting...",
                (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)


def start_camera_worker():
    #     Start camera/model loop once, independent from /video viewers.
    global camera_worker_started

    with camera_worker_lock:
        if camera_worker_started:
            return
        camera_worker_started = True

    def worker():
        while True:
            try:
                for _ in generate_frames():
                    pass
            except Exception as exc:
                print(f"[camera_worker] error: {exc}")
            print("[camera_worker] stopped. retrying in 3 seconds...")
            time.sleep(3)

    threading.Thread(target=worker, daemon=True).start()
    print("[camera_worker] background camera/model loop started")


@app.on_event("startup")
def on_startup():
    start_camera_worker()
    if ARDUINO_MODE == "serial":
        start_arduino_worker()
    else:
        fall_status["arduino_status"] = "WIFI_MODE"
        print(f"[arduino] WiFi mode — waiting for POST to /arduino/status")
    # 활동 데이터 Spring 동기화 루프 시작
    threading.Thread(target=_activity_sync_loop, daemon=True).start()
    print(f"[spring] seniorId={SENIOR_ID}, springUrl={SPRING_URL}")


@app.get("/status")
def get_status():
    return fall_status

@app.get("/video")
def video_feed():
    return StreamingResponse(
        stream_latest_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@app.get("/captures")
def get_captures():
    #     Return captured fall image list.
    if not os.path.exists("captures"):
        return {"captures": []}

    files = [f for f in os.listdir("captures") if f.endswith(".jpg")]
    files.sort(reverse=True)

    result = []
    for fname in files[:10]:
        item = {"image": fname}
        # Include sidecar metadata when available.
        meta_path = os.path.join("captures", fname.replace(".jpg", ".json"))
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    item["metadata"] = json.load(f)
            except Exception:
                pass
        result.append(item)

    return {"captures": result}


@app.get("/captures/{filename}")
def get_capture_file(filename: str):
    #     Return a captured fall image file.
    # Prevent path traversal.
    safe_name = os.path.basename(filename)
    file_path = os.path.join("captures", safe_name)
    if not os.path.exists(file_path):
        return {"error": "not found"}
    return FileResponse(file_path)


@app.post("/arduino/status")
def arduino_status(payload: dict):
    """Arduino WiFi mode: POST {"status": "FALL", "battery": 85} or {"status": "NORMAL"}"""
    raw = str(payload.get("status", "")).upper().strip()
    fall_status["arduino_last_line"] = raw
    if "STATUS:" in raw:
        status = raw.split("STATUS:", 1)[1].strip()
    else:
        status = raw
    fall_status["arduino_status"] = status
    fall_status["arduino_fall"] = status.startswith("FALL")

    battery = payload.get("battery")
    if battery is not None:
        try:
            fall_status["battery"] = int(round(float(battery)))
        except (ValueError, TypeError):
            pass

    return {"ok": True, "received": status}


@app.get("/")
def root():
    return {"status": "Fall Detection Server Running"}


# ============================================================
# Phase 1: Activity condition APIs (behavior-based, not medical diagnosis)
# ============================================================
@app.get("/health/activity/today")
def health_activity_today():
    #     Return today activity condition scores.
    return health_logger.calc_today_scores()


@app.get("/health/activity/summary")
def health_activity_summary():
    #     Return today activity summary.
    return health_logger.get_today_summary()


@app.get("/health/activity/trend")
def health_activity_trend(days: int = 7):
    #     Compare today with recent activity data.
    return health_logger.get_trend(days=days)


@app.get("/health/activity/falls")
def health_activity_falls(days: int = 7):
    #     Return recent fall events.
    return health_logger.get_fall_events(days=days)


@app.get("/health/activity/slots")
def health_activity_slots():
    #     Return today activity scores by time slot.
    return health_logger.get_today_slots()


@app.get("/health/activity/baseline")
def health_activity_baseline(days: int = 14):
    #     Return personal activity baseline.
    return health_logger.calc_personal_baseline(days=days)


@app.get("/health/activity/fall-pattern")
def health_activity_fall_pattern():
    #     Return pre/post fall activity pattern.
    return health_logger.analyze_fall_pattern()


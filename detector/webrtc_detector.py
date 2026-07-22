"""CPU-conscious fall detector for browser WebRTC frames."""

import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from detector.mediapipe_detector import detect_fall_mediapipe, get_threshold_for_posture, posture_status
from detector.phase2_detector import Phase2Detector
from detector.spring_alert import SpringFallNotifier


def _flag(name, default=False):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


class WebRTCFallDetector:
    def __init__(self):
        self.process_every = max(1, int(os.getenv("FALL_PROCESS_EVERY_N_FRAMES", "3")))
        self.input_size = max(256, int(os.getenv("FALL_INPUT_SIZE", "480")))
        self.xgb_threshold = float(os.getenv("FALL_XGBOOST_THR", "0.70"))
        self.confirm_frames = max(1, int(os.getenv("FALL_CONFIRM_FRAMES", "3")))
        self.enable_videomae = _flag("FALL_ENABLE_VIDEOMAE")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0, smooth_landmarks=True,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.phase2 = Phase2Detector(enable_videomae=self.enable_videomae)
        self.notifier = SpringFallNotifier()
        self.lock = threading.Lock()
        self.frame_no = 0
        self.positive_frames = 0
        self.last_result = {"fall": False, "score": 0, "xgb": 0.0, "posture": "unknown"}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        with self.lock:
            self.frame_no += 1
            if self.frame_no % self.process_every == 0:
                self._infer(image)
            return self._overlay(image.copy())

    def _infer(self, rgb):
        height, width = rgb.shape[:2]
        scale = min(1.0, self.input_size / max(height, width))
        work_rgb = cv2.resize(rgb, (int(width * scale), int(height * scale))) if scale < 1.0 else rgb
        work_bgr = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2BGR)
        pose_result = self.pose.process(work_rgb)
        rule_score = 0
        if pose_result.pose_landmarks:
            rule_score = detect_fall_mediapipe(
                pose_result.pose_landmarks.landmark, work_rgb.shape[0], time.time()
            )
        xgb = self.phase2.predict(work_bgr, pose_result, work_rgb.shape[0])
        posture = posture_status.get("posture", "unknown")
        rule_fall = rule_score >= get_threshold_for_posture(posture)
        fall = rule_fall or xgb >= self.xgb_threshold
        self.positive_frames = self.positive_frames + 1 if fall else 0
        confirmed = self.positive_frames >= self.confirm_frames
        self.last_result = {"fall": confirmed, "score": rule_score, "xgb": xgb, "posture": posture}
        if confirmed:
            self.notifier.notify_async({
                "source": "hf-webrtc", "posture": posture, "score": rule_score,
                "xgbProba": round(xgb, 4), "cameraFall": True,
                "persistent": self.positive_frames >= self.confirm_frames * 2,
                "videoMAEEnabled": self.enable_videomae,
            })

    def _overlay(self, image):
        result = self.last_result
        color = (255, 45, 45) if result["fall"] else (40, 210, 95)
        label = "FALL DETECTED" if result["fall"] else "MONITORING"
        cv2.rectangle(image, (0, 0), (image.shape[1], 72), (12, 18, 28), -1)
        cv2.putText(image, label, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        detail = f"posture={result['posture']}  rule={result['score']}  xgb={result['xgb']:.2f}"
        cv2.putText(image, detail, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        return image

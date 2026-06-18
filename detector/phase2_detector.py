"""
Phase 2 실시간 추론 detector
- XGBoost 모델 (xgboost_model.json) 로드
- 8 프레임 시퀀스 버퍼 유지
- 53 features × 3 통계 = 159 features 윈도우 특징 → fall 확률 반환
- extract_features.py와 정확히 동일한 features 순서 보장 (모델 호환)
"""
import numpy as np
import xgboost as xgb
import mediapipe as mp
import torch
from pathlib import Path
from detector.yolo_detector import model as yolo_model
from detector.videomae_detector import processor as mae_processor, model as mae_model

MODEL_PATH = Path(r"C:\fall-detection\phase2_features\xgboost_model.json")
WINDOW = 8

# extract_features.py와 동일한 keypoint 순서 (절대 바꾸면 안 됨!)
KEY_LANDMARKS = [
    mp.solutions.pose.PoseLandmark.NOSE,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
]

POSTURE_TO_IDX = {"lying": 0, "sitting": 1, "standing": 2, "unknown": 3}


class Phase2Detector:
    """Phase 2 XGBoost 실시간 추론 detector"""

    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"XGBoost 모델 없음: {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        self.buffer = []                # 최근 WINDOW 프레임 특징 (53-dim)
        self.prev_nose_y = None
        self.prev_vel = 0.0
        self.mae_buffer = []            # VideoMAE 16 프레임 버퍼
        self.last_mae_probs = [0.0, 0.0, 0.0, 0.0]
        self.frame_count = 0

    def _yolo_max_fall_conf(self, frame):
        """YOLO fall 클래스 raw 최고 신뢰도"""
        results = yolo_model(frame, verbose=False)
        max_conf = 0.0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls]
                if "fall" in label.lower() and conf > max_conf:
                    max_conf = conf
        return max_conf

    def _videomae_probs(self):
        """VideoMAE 라벨별 확률 (FallDown, LyingDown, Sitting, Walking)"""
        if len(self.mae_buffer) < 16:
            return self.last_mae_probs
        inputs = mae_processor(list(self.mae_buffer), return_tensors="pt")
        with torch.no_grad():
            logits = mae_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        id2label = mae_model.config.id2label
        out = [0.0, 0.0, 0.0, 0.0]
        target = {"FallDown": 0, "LyingDown": 1, "Sitting": 2, "Walking": 3}
        for i in range(len(id2label)):
            lbl = id2label[i]
            if lbl in target:
                out[target[lbl]] = float(probs[i])
        self.last_mae_probs = out
        return out

    def _classify_posture_simple(self, lms):
        """extract_features.py와 동일한 자세 분류"""
        import mediapipe as mp
        l_sh = lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = lms[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        r_hip = lms[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        l_knee = lms[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        r_knee = lms[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]

        hip_vis = (l_hip.visibility + r_hip.visibility) / 2
        knee_vis = (l_knee.visibility + r_knee.visibility) / 2
        if hip_vis < 0.5:
            return "unknown", 0.0, 0.0, 0.0

        sh_y = (l_sh.y + r_sh.y) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        knee_y = (l_knee.y + r_knee.y) / 2
        sh_x = (l_sh.x + r_sh.x) / 2
        hip_x = (l_hip.x + r_hip.x) / 2

        dx, dy = abs(sh_x - hip_x), abs(sh_y - hip_y)
        horizontal_ratio = dx / (dx + dy) if (dx + dy) > 0 else 0
        sh_diff = hip_y - sh_y
        hk_diff = knee_y - hip_y
        abs_sh = abs(sh_diff)

        if horizontal_ratio > 0.55 and abs_sh < 0.12:
            return "lying", horizontal_ratio, sh_diff, hk_diff
        if abs_sh < 0.06 or horizontal_ratio > 0.70:
            return "lying", horizontal_ratio, sh_diff, hk_diff
        if knee_vis < 0.5:
            return ("standing" if sh_diff > 0.18 else "unknown"), horizontal_ratio, sh_diff, hk_diff
        if knee_y < hip_y - 0.03:
            return "lying", horizontal_ratio, sh_diff, hk_diff
        if sh_diff > 0.18 and hk_diff > 0.15:
            return "standing", horizontal_ratio, sh_diff, hk_diff
        if sh_diff > 0.10 and 0.02 < hk_diff < 0.12:
            return "sitting", horizontal_ratio, sh_diff, hk_diff
        return "unknown", horizontal_ratio, sh_diff, hk_diff

    def extract_frame_features(self, frame, pose_results, frame_height):
        """프레임 1개에서 53 features 추출 (extract_features.py와 정확히 동일)"""
        import cv2
        frame_feat = []

        if pose_results.pose_landmarks:
            lms = pose_results.pose_landmarks.landmark
            # 13개 keypoints × (x, y, visibility) = 39
            for lm_id in KEY_LANDMARKS:
                lm = lms[lm_id]
                frame_feat.extend([lm.x, lm.y, lm.visibility])
            posture, horiz, sh_diff, hk_diff = self._classify_posture_simple(lms)
            nose_y = lms[mp.solutions.pose.PoseLandmark.NOSE].y * frame_height
        else:
            frame_feat.extend([0.0] * 39)
            posture, horiz, sh_diff, hk_diff = "unknown", 0.0, 0.0, 0.0
            nose_y = 0.0

        # velocity, accel
        velocity = (nose_y - self.prev_nose_y) if self.prev_nose_y is not None else 0
        accel = velocity - self.prev_vel
        self.prev_vel = velocity
        self.prev_nose_y = nose_y

        # 파생 5개
        frame_feat.extend([horiz, sh_diff, hk_diff, velocity / 100.0, accel / 100.0])

        # posture one-hot 4개
        pos_oh = [0, 0, 0, 0]
        pos_oh[POSTURE_TO_IDX[posture]] = 1
        frame_feat.extend(pos_oh)

        # YOLO 1개
        frame_feat.append(self._yolo_max_fall_conf(frame))

        # VideoMAE 4개 - 16 프레임 버퍼링, 매 8 프레임마다 새로 계산
        resized = cv2.resize(frame, (224, 224))
        rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.mae_buffer.append(rgb_small)
        if len(self.mae_buffer) > 16:
            self.mae_buffer.pop(0)
        self.frame_count += 1
        if len(self.mae_buffer) == 16 and self.frame_count % 8 == 0:
            mae_probs = self._videomae_probs()
        else:
            mae_probs = self.last_mae_probs
        frame_feat.extend(mae_probs)

        return frame_feat

    def predict(self, frame, pose_results, frame_height):
        """프레임 1개 입력 → fall 확률 반환 (0~1)"""
        feat = self.extract_frame_features(frame, pose_results, frame_height)
        self.buffer.append(feat)
        if len(self.buffer) > WINDOW:
            self.buffer.pop(0)

        if len(self.buffer) < WINDOW:
            return 0.0

        arr = np.array(self.buffer)
        # train_phase2_xgboost.py와 동일한 윈도우 특징
        # 현재 + mean + max = 53 * 3 = 159
        cur = arr[-1]
        mean = arr.mean(axis=0)
        mx = arr.max(axis=0)
        feat_window = np.concatenate([cur, mean, mx]).reshape(1, -1).astype(np.float32)

        proba = float(self.model.predict_proba(feat_window)[0, 1])
        return proba

    def reset(self):
        """카메라/세션 재시작 시 호출"""
        self.buffer = []
        self.prev_nose_y = None
        self.prev_vel = 0.0
        self.mae_buffer = []
        self.last_mae_probs = [0.0, 0.0, 0.0, 0.0]
        self.frame_count = 0

"""
Phase 2 - Step 1: 시계열 특징 추출
각 영상에서 frame-by-frame 특징 시퀀스 추출 → npz로 저장
이후 XGBoost/1D-CNN/LSTM 모두 같은 데이터 사용 가능 (재추출 X)

특징 (프레임당):
  - mediapipe pose 13개 핵심 keypoints × (x, y, visibility) = 39
  - 파생 특징: horizontal_ratio, sh_diff, hk_diff, velocity, accel = 5
  - posture one-hot (lying/sitting/standing/unknown) = 4
  - YOLO fall conf (raw) = 1
  - VideoMAE: FallDown_prob, LyingDown_prob, Sitting_prob, Walking_prob = 4
  총 53 features/frame

라벨:
  - Le2i: annotation 시작~끝 프레임 = fall(1), 나머지 = 0
  - uttej: Fall 폴더 = 모든 프레임 fall(1), No_Fall = 0
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import torch
from pathlib import Path
from detector.yolo_detector import model as yolo_model
from detector.videomae_detector import processor, model as videomae_model

LE2I_DIR = r"data\falld_2"
UTTEJ_DIR = r"data\falld_3"
OUTPUT_DIR = Path(r"C:\fall-detection\phase2_features")
OUTPUT_DIR.mkdir(exist_ok=True)

# 작은 데이터셋으로 시작 (검증용). 효과 확인 후 늘림
LE2I_PER_SCENARIO = 10
UTTEJ_PER_CLASS = 100
FRAME_SKIP = 3       # 1/3 프레임만 처리 (시간 절약)

# mediapipe 핵심 keypoints (13개 - 너무 많으면 노이즈)
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


def yolo_max_fall_conf(frame):
    """YOLO가 출력한 fall 클래스 중 최고 신뢰도 (raw, 임계값 우회)"""
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


def videomae_label_probs(buffer):
    """VideoMAE 라벨별 확률 (FallDown, LyingDown, Sitting, Walking)"""
    if len(buffer) < 16:
        return [0.0, 0.0, 0.0, 0.0]
    inputs = processor(list(buffer), return_tensors="pt")
    with torch.no_grad():
        logits = videomae_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    id2label = videomae_model.config.id2label
    out = [0.0, 0.0, 0.0, 0.0]
    target = {"FallDown": 0, "LyingDown": 1, "Sitting": 2, "Walking": 3}
    for i in range(len(id2label)):
        lbl = id2label[i]
        if lbl in target:
            out[target[lbl]] = float(probs[i])
    return out


def classify_posture_simple(lms, frame_h):
    """간략 자세 분류 (mediapipe_detector의 classify_posture 로직 압축)"""
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


def extract_video_features(video_path, pose, fall_frames=None):
    """
    영상 1개 처리 → frame별 features 시퀀스 + 라벨 시퀀스
    fall_frames: (start, end) 튜플. None이면 전체 fall=0, 둘 다 0이면 전체 fall=0
                 폴더 기반 영상은 fall_frames=("all", None)로 전달
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    feats = []   # [T, 53]
    labels = [] # [T]
    videomae_buffer = []
    prev_y = None
    prev_v = 0.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        frame_feat = []

        # mediapipe pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            # 13개 keypoints × (x, y, visibility) = 39
            for lm_id in KEY_LANDMARKS:
                lm = lms[lm_id]
                frame_feat.extend([lm.x, lm.y, lm.visibility])

            posture, horiz, sh_diff, hk_diff = classify_posture_simple(lms, frame.shape[0])
            nose_y = lms[mp.solutions.pose.PoseLandmark.NOSE].y * frame.shape[0]
        else:
            frame_feat.extend([0.0] * 39)
            posture, horiz, sh_diff, hk_diff = "unknown", 0.0, 0.0, 0.0
            nose_y = 0.0

        # velocity, accel
        velocity = (nose_y - prev_y) if prev_y is not None else 0
        accel = velocity - prev_v
        prev_v = velocity
        prev_y = nose_y

        # 파생 특징 5개
        frame_feat.extend([horiz, sh_diff, hk_diff, velocity / 100.0, accel / 100.0])

        # posture one-hot 4개
        pos_oh = [0, 0, 0, 0]
        pos_oh[POSTURE_TO_IDX[posture]] = 1
        frame_feat.extend(pos_oh)

        # YOLO fall conf
        frame_feat.append(yolo_max_fall_conf(frame))

        # VideoMAE label probs (16프레임 버퍼)
        resized = cv2.resize(frame, (224, 224))
        rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        videomae_buffer.append(rgb_small)
        if len(videomae_buffer) > 16:
            videomae_buffer.pop(0)
        # 매 8 처리프레임마다 호출 (속도)
        if len(videomae_buffer) == 16 and (len(feats) % 8 == 0):
            mae_probs = videomae_label_probs(videomae_buffer)
        else:
            mae_probs = feats[-1][-4:] if feats else [0.0, 0.0, 0.0, 0.0]
        frame_feat.extend(mae_probs)

        feats.append(frame_feat)

        # 라벨
        if fall_frames is None:
            labels.append(0)
        elif fall_frames == ("all", None):
            labels.append(1)
        else:
            start, end = fall_frames
            if start == 0 and end == 0:
                labels.append(0)
            else:
                labels.append(1 if (start - 10) <= frame_idx <= (end + 30) else 0)

    cap.release()
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int8)


def collect_le2i(pose):
    """Le2i 영상-annotation 페어 + 라벨 추출"""
    print("\n[Le2i 처리 중]", flush=True)
    all_feats = []
    all_labels = []
    all_meta = []

    for room in sorted(os.listdir(LE2I_DIR)):
        room_path = os.path.join(LE2I_DIR, room)
        if not os.path.isdir(room_path):
            continue

        # Videos + Annotation_files 찾기
        videos_dir, anno_dir = None, None
        for cur, dirs, files in os.walk(room_path):
            if "Videos" in dirs and "Annotation_files" in dirs:
                videos_dir = os.path.join(cur, "Videos")
                anno_dir = os.path.join(cur, "Annotation_files")
                break
        if not videos_dir:
            continue

        videos = sorted([f for f in os.listdir(videos_dir) if f.lower().endswith(".avi")])[:LE2I_PER_SCENARIO]
        for vf in videos:
            vpath = os.path.join(videos_dir, vf)
            apath = os.path.join(anno_dir, os.path.splitext(vf)[0] + ".txt")
            try:
                with open(apath, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                start = int(lines[0]); end = int(lines[1])
            except Exception:
                start, end = 0, 0

            feats, labels = extract_video_features(vpath, pose, fall_frames=(start, end))
            if feats is not None and len(feats) > 0:
                all_feats.append(feats)
                all_labels.append(labels)
                all_meta.append({"source": "le2i", "scenario": room, "video": vf, "fall_range": (start, end)})
                print(f"  Le2i/{room}/{vf}: {len(feats)} frames, fall={labels.sum()}", flush=True)

    return all_feats, all_labels, all_meta


def collect_uttej(pose):
    """uttej Fall/No_Fall 영상 추출"""
    print("\n[uttej 처리 중]", flush=True)
    all_feats = []
    all_labels = []
    all_meta = []

    for subdir, lbl_kind in [("Fall", ("all", None)), ("No_Fall", None)]:
        vdir = os.path.join(UTTEJ_DIR, subdir, "Raw_Video")
        if not os.path.exists(vdir):
            continue
        videos = sorted([f for f in os.listdir(vdir) if f.lower().endswith(".mp4")])[:UTTEJ_PER_CLASS]
        for vf in videos:
            vpath = os.path.join(vdir, vf)
            feats, labels = extract_video_features(vpath, pose, fall_frames=lbl_kind)
            if feats is not None and len(feats) > 0:
                all_feats.append(feats)
                all_labels.append(labels)
                all_meta.append({"source": "uttej", "kind": subdir, "video": vf})
        print(f"  uttej/{subdir}: {len(videos)}개 영상 처리 완료", flush=True)

    return all_feats, all_labels, all_meta


def main():
    print("=" * 70)
    print("Phase 2 - Step 1: 시계열 특징 추출")
    print("=" * 70)
    print(f"  Le2i  : 시나리오별 {LE2I_PER_SCENARIO}개 영상")
    print(f"  uttej : 클래스별 {UTTEJ_PER_CLASS}개 영상")
    print(f"  Frame skip: 1/{FRAME_SKIP}")
    print(f"  특징 수: 53 features/frame")
    print(f"  저장 위치: {OUTPUT_DIR}")
    print("=" * 70)

    pose = mp.solutions.pose.Pose(model_complexity=0)

    import time
    t0 = time.time()
    le2i_feats, le2i_labels, le2i_meta = collect_le2i(pose)
    uttej_feats, uttej_labels, uttej_meta = collect_uttej(pose)

    all_feats = le2i_feats + uttej_feats
    all_labels = le2i_labels + uttej_labels
    all_meta = le2i_meta + uttej_meta

    print(f"\n총 영상 수: {len(all_feats)}")
    print(f"총 프레임 수: {sum(len(f) for f in all_feats)}")
    print(f"낙상 프레임 수: {sum(int(l.sum()) for l in all_labels)}")
    print(f"처리 시간: {(time.time()-t0)/60:.1f}분")

    # 가변 길이 시퀀스 → 영상별로 npz 저장
    out_path = OUTPUT_DIR / "features.npz"
    np.savez_compressed(
        out_path,
        feats=np.array(all_feats, dtype=object),
        labels=np.array(all_labels, dtype=object),
        meta=np.array(all_meta, dtype=object),
    )
    print(f"\n저장 완료: {out_path}")
    print(f"다음 단계: python train_phase2_xgboost.py")


if __name__ == "__main__":
    main()

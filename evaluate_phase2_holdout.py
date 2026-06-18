"""
Phase 2 통합 평가 - Holdout 검증 (학습에 안 쓴 새 영상만 사용)
- 학습/이전평가 사용 범위: 정렬 후 앞 [0:N] 영상
- Holdout 평가 범위:       정렬 후 [N : N+M] 영상  → 모델이 한 번도 본 적 없음
- 진짜 일반화 성능 측정 (데이터 누수 0%)
"""
import os
import cv2
import time
import mediapipe as mp

from detector import mediapipe_detector
from detector.yolo_detector import detect_fall_yolo
from detector.mediapipe_detector import (
    detect_fall_mediapipe, posture_status, get_threshold_for_posture as get_threshold,
)
from detector.videomae_detector import detect_fall_videomae
from detector.phase2_detector import Phase2Detector

LE2I_DIR = r"data\falld_2"
UTTEJ_DIR = r"data\falld_3"

# ============ Holdout 설정 ============
# uttej: 학습에 [0:100] 사용 → holdout은 [100:100+UTTEJ_HOLDOUT]
UTTEJ_TRAINED = 100
UTTEJ_HOLDOUT = 100      # 새 영상 100개씩 (총 200개)

# Le2i: 학습에 [0:10] 사용 → holdout은 [10:10+LE2I_HOLDOUT]
LE2I_TRAINED = 10
LE2I_HOLDOUT = 10        # 시나리오별 새 영상 10개씩

FRAME_SKIP = 3
DURATION_FRAMES = 3
XGBOOST_THRESHOLD = 0.7
MODES = ["rule_only", "ml_only", "ensemble_or", "ensemble_and"]


def reset_states(phase2):
    mediapipe_detector.prev_nose_y = None
    mediapipe_detector.prev_shoulder_y = None
    mediapipe_detector.prev_hip_y = None
    mediapipe_detector.prev_time = None
    mediapipe_detector.prev_velocity = 0
    mediapipe_detector.fall_start_time = None
    mediapipe_detector.stillness_start_time = None
    mediapipe_detector.last_posture = None
    mediapipe_detector.posture_start_time = None
    mediapipe_detector.motion_history = []
    mediapipe_detector.abnormal_posture_start = None
    mediapipe_detector.last_upright_time = None
    posture_status["posture"] = "unknown"
    posture_status["stillness_sec"] = 0
    posture_status["velocity"] = 0
    posture_status["abnormal"] = False
    posture_status["micro_motion"] = 0
    phase2.reset()


def process_video(video_path, pose, phase2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    reset_states(phase2)

    detected = {m: [] for m in MODES}
    consec = {m: 0 for m in MODES}
    max_xgb = 0.0

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
        processed += 1
        video_time = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        mp_score = 0
        if results.pose_landmarks:
            mp_score = detect_fall_mediapipe(
                results.pose_landmarks.landmark, frame.shape[0], current_time=video_time
            )
        yolo_score = detect_fall_yolo(frame)
        videomae_score = 0
        try:
            if processed % 8 == 0:
                resized = cv2.resize(frame, (224, 224))
                rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                videomae_score = detect_fall_videomae(rgb_small)
        except Exception:
            pass
        total_score = mp_score + yolo_score + videomae_score

        posture = posture_status.get("posture", "unknown")
        threshold = get_threshold(posture)
        stillness = posture_status.get("stillness_sec", 0)
        abnormal = posture_status.get("abnormal", False)
        accident = (posture == "lying" and stillness >= 10.0) or \
                   (posture in ("sitting", "standing") and abnormal and stillness >= 5.0)
        rule_pos = (total_score >= threshold) or accident

        try:
            xgb_proba = phase2.predict(frame, results, frame.shape[0])
        except Exception:
            xgb_proba = 0.0
        ml_pos = xgb_proba >= XGBOOST_THRESHOLD
        max_xgb = max(max_xgb, xgb_proba)

        mode_pos = {
            "rule_only":     rule_pos,
            "ml_only":       ml_pos,
            "ensemble_or":   rule_pos or ml_pos,
            "ensemble_and":  rule_pos and ml_pos,
        }

        for m in MODES:
            if mode_pos[m]:
                consec[m] += 1
                if consec[m] >= DURATION_FRAMES:
                    detected[m].append(frame_idx)
            else:
                consec[m] = 0

    cap.release()
    return detected, max_xgb


def find_le2i_pairs():
    pairs_by_scenario = {}
    for room in sorted(os.listdir(LE2I_DIR)):
        room_path = os.path.join(LE2I_DIR, room)
        if not os.path.isdir(room_path):
            continue
        for cur, dirs, files in os.walk(room_path):
            depth = cur[len(room_path):].count(os.sep)
            if depth > 4:
                continue
            if "Videos" in dirs and "Annotation_files" in dirs:
                vd = os.path.join(cur, "Videos")
                ad = os.path.join(cur, "Annotation_files")
                pairs = []
                for vf in sorted(os.listdir(vd)):
                    if vf.lower().endswith(".avi"):
                        af = os.path.join(ad, os.path.splitext(vf)[0] + ".txt")
                        if os.path.exists(af):
                            pairs.append((os.path.join(vd, vf), af))
                if pairs:
                    pairs_by_scenario[room] = pairs
                    break
    return pairs_by_scenario


def evaluate_le2i(pose, phase2):
    print("\n" + "=" * 70, flush=True)
    print(f"Le2i Holdout 평가 - 시나리오별 [{LE2I_TRAINED}:{LE2I_TRAINED+LE2I_HOLDOUT}] (학습에 안 쓴 영상)", flush=True)
    print("=" * 70, flush=True)
    pairs_by_sc = find_le2i_pairs()

    results = {m: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for m in MODES}
    start = time.time()

    for scenario, pairs in pairs_by_sc.items():
        # Holdout 범위만 사용
        holdout_pairs = pairs[LE2I_TRAINED:LE2I_TRAINED + LE2I_HOLDOUT]
        if not holdout_pairs:
            print(f"\n  [{scenario}] holdout 영상 없음 (전체 {len(pairs)}개만 있음)", flush=True)
            continue
        print(f"\n  [{scenario}] {len(holdout_pairs)}개 (영상 {LE2I_TRAINED+1}~{LE2I_TRAINED+len(holdout_pairs)})", flush=True)

        for i, (vp, ap) in enumerate(holdout_pairs):
            try:
                with open(ap, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                fall_start = int(lines[0]); fall_end = int(lines[1])
            except Exception:
                continue

            t0 = time.time()
            ret = process_video(vp, pose, phase2)
            if ret is None: continue
            detected, max_xgb = ret
            elapsed = time.time() - t0

            no_fall_event = (fall_start == 0 and fall_end == 0)
            lo, hi = fall_start - 30, fall_end + 60

            for m in MODES:
                df = detected[m]
                in_win = [] if no_fall_event else [f for f in df if lo <= f <= hi]
                out_win = df if no_fall_event else [f for f in df if f < lo or f > hi]
                if no_fall_event:
                    if df: results[m]["fp"] += 1
                    else:  results[m]["tn"] += 1
                else:
                    if in_win: results[m]["tp"] += 1
                    else:      results[m]["fn"] += 1
                    if out_win: results[m]["fp"] += 1
                    else:       results[m]["tn"] += 1

            tag = "[비낙상]" if no_fall_event else f"[{fall_start}-{fall_end}]"
            r_n = len(detected["rule_only"])
            m_n = len(detected["ml_only"])
            o_n = len(detected["ensemble_or"])
            a_n = len(detected["ensemble_and"])
            print(f"    [{i+1:2d}] {os.path.basename(vp):20s} {tag} | rule:{r_n} ml:{m_n} or:{o_n} and:{a_n} | xgb_max:{max_xgb:.2f} | {elapsed:.1f}s", flush=True)

    print(f"\n  Le2i 처리 시간: {(time.time()-start)/60:.1f}분", flush=True)
    return results


def evaluate_uttej(pose, phase2):
    print("\n" + "=" * 70, flush=True)
    print(f"uttej Holdout 평가 - [{UTTEJ_TRAINED}:{UTTEJ_TRAINED+UTTEJ_HOLDOUT}] (학습에 안 쓴 영상)", flush=True)
    print("=" * 70, flush=True)

    fall_dir = os.path.join(UTTEJ_DIR, "Fall", "Raw_Video")
    no_fall_dir = os.path.join(UTTEJ_DIR, "No_Fall", "Raw_Video")
    fall_videos = sorted([os.path.join(fall_dir, f) for f in os.listdir(fall_dir)
                          if f.lower().endswith(".mp4")])[UTTEJ_TRAINED:UTTEJ_TRAINED+UTTEJ_HOLDOUT]
    no_fall_videos = sorted([os.path.join(no_fall_dir, f) for f in os.listdir(no_fall_dir)
                             if f.lower().endswith(".mp4")])[UTTEJ_TRAINED:UTTEJ_TRAINED+UTTEJ_HOLDOUT]

    print(f"  낙상 {len(fall_videos)}개 (영상 {UTTEJ_TRAINED+1}~{UTTEJ_TRAINED+len(fall_videos)})", flush=True)
    print(f"  정상 {len(no_fall_videos)}개 (영상 {UTTEJ_TRAINED+1}~{UTTEJ_TRAINED+len(no_fall_videos)})", flush=True)

    results = {m: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for m in MODES}
    start = time.time()

    print("\n  [낙상 영상]", flush=True)
    for i, vp in enumerate(fall_videos):
        ret = process_video(vp, pose, phase2)
        if ret is None: continue
        detected, _ = ret
        for m in MODES:
            if detected[m]: results[m]["tp"] += 1
            else:           results[m]["fn"] += 1
        if (i+1) % 10 == 0:
            line = f"    낙상 {i+1:3d}/{len(fall_videos)} | "
            for m in MODES:
                sens = results[m]["tp"] / max(1, results[m]["tp"] + results[m]["fn"]) * 100
                line += f"{m}:{sens:.0f}% "
            print(line, flush=True)

    print("\n  [정상 영상]", flush=True)
    for i, vp in enumerate(no_fall_videos):
        ret = process_video(vp, pose, phase2)
        if ret is None: continue
        detected, _ = ret
        for m in MODES:
            if detected[m]: results[m]["fp"] += 1
            else:           results[m]["tn"] += 1
        if (i+1) % 10 == 0:
            line = f"    정상 {i+1:3d}/{len(no_fall_videos)} | "
            for m in MODES:
                spec = results[m]["tn"] / max(1, results[m]["tn"] + results[m]["fp"]) * 100
                line += f"{m}:{spec:.0f}% "
            print(line, flush=True)

    print(f"\n  uttej 처리 시간: {(time.time()-start)/60:.1f}분", flush=True)
    return results


def print_metrics(name, results):
    print("\n" + "=" * 70, flush=True)
    print(f"{name} - Holdout 평가 (학습 안 쓴 새 영상)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'모드':<15} {'정확도':>7} {'민감도':>7} {'특이도':>7} {'정밀도':>7} {'F1':>7}  |  TP  FN  TN  FP", flush=True)
    for m in MODES:
        r = results[m]
        total = r["tp"] + r["fn"] + r["tn"] + r["fp"]
        acc = (r["tp"] + r["tn"]) / total * 100 if total else 0
        sens = r["tp"] / max(1, r["tp"] + r["fn"]) * 100
        spec = r["tn"] / max(1, r["tn"] + r["fp"]) * 100
        prec = r["tp"] / max(1, r["tp"] + r["fp"]) * 100
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0
        print(f"  {m:<15} {acc:>6.1f}% {sens:>6.1f}% {spec:>6.1f}% {prec:>6.1f}% {f1:>6.1f}   |  {r['tp']:>3} {r['fn']:>3} {r['tn']:>3} {r['fp']:>3}", flush=True)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("Phase 2 Holdout 평가 (데이터 누수 0% - 진짜 일반화 성능)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Le2i  : 시나리오별 [{LE2I_TRAINED}:{LE2I_TRAINED+LE2I_HOLDOUT}] - 학습에 안 쓴 새 영상", flush=True)
    print(f"  uttej : 클래스별  [{UTTEJ_TRAINED}:{UTTEJ_TRAINED+UTTEJ_HOLDOUT}] - 학습에 안 쓴 새 영상", flush=True)
    print(f"  XGBoost threshold: {XGBOOST_THRESHOLD}", flush=True)
    print("=" * 70, flush=True)

    pose = mp.solutions.pose.Pose(model_complexity=0)
    phase2 = Phase2Detector()

    overall_start = time.time()
    le2i_results = evaluate_le2i(pose, phase2)
    uttej_results = evaluate_uttej(pose, phase2)

    print_metrics("Le2i Holdout (낙상 사건 단위)", le2i_results)
    print_metrics("uttej Holdout (영상 단위 binary)", uttej_results)

    print(f"\n총 처리 시간: {(time.time()-overall_start)/60:.1f}분", flush=True)
    print("\n=== 진짜 일반화 성능 (데이터 누수 0%) ===", flush=True)
    print("이 수치가 시스템의 진짜 실력입니다.", flush=True)

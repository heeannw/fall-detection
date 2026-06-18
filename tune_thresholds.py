"""
Phase 1 - 길 1: 임계값 그리드 서치
- mediapipe_detector.py의 핵심 임계값들을 데이터로 자동 튜닝
- 작은 샘플(Le2i 시나리오별 3개 + uttej 각 15개)로 빠른 평가
- F1 점수가 가장 높은 조합 찾기 → 결과를 콘솔 출력 + JSON 저장
"""
import os
import cv2
import time
import json
import itertools
import mediapipe as mp

from detector import mediapipe_detector
from detector.yolo_detector import detect_fall_yolo
from detector.mediapipe_detector import detect_fall_mediapipe, posture_status, get_threshold_for_posture

# ============================================================
# 평가용 데이터셋 경로 (evaluate_video.py와 동일)
# ============================================================
LE2I_DIR = r"data\falld_2"
UTTEJ_DIR = r"data\falld_3"

# 작은 샘플 크기 (그리드 서치용)
LE2I_PER_SCENARIO = 3
UTTEJ_PER_CLASS = 15
FRAME_SKIP = 3
DURATION_FRAMES = 5

# ============================================================
# 그리드 서치 대상 임계값 (모듈 변수 이름 기준)
# 너무 많으면 폭주: 핵심 3-4개만, 각 2-3 값
# ============================================================
GRID = {
    "LYING_HORIZONTAL_RATIO":    [0.50, 0.55, 0.60],
    "LYING_SH_DIFF_MAX":         [0.12, 0.15, 0.20],
    "FALL_THRESHOLD_LYING":      [3, 4],
    "UPRIGHT_TO_LYING_WINDOW":   [3.0, 5.0, 7.0],
}
# 총 조합 수 = 3 × 3 × 2 × 3 = 54


def reset_mediapipe_state():
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


def process_video(video_path, pose):
    """영상 처리: detected_frames 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    reset_mediapipe_state()

    detected_frames = []
    consecutive_pos = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        mp_score = 0
        if results.pose_landmarks:
            mp_score = detect_fall_mediapipe(results.pose_landmarks.landmark, frame.shape[0])
        yolo_score = detect_fall_yolo(frame)
        total = mp_score + yolo_score

        posture = posture_status.get("posture", "unknown")
        threshold = get_threshold_for_posture(posture)
        stillness = posture_status.get("stillness_sec", 0)
        abnormal = posture_status.get("abnormal", False)

        accident = (posture == "lying" and stillness >= 10.0) or \
                   (posture in ("sitting", "standing") and abnormal and stillness >= 5.0)

        is_pos = (total >= threshold) or accident
        if is_pos:
            consecutive_pos += 1
            if consecutive_pos >= DURATION_FRAMES:
                detected_frames.append(frame_idx)
        else:
            consecutive_pos = 0

    cap.release()
    return detected_frames


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


def evaluate_combo(pose, le2i_pairs_by_sc, uttej_fall, uttej_no_fall):
    """현재 mediapipe_detector 임계값으로 평가 → F1 등 반환"""
    tp = fn = tn = fp = 0

    # Le2i (사건 단위)
    for sc, pairs in le2i_pairs_by_sc.items():
        for vp, ap in pairs[:LE2I_PER_SCENARIO]:
            try:
                with open(ap, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                fall_start = int(lines[0])
                fall_end = int(lines[1])
                detected = process_video(vp, pose)
                if detected is None:
                    continue
                no_fall_event = (fall_start == 0 and fall_end == 0)
                lo, hi = fall_start - 30, fall_end + 60
                in_win = [] if no_fall_event else [f for f in detected if lo <= f <= hi]
                out_win = detected if no_fall_event else [f for f in detected if f < lo or f > hi]
                if no_fall_event:
                    if detected: fp += 1
                    else:        tn += 1
                else:
                    if in_win: tp += 1
                    else:      fn += 1
                    if out_win: fp += 1
                    else:       tn += 1
            except Exception:
                continue

    # uttej binary
    for vp in uttej_fall:
        try:
            detected = process_video(vp, pose)
            if detected is None: continue
            if detected: tp += 1
            else:        fn += 1
        except Exception:
            continue
    for vp in uttej_no_fall:
        try:
            detected = process_video(vp, pose)
            if detected is None: continue
            if detected: fp += 1
            else:        tn += 1
        except Exception:
            continue

    total = tp + fn + tn + fp
    if total == 0:
        return {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "acc": 0, "sens": 0, "spec": 0, "prec": 0, "f1": 0}
    acc = (tp + tn) / total * 100
    sens = tp / (tp + fn) * 100 if (tp + fn) else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "acc": round(acc, 1), "sens": round(sens, 1),
            "spec": round(spec, 1), "prec": round(prec, 1), "f1": round(f1, 1)}


def apply_combo(combo):
    """combo: dict[name] = value → mediapipe_detector 모듈 변수 업데이트"""
    for name, value in combo.items():
        setattr(mediapipe_detector, name, value)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("Phase 1 - 길 1: 임계값 그리드 서치", flush=True)
    print("=" * 70, flush=True)

    # 그리드 조합 생성
    keys = list(GRID.keys())
    value_lists = [GRID[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))

    print(f"  총 조합 수: {len(combos)}", flush=True)
    print(f"  그리드:", flush=True)
    for k, v in GRID.items():
        print(f"    {k}: {v}", flush=True)
    print(f"  샘플: Le2i 시나리오별 {LE2I_PER_SCENARIO}개, uttej 클래스별 {UTTEJ_PER_CLASS}개", flush=True)

    # 데이터셋 미리 로드 (조합별로 동일 영상 사용)
    pose = mp.solutions.pose.Pose(model_complexity=0)
    le2i_pairs_by_sc = find_le2i_pairs()
    print(f"\n  발견된 Le2i 시나리오: {list(le2i_pairs_by_sc.keys())}", flush=True)

    fall_dir = os.path.join(UTTEJ_DIR, "Fall", "Raw_Video")
    no_fall_dir = os.path.join(UTTEJ_DIR, "No_Fall", "Raw_Video")
    uttej_fall = sorted([os.path.join(fall_dir, f) for f in os.listdir(fall_dir)
                         if f.lower().endswith(".mp4")])[:UTTEJ_PER_CLASS]
    uttej_no_fall = sorted([os.path.join(no_fall_dir, f) for f in os.listdir(no_fall_dir)
                            if f.lower().endswith(".mp4")])[:UTTEJ_PER_CLASS]
    print(f"  uttej 낙상 {len(uttej_fall)}개 / 정상 {len(uttej_no_fall)}개", flush=True)

    print("\n  Baseline (현재 디폴트값) 평가...", flush=True)
    t0 = time.time()
    # 디폴트 값으로 한 번 측정
    baseline_combo = {k: getattr(mediapipe_detector, k) for k in keys}
    baseline = evaluate_combo(pose, le2i_pairs_by_sc, uttej_fall, uttej_no_fall)
    baseline_time = time.time() - t0
    print(f"  Baseline F1={baseline['f1']} acc={baseline['acc']} sens={baseline['sens']} spec={baseline['spec']} prec={baseline['prec']}", flush=True)
    print(f"  조합당 평가 시간 ~{baseline_time:.1f}초, 전체 예상 ~{baseline_time*len(combos)/60:.1f}분", flush=True)

    results = []
    overall_start = time.time()
    for i, combo in enumerate(combos):
        apply_combo(combo)
        t0 = time.time()
        metrics = evaluate_combo(pose, le2i_pairs_by_sc, uttej_fall, uttej_no_fall)
        elapsed = time.time() - t0
        results.append({"combo": combo, "metrics": metrics, "time": elapsed})

        eta_min = (time.time() - overall_start) / (i + 1) * (len(combos) - i - 1) / 60
        print(f"  [{i+1:2d}/{len(combos)}] F1={metrics['f1']:>5.1f} acc={metrics['acc']:>5.1f} "
              f"sens={metrics['sens']:>5.1f} spec={metrics['spec']:>5.1f} prec={metrics['prec']:>5.1f} | "
              f"{combo} | {elapsed:.0f}s (남은 ~{eta_min:.0f}분)",
              flush=True)

    # 결과 정렬 (F1 내림차순)
    results.sort(key=lambda x: x["metrics"]["f1"], reverse=True)

    print("\n" + "=" * 70, flush=True)
    print("Top 5 조합 (F1 기준)", flush=True)
    print("=" * 70, flush=True)
    for r in results[:5]:
        m = r["metrics"]
        print(f"  F1={m['f1']:>5.1f} acc={m['acc']:>5.1f} sens={m['sens']:>5.1f} "
              f"spec={m['spec']:>5.1f} prec={m['prec']:>5.1f} | {r['combo']}", flush=True)

    print("\nBaseline 대비 Top 1 개선:", flush=True)
    top = results[0]["metrics"]
    print(f"  F1:   {baseline['f1']} → {top['f1']} ({top['f1']-baseline['f1']:+.1f})", flush=True)
    print(f"  acc:  {baseline['acc']} → {top['acc']} ({top['acc']-baseline['acc']:+.1f})", flush=True)
    print(f"  sens: {baseline['sens']} → {top['sens']} ({top['sens']-baseline['sens']:+.1f})", flush=True)
    print(f"  spec: {baseline['spec']} → {top['spec']} ({top['spec']-baseline['spec']:+.1f})", flush=True)
    print(f"  prec: {baseline['prec']} → {top['prec']} ({top['prec']-baseline['prec']:+.1f})", flush=True)

    print(f"\n총 처리 시간: {(time.time()-overall_start)/60:.1f}분", flush=True)

    # 최적 조합을 JSON으로 저장
    output = {
        "baseline": {"combo": baseline_combo, "metrics": baseline},
        "best": results[0],
        "top5": results[:5],
        "all": results,
        "grid": GRID,
    }
    with open("tune_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n결과 JSON 저장: tune_results.json", flush=True)
    print(f"\n다음 단계: 'apply_best_thresholds.py' 실행하면 최적 조합을 mediapipe_detector.py에 자동 적용", flush=True)

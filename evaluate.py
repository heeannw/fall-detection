import cv2
import os
import mediapipe as mp
from detector import mediapipe_detector
from detector.yolo_detector import detect_fall_yolo, model as yolo_model
from detector.mediapipe_detector import detect_fall_mediapipe, posture_status
from detector.videomae_detector import detect_fall_videomae

mp_pose = mp.solutions.pose

FALL_DIR = r"data\archive\Labelled Dataset\Fall"
NOT_FALL_DIR = r"data\archive\Labelled Dataset\Not Fall"


def reset_mediapipe_state():
    """이미지 단위 평가: 매 이미지마다 시간 기반 상태 + posture_status 초기화"""
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
    mediapipe_detector.last_upright_time = None   # 자세 전환 추적도 리셋
    # posture_status도 reset (이전 이미지 자세 누수 차단)
    posture_status["posture"] = "unknown"
    posture_status["stillness_sec"] = 0
    posture_status["velocity"] = 0
    posture_status["abnormal"] = False
    posture_status["micro_motion"] = 0


def get_threshold_for_posture(posture):
    """main.py와 동일한 자세별 임계값"""
    if posture == "lying":
        return 3
    elif posture == "sitting":
        return 5
    elif posture == "standing":
        return 4
    else:
        return 4


def yolo_raw_max_conf(frame):
    """YOLO가 fall 클래스에 대해 출력한 '원시' 최고 신뢰도 (임계값 우회)"""
    results = yolo_model(frame, verbose=False)
    max_fall_conf = 0.0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls]
            if "fall" in label.lower() and conf > max_fall_conf:
                max_fall_conf = conf
    return max_fall_conf


def process_image(frame, pose, diag):
    """이미지 1장 평가 + 진단 정보 기록."""
    reset_mediapipe_state()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # ---- MediaPipe 진단 ----
    landmark_found = results.pose_landmarks is not None
    shoulder_vis = hip_vis = knee_vis = 0.0
    if landmark_found:
        lms = results.pose_landmarks.landmark
        shoulder_vis = (lms[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility +
                        lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility) / 2
        hip_vis = (lms[mp_pose.PoseLandmark.LEFT_HIP].visibility +
                   lms[mp_pose.PoseLandmark.RIGHT_HIP].visibility) / 2
        knee_vis = (lms[mp_pose.PoseLandmark.LEFT_KNEE].visibility +
                    lms[mp_pose.PoseLandmark.RIGHT_KNEE].visibility) / 2

    mediapipe_score = 0
    if landmark_found:
        mediapipe_score = detect_fall_mediapipe(results.pose_landmarks.landmark, frame.shape[0])

    yolo_score = detect_fall_yolo(frame)
    yolo_raw_conf = yolo_raw_max_conf(frame)
    videomae_score = detect_fall_videomae(frame)
    total_score = mediapipe_score + yolo_score + videomae_score

    posture = posture_status.get("posture", "unknown")
    threshold = get_threshold_for_posture(posture)

    # 진단 카운터
    if not landmark_found:
        diag["no_landmark"] += 1
    else:
        if shoulder_vis < 0.5:
            diag["low_shoulder_vis"] += 1
        if hip_vis < 0.5:
            diag["low_hip_vis"] += 1
        if knee_vis < 0.5:
            diag["low_knee_vis"] += 1

    diag["yolo_conf_max"] = max(diag["yolo_conf_max"], yolo_raw_conf)
    if yolo_raw_conf >= 0.5:
        diag["yolo_above_0.5"] += 1
    if yolo_raw_conf >= 0.7:
        diag["yolo_above_0.7"] += 1
    if yolo_raw_conf >= 0.85:
        diag["yolo_above_0.85"] += 1

    return {
        "mediapipe_score": mediapipe_score,
        "yolo_score": yolo_score,
        "yolo_raw_conf": yolo_raw_conf,
        "videomae_score": videomae_score,
        "total_score": total_score,
        "posture": posture,
        "threshold": threshold,
        "landmark_found": landmark_found,
        "shoulder_vis": shoulder_vis,
        "hip_vis": hip_vis,
        "knee_vis": knee_vis,
    }


def evaluate():
    pose = mp_pose.Pose()

    tp = 0; fn = 0; tn = 0; fp = 0
    fall_posture_dist = {"lying": 0, "sitting": 0, "standing": 0, "unknown": 0}
    not_fall_posture_dist = {"lying": 0, "sitting": 0, "standing": 0, "unknown": 0}
    fall_by_posture = {"lying": [0, 0], "sitting": [0, 0], "standing": [0, 0], "unknown": [0, 0]}
    not_fall_by_posture = {"lying": [0, 0], "sitting": [0, 0], "standing": [0, 0], "unknown": [0, 0]}

    # 진단 카운터 (낙상셋 / 정상셋 별도)
    fall_diag = {"no_landmark": 0, "low_shoulder_vis": 0, "low_hip_vis": 0, "low_knee_vis": 0,
                 "yolo_conf_max": 0.0, "yolo_above_0.5": 0, "yolo_above_0.7": 0, "yolo_above_0.85": 0}
    not_fall_diag = {"no_landmark": 0, "low_shoulder_vis": 0, "low_hip_vis": 0, "low_knee_vis": 0,
                     "yolo_conf_max": 0.0, "yolo_above_0.5": 0, "yolo_above_0.7": 0, "yolo_above_0.85": 0}

    # 미탐(FN) 케이스 상세 추적
    fn_cases = []  # [(filename, posture, scores, landmark_found, vis), ...]

    print("낙상 이미지 테스트 중...")
    fall_files = os.listdir(FALL_DIR)[:100]
    for i, fname in enumerate(fall_files):
        path = os.path.join(FALL_DIR, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        r = process_image(frame, pose, fall_diag)
        fall_posture_dist[r["posture"]] = fall_posture_dist.get(r["posture"], 0) + 1

        if (i + 1) % 20 == 0:
            print(f"  낙상 {i+1}/{len(fall_files)} | 자세:{r['posture']:>8} | MP:{r['mediapipe_score']} YOLO:{r['yolo_score']}(raw:{r['yolo_raw_conf']:.2f}) MAE:{r['videomae_score']} 합계:{r['total_score']}/{r['threshold']} | LM:{r['landmark_found']} vis(s/h/k):{r['shoulder_vis']:.2f}/{r['hip_vis']:.2f}/{r['knee_vis']:.2f}")

        if r["total_score"] >= r["threshold"]:
            tp += 1
            fall_by_posture[r["posture"]][0] += 1
        else:
            fn += 1
            fall_by_posture[r["posture"]][1] += 1
            if len(fn_cases) < 10:
                fn_cases.append((fname, r["posture"], r["mediapipe_score"], r["yolo_raw_conf"], r["videomae_score"],
                                 r["landmark_found"], r["shoulder_vis"], r["hip_vis"], r["knee_vis"]))

    print("\n정상 이미지 테스트 중...")
    not_fall_files = os.listdir(NOT_FALL_DIR)[:100]
    for i, fname in enumerate(not_fall_files):
        path = os.path.join(NOT_FALL_DIR, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        r = process_image(frame, pose, not_fall_diag)
        not_fall_posture_dist[r["posture"]] = not_fall_posture_dist.get(r["posture"], 0) + 1

        if (i + 1) % 20 == 0:
            print(f"  정상 {i+1}/{len(not_fall_files)} | 자세:{r['posture']:>8} | MP:{r['mediapipe_score']} YOLO:{r['yolo_score']}(raw:{r['yolo_raw_conf']:.2f}) MAE:{r['videomae_score']} 합계:{r['total_score']}/{r['threshold']} | LM:{r['landmark_found']} vis(s/h/k):{r['shoulder_vis']:.2f}/{r['hip_vis']:.2f}/{r['knee_vis']:.2f}")

        if r["total_score"] >= r["threshold"]:
            fp += 1
            not_fall_by_posture[r["posture"]][1] += 1
        else:
            tn += 1
            not_fall_by_posture[r["posture"]][0] += 1

    total = tp + fn + tn + fp
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0

    print("\n===== 정확도 결과 (자세별 임계값 적용) =====")
    print(f"전체 정확도:           {accuracy:.1f}%")
    print(f"민감도 (낙상 감지율):  {sensitivity:.1f}%")
    print(f"특이도 (정상 정확률):  {specificity:.1f}%")
    print(f"정밀도 (신뢰도):       {precision:.1f}%")
    print(f"F1 Score:              {f1:.1f}")
    print(f"\nTP(낙상→낙상): {tp}    FN(낙상→정상): {fn}")
    print(f"TN(정상→정상): {tn}    FP(정상→낙상): {fp}")

    print("\n===== 자세 분포 =====")
    print(f"  {'자세':>8} | {'낙상셋':>6} | {'정상셋':>6}")
    for p in ["lying", "sitting", "standing", "unknown"]:
        print(f"  {p:>8} | {fall_posture_dist.get(p, 0):>6} | {not_fall_posture_dist.get(p, 0):>6}")

    print("\n===== 자세별 세부 성능 =====")
    print(f"  {'자세':>8} | {'TP':>3} {'FN':>3} | {'TN':>3} {'FP':>3} | 민감도   특이도")
    for p in ["lying", "sitting", "standing", "unknown"]:
        tp_p, fn_p = fall_by_posture[p]
        tn_p, fp_p = not_fall_by_posture[p]
        sens_p = tp_p / (tp_p + fn_p) * 100 if (tp_p + fn_p) > 0 else 0
        spec_p = tn_p / (tn_p + fp_p) * 100 if (tn_p + fp_p) > 0 else 0
        print(f"  {p:>8} | {tp_p:>3} {fn_p:>3} | {tn_p:>3} {fp_p:>3} | {sens_p:>5.1f}%  {spec_p:>5.1f}%")

    print("\n===== 진단 정보 (왜 unknown으로 분류되었는가) =====")
    print(f"  {'항목':<25} | {'낙상셋':>6} | {'정상셋':>6}")
    print(f"  {'landmark 미검출':<25} | {fall_diag['no_landmark']:>6} | {not_fall_diag['no_landmark']:>6}")
    print(f"  {'shoulder visibility<0.5':<25} | {fall_diag['low_shoulder_vis']:>6} | {not_fall_diag['low_shoulder_vis']:>6}")
    print(f"  {'hip visibility<0.5':<25} | {fall_diag['low_hip_vis']:>6} | {not_fall_diag['low_hip_vis']:>6}")
    print(f"  {'knee visibility<0.5':<25} | {fall_diag['low_knee_vis']:>6} | {not_fall_diag['low_knee_vis']:>6}")

    print("\n===== YOLO 신뢰도 분포 (원시값, 임계값 우회) =====")
    print(f"  {'항목':<25} | {'낙상셋':>6} | {'정상셋':>6}")
    print(f"  {'최고 신뢰도':<25} | {fall_diag['yolo_conf_max']:>6.2f} | {not_fall_diag['yolo_conf_max']:>6.2f}")
    print(f"  {'conf>=0.5 검출 수':<25} | {fall_diag['yolo_above_0.5']:>6} | {not_fall_diag['yolo_above_0.5']:>6}")
    print(f"  {'conf>=0.7 검출 수':<25} | {fall_diag['yolo_above_0.7']:>6} | {not_fall_diag['yolo_above_0.7']:>6}")
    print(f"  {'conf>=0.85 검출 수':<25} | {fall_diag['yolo_above_0.85']:>6} | {not_fall_diag['yolo_above_0.85']:>6}")
    print("  → 낙상셋 conf>=0.5는 많은데 0.85는 적다 = 임계값 너무 높음")
    print("  → 낙상셋도 0.5 미만이 대부분이면 = YOLO 모델 자체가 약함")

    print("\n===== 놓친 낙상(FN) 샘플 10개 =====")
    print(f"  {'파일':<35} {'자세':>8} | MP YOLO_raw MAE | LM vis(s/h/k)")
    for fname, posture, mp_s, yc, mae_s, lm, sv, hv, kv in fn_cases:
        print(f"  {fname[:33]:<35} {posture:>8} | {mp_s}   {yc:.2f}     {mae_s}  | {lm} {sv:.2f}/{hv:.2f}/{kv:.2f}")

    print("\n===== 자세별 임계값 (main.py와 동일) =====")
    print("  lying=3pt, sitting=5pt, standing=4pt, unknown=4pt")


if __name__ == "__main__":
    evaluate()

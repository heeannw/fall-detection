"""
옵션 B: 영상 기반 평가 스크립트
- falld_2 (Le2i): Annotation 기반 프레임 단위 정확한 평가
- falld_3 (uttejkumarkandagatla): 폴더 기반 binary 평가 (Fall vs No_Fall)
- VideoMAE 제외 (영상에서 16프레임 버퍼링 비용 너무 큼 - main.py 실시간은 그대로 동작)
- 시간 기반 로직(velocity/stillness/accident_by_stillness) 정상 작동 → 진짜 운영 환경에 가까운 평가
"""
import cv2
import os
import time
import sys
import mediapipe as mp
from detector import mediapipe_detector
from detector.yolo_detector import detect_fall_yolo
from detector.mediapipe_detector import detect_fall_mediapipe, posture_status, get_threshold_for_posture as _get_threshold
from detector.videomae_detector import detect_fall_videomae   # Fix #4: VideoMAE 추가
# Phase 2 통합 평가는 별도 스크립트(evaluate_phase2.py) 사용

LE2I_DIR = r"data\falld_2"
UTTEJ_DIR = r"data\falld_3"

LE2I_PER_SCENARIO = 10     # 시나리오별 영상 수
UTTEJ_PER_CLASS = 50       # Fall/No_Fall 각각 영상 수 (진단 모드에선 50으로 축소)
FRAME_SKIP = 3             # 1/3 프레임만 처리
DURATION_FRAMES = 3        # Fix #7: 5→3 완화. 자세 분류 흔들림 대응 (원본 ~9프레임 = ~0.36초)


def reset_mediapipe_state():
    """영상 시작 시 시간 상태 초기화 (영상 간 누수 방지)"""
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
    posture_status["posture"] = "unknown"
    posture_status["stillness_sec"] = 0
    posture_status["velocity"] = 0
    posture_status["abnormal"] = False
    posture_status["micro_motion"] = 0


def get_threshold_for_posture(posture):
    """공용 함수로 위임 (mediapipe_detector.py에서 임계값 관리)"""
    return _get_threshold(posture)


def process_video(video_path, pose):
    """영상 1개 처리. fall_detected=True인 프레임 인덱스 리스트 + 진단 정보 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Fix #5: 영상 fps로 정확한 시간 계산
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0   # fps 안 읽히면 기본 25

    reset_mediapipe_state()

    detected_frames = []
    detect_reasons = {"score_only": 0, "accident_only": 0, "both": 0}
    posture_at_detect = {"lying": 0, "sitting": 0, "standing": 0, "unknown": 0}
    posture_dist_all = {"lying": 0, "sitting": 0, "standing": 0, "unknown": 0}
    no_landmark_count = 0
    low_knee_vis_count = 0
    max_mp_score = 0
    max_yolo_score = 0
    max_videomae_score = 0

    consecutive_pos = 0
    frame_idx = 0
    processed = 0

    # Fix #4: VideoMAE용 프레임 버퍼
    videomae_buffer = []
    cached_videomae_score = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
        processed += 1

        # Fix #5: frame_idx 기반 영상 시간 (wall clock 대신)
        video_time = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        mp_score = 0
        if results.pose_landmarks:
            mp_score = detect_fall_mediapipe(results.pose_landmarks.landmark, frame.shape[0], current_time=video_time)
            lms = results.pose_landmarks.landmark
            knee_vis = (lms[mp.solutions.pose.PoseLandmark.LEFT_KNEE].visibility +
                        lms[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].visibility) / 2
            if knee_vis < 0.5:
                low_knee_vis_count += 1
        else:
            no_landmark_count += 1

        yolo_score = detect_fall_yolo(frame)

        # Fix #4: VideoMAE 평가 추가 (16프레임 버퍼)
        resized = cv2.resize(frame, (224, 224))
        rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        videomae_buffer.append(rgb_small)
        if len(videomae_buffer) > 16:
            videomae_buffer.pop(0)
        # 8 처리프레임마다(약 0.3초) VideoMAE 호출
        if len(videomae_buffer) == 16 and processed % 8 == 0:
            try:
                cached_videomae_score = detect_fall_videomae(rgb_small)
            except Exception:
                cached_videomae_score = 0
        videomae_score = cached_videomae_score
        max_videomae_score = max(max_videomae_score, videomae_score)

        total = mp_score + yolo_score + videomae_score   # Fix #4: VideoMAE 포함

        posture = posture_status.get("posture", "unknown")
        threshold = get_threshold_for_posture(posture)
        stillness = posture_status.get("stillness_sec", 0)
        abnormal = posture_status.get("abnormal", False)

        posture_dist_all[posture] = posture_dist_all.get(posture, 0) + 1
        max_mp_score = max(max_mp_score, mp_score)
        max_yolo_score = max(max_yolo_score, yolo_score)

        # main.py와 동일한 사고 정지 판정
        accident = (posture == "lying" and stillness >= 10.0) or \
                   (posture in ("sitting", "standing") and abnormal and stillness >= 5.0)

        score_pos = total >= threshold
        is_pos = score_pos or accident

        if is_pos:
            consecutive_pos += 1
            if consecutive_pos >= DURATION_FRAMES:
                detected_frames.append(frame_idx)
                # 감지 시 발동 원인 추적
                if score_pos and accident:
                    detect_reasons["both"] += 1
                elif score_pos:
                    detect_reasons["score_only"] += 1
                else:
                    detect_reasons["accident_only"] += 1
                posture_at_detect[posture] = posture_at_detect.get(posture, 0) + 1
        else:
            consecutive_pos = 0

    cap.release()
    diag = {
        "processed_frames": processed,
        "no_landmark": no_landmark_count,
        "low_knee_vis": low_knee_vis_count,
        "posture_dist_all": posture_dist_all,
        "posture_at_detect": posture_at_detect,
        "detect_reasons": detect_reasons,
        "max_mp_score": max_mp_score,
        "max_yolo_score": max_yolo_score,
        "max_videomae_score": max_videomae_score,
    }
    return detected_frames, diag


def find_le2i_pairs():
    """Le2i: 시나리오별 (영상, annotation) 쌍 수집 - os.walk로 어떤 폴더 구조든 다 찾음"""
    pairs_by_scenario = {}
    for room in sorted(os.listdir(LE2I_DIR)):
        room_path = os.path.join(LE2I_DIR, room)
        if not os.path.isdir(room_path):
            continue
        # 재귀적으로 Videos + Annotation_files 폴더 페어 찾기
        found = False
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
                    found = True
                    break
        if not found:
            print(f"  [경고] {room}: Videos/Annotation_files 폴더를 찾지 못함", flush=True)
    return pairs_by_scenario


def evaluate_le2i(pose):
    print("\n" + "=" * 70, flush=True)
    print("Le2i (falld_2) 평가 시작 - Annotation 기반", flush=True)
    print("=" * 70, flush=True)
    pairs_by_sc = find_le2i_pairs()
    print(f"  발견된 시나리오: {list(pairs_by_sc.keys())}", flush=True)

    total_tp = total_fn = total_fp = total_tn = 0
    by_scenario = {}
    # FN/FP 케이스 진단 누적 (모든 시나리오 통합)
    fn_diag_agg = {"no_landmark": 0, "low_knee_vis": 0, "max_mp": [], "max_yolo": [],
                   "posture_lying": 0, "posture_unknown": 0, "posture_sitting": 0, "posture_standing": 0}
    fp_diag_agg = {"score_only": 0, "accident_only": 0, "both": 0,
                   "posture_lying": 0, "posture_unknown": 0, "posture_sitting": 0, "posture_standing": 0}
    start_time = time.time()

    for scenario, pairs in pairs_by_sc.items():
        selected = pairs[:LE2I_PER_SCENARIO]
        print(f"\n  [{scenario}] {len(selected)}개 영상 평가 시작...", flush=True)
        s_tp = s_fn = s_fp = s_tn = 0

        for i, (vp, ap) in enumerate(selected):
            try:
                with open(ap, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                fall_start = int(lines[0])
                fall_end = int(lines[1])

                t0 = time.time()
                result = process_video(vp, pose)
                if result is None:
                    print(f"    [{i+1}] {os.path.basename(vp)} - 영상 열기 실패", flush=True)
                    continue
                detected, diag = result
                elapsed = time.time() - t0

                # 낙상 구간(시작-30 ~ 끝+60) 내에 한 번이라도 감지되면 TP, 아니면 FN
                # 구간 외 감지 = FP, 외 감지 없으면 TN
                MARGIN_BEFORE = 30
                MARGIN_AFTER = 60
                lo = fall_start - MARGIN_BEFORE
                hi = fall_end + MARGIN_AFTER
                # annotation이 0-0이면 비낙상 영상으로 처리 (Le2i에선 일부 영상)
                no_fall_event = (fall_start == 0 and fall_end == 0)
                in_win = [] if no_fall_event else [f for f in detected if lo <= f <= hi]
                out_win = [f for f in detected if f < lo or f > hi] if not no_fall_event else detected

                if no_fall_event:
                    # 사건 없는 영상 - 감지되면 FP, 아니면 TN
                    if detected: s_fp += 1
                    else:        s_tn += 1
                else:
                    if in_win: s_tp += 1
                    else:     s_fn += 1
                    if out_win: s_fp += 1
                    else:       s_tn += 1

                # FN 케이스 진단 누적
                if not no_fall_event and not in_win:
                    fn_diag_agg["no_landmark"] += diag["no_landmark"]
                    fn_diag_agg["low_knee_vis"] += diag["low_knee_vis"]
                    fn_diag_agg["max_mp"].append(diag["max_mp_score"])
                    fn_diag_agg["max_yolo"].append(diag["max_yolo_score"])
                    pd_all = diag["posture_dist_all"]
                    fn_diag_agg["posture_lying"] += pd_all.get("lying", 0)
                    fn_diag_agg["posture_unknown"] += pd_all.get("unknown", 0)
                    fn_diag_agg["posture_sitting"] += pd_all.get("sitting", 0)
                    fn_diag_agg["posture_standing"] += pd_all.get("standing", 0)
                # FP 케이스 진단 누적
                if out_win or (no_fall_event and detected):
                    dr = diag["detect_reasons"]
                    fp_diag_agg["score_only"] += dr.get("score_only", 0)
                    fp_diag_agg["accident_only"] += dr.get("accident_only", 0)
                    fp_diag_agg["both"] += dr.get("both", 0)
                    pad = diag["posture_at_detect"]
                    fp_diag_agg["posture_lying"] += pad.get("lying", 0)
                    fp_diag_agg["posture_unknown"] += pad.get("unknown", 0)
                    fp_diag_agg["posture_sitting"] += pad.get("sitting", 0)
                    fp_diag_agg["posture_standing"] += pad.get("standing", 0)

                tag = "[비낙상]" if no_fall_event else f"낙상[{fall_start:4d}-{fall_end:4d}]"
                print(f"    [{i+1:2d}/{len(selected)}] {os.path.basename(vp):20s} | "
                      f"{tag} | 감지{len(detected):3d}회 "
                      f"(in:{len(in_win)} out:{len(out_win)}) | "
                      f"TP={bool(in_win)} FP={bool(out_win) or (no_fall_event and bool(detected))} | {elapsed:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"    [{i+1}] 실패: {e}", flush=True)

        by_scenario[scenario] = (s_tp, s_fn, s_fp, s_tn)
        total_tp += s_tp; total_fn += s_fn; total_fp += s_fp; total_tn += s_tn
        sens = s_tp/(s_tp+s_fn)*100 if s_tp+s_fn else 0
        spec = s_tn/(s_tn+s_fp)*100 if s_tn+s_fp else 0
        print(f"    >> {scenario}: 민감도 {sens:.1f}% / 사건외 오탐X {spec:.1f}%", flush=True)

    elapsed = time.time() - start_time
    sens = total_tp/(total_tp+total_fn)*100 if total_tp+total_fn else 0
    spec = total_tn/(total_tn+total_fp)*100 if total_tn+total_fp else 0

    print("\n" + "-" * 70, flush=True)
    print(f"Le2i 종합 (처리 시간 {elapsed/60:.1f}분)", flush=True)
    print(f"  TP(낙상 사건 감지):  {total_tp}", flush=True)
    print(f"  FN(낙상 사건 놓침):  {total_fn}", flush=True)
    print(f"  FP(사건 외 오탐):    {total_fp}", flush=True)
    print(f"  TN(사건 외 OK):      {total_tn}", flush=True)
    print(f"  민감도: {sens:.1f}%   사건외 오탐X: {spec:.1f}%", flush=True)
    print(f"\n  시나리오별 세부:", flush=True)
    print(f"  {'시나리오':<20} | TP  FN  FP  TN | 민감도   사건외OK", flush=True)
    for sc, (a, b, c, d) in by_scenario.items():
        ss = a/(a+b)*100 if a+b else 0
        sp = d/(d+c)*100 if d+c else 0
        print(f"  {sc:<20} | {a:>2}  {b:>2}  {c:>2}  {d:>2} | {ss:>5.1f}%  {sp:>5.1f}%", flush=True)

    # FN 진단 (Le2i에서 놓친 낙상)
    print(f"\n  [Le2i FN 진단 - 왜 놓쳤나]", flush=True)
    if fn_diag_agg["max_mp"]:
        avg_max_mp = sum(fn_diag_agg["max_mp"]) / len(fn_diag_agg["max_mp"])
        avg_max_yolo = sum(fn_diag_agg["max_yolo"]) / len(fn_diag_agg["max_yolo"])
        total_p = sum([fn_diag_agg["posture_lying"], fn_diag_agg["posture_unknown"],
                       fn_diag_agg["posture_sitting"], fn_diag_agg["posture_standing"]])
        print(f"  landmark 미검출 프레임 누적:  {fn_diag_agg['no_landmark']}", flush=True)
        print(f"  knee vis<0.5 프레임 누적:      {fn_diag_agg['low_knee_vis']}", flush=True)
        print(f"  FN 영상 평균 최고 MP 점수:    {avg_max_mp:.1f} (lying threshold=3)", flush=True)
        print(f"  FN 영상 평균 최고 YOLO 점수:  {avg_max_yolo:.1f}", flush=True)
        if total_p > 0:
            print(f"  FN 영상 자세 분포: lying={fn_diag_agg['posture_lying']/total_p*100:.0f}% "
                  f"sitting={fn_diag_agg['posture_sitting']/total_p*100:.0f}% "
                  f"standing={fn_diag_agg['posture_standing']/total_p*100:.0f}% "
                  f"unknown={fn_diag_agg['posture_unknown']/total_p*100:.0f}%", flush=True)
    # FP 진단 (Le2i에서 사건 외 오탐)
    print(f"\n  [Le2i FP 진단 - 왜 오탐했나]", flush=True)
    total_r = fp_diag_agg["score_only"] + fp_diag_agg["accident_only"] + fp_diag_agg["both"]
    if total_r > 0:
        print(f"  발동 원인: score만={fp_diag_agg['score_only']} "
              f"accident만={fp_diag_agg['accident_only']} "
              f"둘 다={fp_diag_agg['both']}", flush=True)
        total_p = sum([fp_diag_agg["posture_lying"], fp_diag_agg["posture_unknown"],
                       fp_diag_agg["posture_sitting"], fp_diag_agg["posture_standing"]])
        if total_p > 0:
            print(f"  감지 시 자세: lying={fp_diag_agg['posture_lying']} "
                  f"sitting={fp_diag_agg['posture_sitting']} "
                  f"standing={fp_diag_agg['posture_standing']} "
                  f"unknown={fp_diag_agg['posture_unknown']}", flush=True)


def evaluate_uttej(pose):
    print("\n" + "=" * 70, flush=True)
    print("uttejkumarkandagatla (falld_3) 평가 시작 - 폴더 기반 binary", flush=True)
    print("=" * 70, flush=True)

    fall_dir = os.path.join(UTTEJ_DIR, "Fall", "Raw_Video")
    no_fall_dir = os.path.join(UTTEJ_DIR, "No_Fall", "Raw_Video")

    fall_videos = sorted([os.path.join(fall_dir, f) for f in os.listdir(fall_dir)
                          if f.lower().endswith(".mp4")])[:UTTEJ_PER_CLASS]
    no_fall_videos = sorted([os.path.join(no_fall_dir, f) for f in os.listdir(no_fall_dir)
                             if f.lower().endswith(".mp4")])[:UTTEJ_PER_CLASS]

    print(f"  낙상 {len(fall_videos)}개 + 정상 {len(no_fall_videos)}개", flush=True)
    start_time = time.time()

    # 진단 누적
    fn_diag = {"no_landmark": 0, "low_knee_vis": 0, "max_mp": [], "max_yolo": [],
               "p_lying": 0, "p_sitting": 0, "p_standing": 0, "p_unknown": 0}
    fp_diag = {"score_only": 0, "accident_only": 0, "both": 0,
               "p_lying": 0, "p_sitting": 0, "p_standing": 0, "p_unknown": 0}

    tp = fn = 0
    print("\n  [낙상 영상 처리]", flush=True)
    for i, vp in enumerate(fall_videos):
        try:
            result = process_video(vp, pose)
            if result is None:
                continue
            detected, diag = result
            if len(detected) > 0:
                tp += 1
            else:
                fn += 1
                # FN 진단 누적
                fn_diag["no_landmark"] += diag["no_landmark"]
                fn_diag["low_knee_vis"] += diag["low_knee_vis"]
                fn_diag["max_mp"].append(diag["max_mp_score"])
                fn_diag["max_yolo"].append(diag["max_yolo_score"])
                pd = diag["posture_dist_all"]
                fn_diag["p_lying"] += pd.get("lying", 0)
                fn_diag["p_sitting"] += pd.get("sitting", 0)
                fn_diag["p_standing"] += pd.get("standing", 0)
                fn_diag["p_unknown"] += pd.get("unknown", 0)
        except Exception as e:
            print(f"    [{i+1}] 실패: {e}", flush=True)
            continue
        if (i+1) % 10 == 0:
            sens = tp/(tp+fn)*100 if tp+fn else 0
            print(f"    낙상 {i+1:3d}/{len(fall_videos)} | TP:{tp} FN:{fn} (현재 민감도:{sens:.1f}%)", flush=True)

    tn = fp = 0
    print("\n  [정상 영상 처리]", flush=True)
    for i, vp in enumerate(no_fall_videos):
        try:
            result = process_video(vp, pose)
            if result is None:
                continue
            detected, diag = result
            if len(detected) > 0:
                fp += 1
                # FP 진단 누적
                dr = diag["detect_reasons"]
                fp_diag["score_only"] += dr.get("score_only", 0)
                fp_diag["accident_only"] += dr.get("accident_only", 0)
                fp_diag["both"] += dr.get("both", 0)
                pad = diag["posture_at_detect"]
                fp_diag["p_lying"] += pad.get("lying", 0)
                fp_diag["p_sitting"] += pad.get("sitting", 0)
                fp_diag["p_standing"] += pad.get("standing", 0)
                fp_diag["p_unknown"] += pad.get("unknown", 0)
            else:
                tn += 1
        except Exception as e:
            print(f"    [{i+1}] 실패: {e}", flush=True)
            continue
        if (i+1) % 10 == 0:
            spec = tn/(tn+fp)*100 if tn+fp else 0
            print(f"    정상 {i+1:3d}/{len(no_fall_videos)} | TN:{tn} FP:{fp} (현재 특이도:{spec:.1f}%)", flush=True)

    elapsed = time.time() - start_time
    total = tp+fn+tn+fp
    acc = (tp+tn)/total*100 if total else 0
    sens = tp/(tp+fn)*100 if tp+fn else 0
    spec = tn/(tn+fp)*100 if tn+fp else 0
    prec = tp/(tp+fp)*100 if tp+fp else 0
    f1 = (2*prec*sens/(prec+sens)) if prec+sens else 0

    print("\n" + "-" * 70, flush=True)
    print(f"uttejkumarkandagatla 종합 (처리 시간 {elapsed/60:.1f}분)", flush=True)
    print(f"  전체 정확도:           {acc:.1f}%", flush=True)
    print(f"  민감도 (낙상 감지율):  {sens:.1f}%", flush=True)
    print(f"  특이도 (정상 정확률):  {spec:.1f}%", flush=True)
    print(f"  정밀도 (신뢰도):       {prec:.1f}%", flush=True)
    print(f"  F1 Score:              {f1:.1f}", flush=True)
    print(f"  TP:{tp} FN:{fn} TN:{tn} FP:{fp}", flush=True)

    # FN 진단 (uttej에서 놓친 낙상)
    print(f"\n  [uttej FN 진단 - 왜 놓쳤나]", flush=True)
    if fn_diag["max_mp"]:
        avg_max_mp = sum(fn_diag["max_mp"]) / len(fn_diag["max_mp"])
        avg_max_yolo = sum(fn_diag["max_yolo"]) / len(fn_diag["max_yolo"])
        total_p = fn_diag["p_lying"] + fn_diag["p_sitting"] + fn_diag["p_standing"] + fn_diag["p_unknown"]
        print(f"  landmark 미검출 프레임 누적:  {fn_diag['no_landmark']}", flush=True)
        print(f"  knee vis<0.5 프레임 누적:      {fn_diag['low_knee_vis']}", flush=True)
        print(f"  FN 영상 평균 최고 MP 점수:    {avg_max_mp:.1f} (lying threshold=3)", flush=True)
        print(f"  FN 영상 평균 최고 YOLO 점수:  {avg_max_yolo:.1f}", flush=True)
        if total_p > 0:
            print(f"  FN 영상 자세 분포: lying={fn_diag['p_lying']/total_p*100:.0f}% "
                  f"sitting={fn_diag['p_sitting']/total_p*100:.0f}% "
                  f"standing={fn_diag['p_standing']/total_p*100:.0f}% "
                  f"unknown={fn_diag['p_unknown']/total_p*100:.0f}%", flush=True)

    # FP 진단 (uttej에서 정상 영상 오탐)
    print(f"\n  [uttej FP 진단 - 왜 오탐했나]", flush=True)
    total_r = fp_diag["score_only"] + fp_diag["accident_only"] + fp_diag["both"]
    if total_r > 0:
        print(f"  발동 원인: score만={fp_diag['score_only']} "
              f"accident만={fp_diag['accident_only']} "
              f"둘 다={fp_diag['both']}", flush=True)
        print(f"  → 'accident만' 많으면 자는 사람/누운 일상 자세가 사고로 오인", flush=True)
        print(f"  → 'score만' 많으면 lying threshold=3이 너무 낮음", flush=True)
        total_p = fp_diag["p_lying"] + fp_diag["p_sitting"] + fp_diag["p_standing"] + fp_diag["p_unknown"]
        if total_p > 0:
            print(f"  감지 시 자세: lying={fp_diag['p_lying']} "
                  f"sitting={fp_diag['p_sitting']} "
                  f"standing={fp_diag['p_standing']} "
                  f"unknown={fp_diag['p_unknown']}", flush=True)


if __name__ == "__main__":
    pose = mp.solutions.pose.Pose(model_complexity=0)
    print("=" * 70, flush=True)
    print("옵션 B 영상 기반 평가", flush=True)
    print("=" * 70, flush=True)
    print(f"설정:", flush=True)
    print(f"  Le2i  : 시나리오별 {LE2I_PER_SCENARIO}개 영상", flush=True)
    print(f"  uttej : 클래스별 {UTTEJ_PER_CLASS}개 영상", flush=True)
    print(f"  Detector: MediaPipe + YOLO + VideoMAE (Fix #4 적용)", flush=True)
    print(f"  Frame skip: 1/{FRAME_SKIP}, Duration: {DURATION_FRAMES}연속프레임", flush=True)
    print(f"  임계값: lying=3 sitting=5 standing=4 unknown=4 (main.py와 동일)", flush=True)
    print("=" * 70, flush=True)

    overall_start = time.time()
    evaluate_le2i(pose)
    evaluate_uttej(pose)
    print(f"\n총 처리 시간: {(time.time()-overall_start)/60:.1f}분", flush=True)

import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose

# ============================================================
# Tunable thresholds (그리드 서치 대상 - 데이터 기반 자동 튜닝)
# ============================================================
LYING_HORIZONTAL_RATIO = 0.55# 누움 판정 수평비 임계
LYING_HORIZONTAL_RATIO_TOO_CLOSE = 0.70 # 가까울 때 누움 판정 강화
LYING_SH_DIFF_MAX = 0.12# 가로 누움 + 어깨-엉덩이 y차 검증
LYING_SH_DIFF_VERY_SMALL = 0.06         # knee 가려져도 lying 인정 임계
UPRIGHT_TO_LYING_WINDOW = 2.0   # 자세 전환 보너스 시간창 (초) - Fix #3: 3.0→2.0 엄격화
TOO_CLOSE_SHOULDER_WIDTH = 0.35         # 카메라 가까움 판정

# 자세별 낙상 판정 임계값 (main.py와 evaluate*.py에서 공용)
# Fix #10 부분 롤백: 4→3 복원. VideoMAE LyingDown 0.85(Fix #9)만으로도 FP 충분히 차단됨
FALL_THRESHOLD_LYING = 3
FALL_THRESHOLD_SITTING = 5
FALL_THRESHOLD_STANDING = 4
FALL_THRESHOLD_UNKNOWN = 4


def get_threshold_for_posture(posture):
    """자세별 낙상 판정 임계값 (공용 함수)"""
    if posture == "lying":    return FALL_THRESHOLD_LYING
    if posture == "sitting":  return FALL_THRESHOLD_SITTING
    if posture == "standing": return FALL_THRESHOLD_STANDING
    return FALL_THRESHOLD_UNKNOWN


def set_sensitivity_mode(mode):
    """
    환경별 민감도 모드 전환 (가정/의료 양쪽 대응)
    - "home"     : 가정용 (현재 균형 모드, F1 최대)
    - "medical"  : 의료/요양용 (민감도 최우선, 절대 안 놓침)
    - "office"   : 사무실/일반 (정밀도 최우선, 오탐 적게)
    main.py에서 환경변수 또는 시작 시 호출: set_sensitivity_mode("medical")
    """
    global FALL_THRESHOLD_LYING, FALL_THRESHOLD_SITTING
    global FALL_THRESHOLD_STANDING, FALL_THRESHOLD_UNKNOWN

    if mode == "medical":
        # 의료/요양: 낙상 절대 안 놓침 (민감도↑, 오탐 약간 허용)
        FALL_THRESHOLD_LYING = 2
        FALL_THRESHOLD_SITTING = 4
        FALL_THRESHOLD_STANDING = 3
        FALL_THRESHOLD_UNKNOWN = 3
    elif mode == "office":
        # 사무실/일반: 오탐 최소화 (정밀도↑)
        FALL_THRESHOLD_LYING = 4
        FALL_THRESHOLD_SITTING = 6
        FALL_THRESHOLD_STANDING = 5
        FALL_THRESHOLD_UNKNOWN = 5
    else:  # "home" (default)
        # 가정용: 균형 (F1 최대)
        FALL_THRESHOLD_LYING = 3
        FALL_THRESHOLD_SITTING = 5
        FALL_THRESHOLD_STANDING = 4
        FALL_THRESHOLD_UNKNOWN = 4
# ============================================================

# 자세 추적 상태
prev_nose_y = None
prev_shoulder_y = None
prev_hip_y = None
prev_time = None
prev_velocity = 0

fall_start_time = None
stillness_start_time = None
last_posture = None
posture_start_time = None

# 미세 움직임 추적 (호흡, 고개 미세 움직임 등)
motion_history = []          # 최근 N프레임의 움직임 변화량
abnormal_posture_start = None  # 비정상 자세(고개 숙임/기울임) 지속 시간

# 자세 전환 추적 (자는 사람 vs 낙상 구분의 핵심)
last_upright_time = None     # 마지막으로 standing/sitting이었던 시간

# 외부에서 자세 상태 확인용
posture_status = {
    "posture": "unknown",
    "stillness_sec": 0,
    "velocity": 0,
    "abnormal": False,
    "micro_motion": 0,
}


def classify_posture(shoulder_y, hip_y, knee_y, nose_y, horizontal_ratio,
                     hip_visible=True, knee_visible=True, too_close=False):
    """
    전신/앉기/눕기 자세 분류 (Fix #1: lying 우선 + sitting 엄격화)
    낙상 영상의 49%가 sitting으로 잘못 분류되던 문제 해결.
    """
    if not hip_visible:
        return "unknown"

    shoulder_hip_diff = hip_y - shoulder_y      # 양수: 어깨가 위
    abs_sh_diff = abs(shoulder_hip_diff)

    # ============ lying 판정 (우선순위 최상) ============
    # (a) 가로로 누운 모습 + 어깨-엉덩이 거의 수평
    lying_threshold = LYING_HORIZONTAL_RATIO_TOO_CLOSE if too_close else LYING_HORIZONTAL_RATIO
    if horizontal_ratio > lying_threshold and abs_sh_diff < LYING_SH_DIFF_MAX:
        return "lying"

    # (b) 어깨-엉덩이 y차 매우 작음 (knee 가려져도 OK)
    if abs_sh_diff < LYING_SH_DIFF_VERY_SMALL:
        return "lying"

    # (c) NEW: horizontal_ratio가 매우 높으면 (몸이 명확히 가로) - abs_sh_diff 조건 완화
    # 옆으로 누운 낙상의 핵심 패턴
    if horizontal_ratio > 0.70:
        return "lying"

    # (d) NEW: 무릎이 엉덩이보다 위에 있음 = 옆/등으로 누움 (낙상 핵심 패턴)
    # 기존엔 이 케이스가 sitting으로 잘못 분류됨
    if knee_visible:
        if knee_y < hip_y - 0.03:  # 무릎이 엉덩이보다 위
            return "lying"

    # ============ knee 안 보임 ============
    if not knee_visible:
        if shoulder_hip_diff > 0.18:
            return "standing"
        # NEW: knee 안 보이고 abs_sh_diff 중간 → lying 추정 (예전엔 unknown으로 손실)
        if abs_sh_diff < 0.10:
            return "lying"
        return "unknown"

    hip_knee_diff = knee_y - hip_y              # 양수: 무릎이 아래

    # ============ standing ============
    if shoulder_hip_diff > 0.18 and hip_knee_diff > 0.15:
        return "standing"

    # ============ sitting (엄격화) ============
    # 기존: shoulder_hip_diff > 0.10 and hip_knee_diff < 0.12  ← 옆 낙상도 통과
    # 새것: 무릎이 엉덩이보다 "명확히" 약간 아래 (0.02 ~ 0.12)
    # 옆 낙상은 무릎이 엉덩이 옆/위라 hip_knee_diff <= 0.02 → 위에서 lying으로 잡힘
    if shoulder_hip_diff > 0.10 and 0.02 < hip_knee_diff < 0.12:
        return "sitting"

    return "unknown"


def detect_fall_mediapipe(landmarks, frame_height, current_time=None):
    """
    낙상 점수 계산. main.py는 실시간이라 current_time=None(time.time() 사용).
    evaluate_video.py는 영상 fps 기반 가짜 시간 주입 (Fix #5).
    """
    global prev_nose_y, prev_shoulder_y, prev_hip_y, prev_time, prev_velocity
    global fall_start_time, stillness_start_time, last_posture, posture_start_time
    global motion_history, abnormal_posture_start, last_upright_time

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    # ---- 핵심 관절 신뢰도(visibility) 체크 ----
    shoulder_vis = (left_shoulder.visibility + right_shoulder.visibility) / 2
    hip_vis = (left_hip.visibility + right_hip.visibility) / 2
    knee_vis = (left_knee.visibility + right_knee.visibility) / 2
    hip_visible = hip_vis > 0.5
    knee_visible = knee_vis > 0.5

    # 어깨조차 안 보이면 판정 불가
    if shoulder_vis < 0.5:
        posture_status["posture"] = "unknown"
        return 0

    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_x = (left_hip.x + right_hip.x) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    knee_y = (left_knee.y + right_knee.y) / 2

    # ---- 신체 크기로 카메라 거리 추정 ----
    # 어깨 너비가 프레임 가로의 몇 %를 차지하는지
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    # 어깨~엉덩이 수직 거리 (몸통 길이)
    torso_length = abs(shoulder_y - hip_y) if hip_visible else shoulder_width * 1.5
    body_scale = max(shoulder_width, torso_length * 0.5, 0.05)  # 0 나누기 방지

    # 어깨 너비가 화면의 N% 이상이면 "가까이 있음"
    too_close = shoulder_width > TOO_CLOSE_SHOULDER_WIDTH

    dx = abs(shoulder_x - hip_x)
    dy = abs(shoulder_y - hip_y)

    if dx + dy == 0:
        return 0

    horizontal_ratio = dx / (dx + dy)
    posture = classify_posture(
        shoulder_y, hip_y, knee_y, nose.y, horizontal_ratio,
        hip_visible=hip_visible, knee_visible=knee_visible, too_close=too_close
    )
    score = 0

    # ---- 시간 기반 계산 (Fix #5: 외부 시간 주입 가능) ----
    now = current_time if current_time is not None else time.time()
    dt = (now - prev_time) if prev_time is not None else 0.033
    prev_time = now

    # ---- 1. 가속도(빠른 낙하) 점수 ----
    # 픽셀 속도를 신체 크기로 정규화 → 카메라 거리에 무관한 비교 가능
    current_nose_y = nose.y * frame_height
    velocity = 0
    accel = 0
    norm_velocity = 0
    if prev_nose_y is not None and dt > 0:
        velocity = (current_nose_y - prev_nose_y) / dt   # px/sec, 양수: 아래로
        accel = (velocity - prev_velocity) / dt          # px/sec^2
        # 신체 크기로 정규화: "몸통 길이 단위/초"
        body_pixel_size = body_scale * frame_height
        norm_velocity = velocity / body_pixel_size if body_pixel_size > 0 else 0
        norm_accel = accel / body_pixel_size if body_pixel_size > 0 else 0

        # 정규화 속도 기준: 1초당 몸통 길이 1.5배 이상 떨어지면 빠른 낙하
        if norm_velocity > 1.5 and norm_accel > 2.5:
            score += 2
        elif norm_velocity > 0.9:
            score += 1
    prev_velocity = velocity
    prev_nose_y = current_nose_y

    # ---- 자세 전환 추적: 마지막 upright(서기/앉기) 시점 기록 ----
    # 진짜 낙상 = 직전 standing/sitting → 현재 lying
    # 자는 사람 = 처음부터 lying (upright 기록 없음)
    if posture in ("standing", "sitting"):
        last_upright_time = now

    # ---- 2. 누워있는 자세 점수 (Fix #2: 누적 점수 폭주 차단) ----
    # 기존: 기본+2, 코<무릎+1, 어깨-엉덩이+1, transition+2, 지속+2, stillness+2 = 최대 10점
    # 새것: 기본+2 + 강한 신호 한 가지만 +1 (누적 X)
    if posture == "lying":
        if too_close and not hip_visible:
            pass  # 가까운데 엉덩이도 안 보이면 신뢰 불가
        else:
            score += 2  # 기본 lying 점수
            strong_signal_added = False

            # Fix #3: transition 보너스 조건 엄격화
            # 기존: 단순히 최근 upright이면 +2 → 자세 분류 흔들림에 거짓 보너스
            # 새것: 최근 upright + 빠른 속도 동반시에만 +2 (진짜 빠른 낙하)
            if (last_upright_time is not None and
                (now - last_upright_time) < UPRIGHT_TO_LYING_WINDOW and
                norm_velocity > 0.5):
                score += 2
                strong_signal_added = True
            elif last_upright_time is None:
                # 영상/카메라 시작부터 lying = 자는 사람 가능성 → 감점 유지
                score -= 1

            # 완전 엎어진 상태 (코가 무릎보다 낮음) - 강한 신호 한 가지만
            if not strong_signal_added and knee_visible and nose.y > knee_y:
                score += 1
                strong_signal_added = True

            # Fix #8: 어깨-엉덩이 거의 같은 높이 (마지막 폴백) - 임계 0.08→0.12로 완화
            # 낙상 자세 더 자주 +1 추가되어 진짜 낙상의 점수가 임계값 안정적으로 통과
            if not strong_signal_added and hip_visible and abs(shoulder_y - hip_y) < 0.12:
                score += 1

    # ---- 3. 앉기/서기 자세에서는 낙상으로 보지 않음 ----
    # (단, 자세 유지 시간 검사를 위해 점수는 0 유지)

    # ---- 4. 비정상 자세 감지 (의식 잃은 앉기/서기 구분용) ----
    # (a) 고개 숙임: 코가 어깨선보다 아래로 떨어짐 (정상은 어깨 위)
    head_slump = nose.y > shoulder_y + 0.05
    # (b) 상체 좌우 기울기: 양쪽 어깨의 y 차이가 큼
    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    body_tilted = shoulder_tilt > 0.10
    # (c) 상체 앞으로 푹 숙임: 어깨-엉덩이 x 차이가 큼 (옆으로 봤을 때)
    forward_slump = abs(shoulder_x - hip_x) > 0.15 and posture == "sitting"

    abnormal_now = head_slump or body_tilted or forward_slump

    if abnormal_now:
        if abnormal_posture_start is None:
            abnormal_posture_start = now
    else:
        abnormal_posture_start = None

    # ---- 5. 미세 움직임(호흡 등) 추적 ----
    # 살아있는 사람은 완전한 0 정지가 거의 없음 -> 매우 작은 변화량도 누적해서 본다
    motion_amount = abs(velocity) + abs(shoulder_tilt - (motion_history[-1] if motion_history else shoulder_tilt)) * 1000
    motion_history.append(motion_amount)
    if len(motion_history) > 90:  # 약 3초 분량
        motion_history.pop(0)
    avg_micro_motion = sum(motion_history) / len(motion_history) if motion_history else 0

    # ---- 6. 자세 변화 추적 + 정지 시간 ----
    if posture != last_posture:
        last_posture = posture
        posture_start_time = now
        stillness_start_time = now
    else:
        # 매우 작은 움직임도 "살아있는 정지"로 인정 (호흡, 미세 흔들림)
        if avg_micro_motion < 5:   # 완전 정지에 가까움
            if stillness_start_time is None:
                stillness_start_time = now
        else:
            stillness_start_time = now

    # 누워있는 자세 유지 시간 (낙상 의심)
    # Fix #2: 누적 가산 차단을 위해 +2 → +1로 축소
    if posture == "lying" and posture_start_time is not None:
        lying_duration = now - posture_start_time
        if lying_duration >= 3.0:
            score += 1

    # ---- 7. 사고 의심 판정 (자세별 차등) ----
    stillness_sec = 0
    if stillness_start_time is not None:
        stillness_sec = now - stillness_start_time

    # 누움 + 10초 이상 거의 정지 → 강한 사고 신호 (영상에선 거의 발동 안 함)
    # Fix #2: +2 → +1
    if posture == "lying" and stillness_sec >= 10.0:
        score += 1

    # 앉기/서기에서는 "정지 시간" 단독으로는 사고 아님!
    # → 비정상 자세(고개 숙임/기울임/앞으로 푹 쓰러짐)가 동반되어야 사고로 본다
    if posture in ("sitting", "standing") and abnormal_posture_start is not None:
        abnormal_duration = now - abnormal_posture_start
        # 비정상 자세 5초 + 정지 상태 동반 → 의식 잃은 것으로 의심
        if abnormal_duration >= 5.0 and stillness_sec >= 5.0:
            score += 3
        # 비정상 자세가 15초 넘게 지속 → 명백한 사고
        elif abnormal_duration >= 15.0:
            score += 2

    # 외부 디버그용 상태 노출
    posture_status["posture"] = posture
    posture_status["stillness_sec"] = round(stillness_sec, 1)
    posture_status["velocity"] = round(velocity, 1)
    posture_status["abnormal"] = abnormal_now
    posture_status["micro_motion"] = round(avg_micro_motion, 2)
    posture_status["too_close"] = too_close
    posture_status["shoulder_width"] = round(shoulder_width, 2)
    posture_status["hip_visible"] = hip_visible
    posture_status["knee_visible"] = knee_visible

    return score

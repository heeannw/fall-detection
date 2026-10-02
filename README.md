# 🚨 Fall Detection System

> 실시간 낙상 감지 AI 시스템 — **[nuri (어르신 케어 서비스)](https://github.com/heeannw/nuri)** 의 낙상 알림 모듈로 실제 서비스에 연동

브라우저 WebRTC 카메라로 어르신의 낙상을 실시간 감지하고, Spring Boot 서버로 즉시 경보를 전송합니다.  
**YOLOv8 + XGBoost 앙상블 (F1 94.3)** 과 MediaPipe 포즈 추정을 결합한 2단계 탐지 파이프라인을 구현했습니다.

🤗 **HuggingFace Spaces 데모**: [heeannw/fall-detection](https://huggingface.co/spaces/heeannw/fall-detection)

---

## 📌 nuri 연동

| 항목 | 내용 |
|------|------|
| 연동 서비스 | nuri — 어르신 재가 케어 플랫폼 |
| 역할 | 낙상 감지 → Spring Boot Alert API 호출 |
| 엔드포인트 | `POST {FALL_ALERT_SPRING_URL}/api/alerts/fall` |
| 알림 페이로드 | `seniorId`, `score`, `notifyGuardian`, `escalationRequired` |
| 쿨다운 | 30초 (중복 알림 방지) |

낙상이 감지되면 `seniorId`와 신뢰 점수를 포함한 JSON을 Spring Boot로 비동기 전송합니다.  
보호자 알림(`notifyGuardian`)은 기본 활성화되며, 지속 낙상 시 `escalationRequired`가 자동 설정됩니다.

---

## 🏗 시스템 아키텍처

```
카메라 (WebRTC/FastRTC)
     ↓
┌────────────────────────────────────────────┐
│          Phase 1: 규칙 기반 탐지           │
│  • YOLOv8 (fall_detection_v3, custom fine-tune) │
│  • MediaPipe Pose (33 keypoints)           │
│  • 자세 이상 + 바운딩박스 비율 규칙        │
└──────────────┬─────────────────────────────┘
               │ 앙상블 (AND / OR / 단독)
┌──────────────▼─────────────────────────────┐
│        Phase 2: XGBoost ML 탐지            │
│  • 53 키포인트 × 3통계 = 159 features      │
│  • 8-frame sliding window buffer           │
│  • F1 94.3 (holdout test set)              │
│  • 임계값: 0.70 (조정 가능)               │
└──────────────┬─────────────────────────────┘
               │ 낙상 확정 (confirm_frames=3)
┌──────────────▼─────────────────────────────┐
│      Spring Boot Alert API (nuri)          │
│  POST /api/alerts/fall                     │
│  + SQLite 활동 로그 기록                   │
└────────────────────────────────────────────┘
```

---

## ⚙️ 앙상블 모드

`FALL_ENSEMBLE_MODE` 환경 변수로 제어합니다.

| 모드 | 동작 | 특징 |
|------|------|------|
| `or` (기본) | 규칙 OR XGBoost 중 하나라도 감지 시 | 높은 민감도, 누락 최소화 |
| `and` | 둘 다 동의해야 낙상 판정 | 오경보 최소화 |
| `ml_only` | XGBoost만 사용 | 규칙 없이 ML만 |
| `rule_only` | 규칙 기반만 사용 | 경량 환경용 |

---

## 🔍 Phase 2 XGBoost 모델 상세

- **특징 추출**: MediaPipe Pose 53 keypoints × 3 통계값 (mean, std, delta) = **159 features**
- **슬라이딩 윈도우**: 8프레임 버퍼로 시간적 변화 캡처
- **학습 데이터**: FallD 공개 데이터셋 (낙상 / 정상 영상)
- **성능**: F1 Score **94.3** (holdout test set)
- **임계값**: 기본 0.70 (`FALL_XGBOOST_THR`로 조정)
- **낙상 확정**: 3 연속 프레임(`FALL_CONFIRM_FRAMES`) 감지 시 최종 판정

---

## 📊 활동 로그 (Health Logger)

낙상 감지 외에도 실시간 활동 데이터를 SQLite에 누적하여 5가지 행동 지표를 산출합니다.

| 지표 | 설명 |
|------|------|
| 활동성 | 움직임 빈도 및 강도 |
| 안정성 | 자세 변동 안정성 |
| 휴식 균형 | 시간대별 활동/휴식 패턴 |
| 자세 상태 | 자세 이상 발생 빈도 |
| 안전도 | 낙상 위험 이벤트 발생률 |

14일 기준선을 쌓은 후 일간/시간대별 활동 리포트를 생성합니다.

---

## 🛠 기술 스택

| 영역 | 기술 |
|------|------|
| 객체 탐지 | YOLOv8n-pose (Ultralytics, custom fine-tune) |
| 포즈 추정 | MediaPipe Pose (33 keypoints) |
| ML 분류 | XGBoost (159 features, F1 94.3) |
| 실시간 스트리밍 | FastRTC / WebRTC |
| 웹 UI | Gradio |
| 배포 | HuggingFace Spaces (ZeroGPU) |
| 백엔드 연동 | Spring Boot REST API (nuri) |
| 추가 하드웨어 | Arduino (IMU 센서, serial/wifi 연동) |

---

## 🚀 로컬 실행

### 사전 준비

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일에서 FALL_ALERT_SPRING_URL, FALL_SENIOR_ID 설정
```

### 환경 변수 (`.env.example`)

```env
FALL_ALERT_SPRING_URL=https://your-spring-server.example.com  # nuri Spring Boot URL
FALL_SENIOR_ID=53                  # 어르신 ID
FALL_ALERT_API_KEY=                # Bearer 인증 (선택)
FALL_ALERT_COOLDOWN_SEC=30         # 알림 쿨다운 (초)
FALL_XGBOOST_THR=0.70              # XGBoost 낙상 임계값
FALL_ENSEMBLE_MODE=or              # 앙상블 모드
HF_TOKEN=hf_...                    # HuggingFace TURN 릴레이용
```

### 실행

```bash
# WebRTC 브라우저 모드 (Gradio UI)
python web_app.py

# 로컬 카메라 직접 모드
python main.py
```

---

## 🤗 HuggingFace Spaces 배포

ZeroGPU 환경에서 Gradio SDK로 배포합니다.  
WebRTC/FastRTC를 통해 브라우저 카메라를 직접 스트리밍하므로 별도 설치 없이 사용 가능합니다.

```bash
# HuggingFace Spaces에 필요한 패키지
# requirements-space.txt / packages.txt 참조
```

---

## 🔗 관련 프로젝트

- **[nuri](https://github.com/heeannw/nuri)** — 어르신 케어 서비스 (이 모델의 낙상 알림 실사용처)
- **[health_model](https://github.com/heeannw/health_model)** — nuri 복지사 페이지 건강 상태 분류 모델
